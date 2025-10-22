import json
from datetime import timedelta

from django.forms import ModelForm
from django import forms
from django.http import JsonResponse, HttpResponse
from django.shortcuts import redirect
from django.utils.timezone import *
from rest_framework.decorators import api_view
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken

from user.models import UserModel

from djcelery.sms.tasks import *


# Create your views here.


class userLoginForm(forms.Form):
    user_email = forms.EmailField()
    password = forms.CharField(max_length=255)


@api_view(['POST'])
def send_register_email(request):
    """
    用户注册激活码发送视图函数
    需要先检查用户是否已经注册
    如果用户邮箱未被注册，调用异步发送函数，向前端传递的用户邮箱发送6位随机数字激活码，不检查用户邮箱是否实际存在
    同样在异步函数中，将用户邮箱、激活码、发送时间和有效标志符保存在数据表中
    发送新的激活码邮件的同时，删除历史所有旧激活码（无论是否有效）
    @param request:
    @return:
    """
    post_data = json.loads(request.body.decode('utf-8'))
    user_email = post_data['user_email']
    if UserModel.objects.filter(user_email=user_email).exists():
        return JsonResponse({
            'code': 500,
            'msg': '用户邮箱已被注册',
        })

    # 检查是否有历史激活码，如果历史激活码发送间隔过短，则不发送新的激活码，并返回剩余秒数；如果超过间隔，则删除历史激活码，添加新的激活码
    history_activation_code = UserActivationModel.objects.filter(user_email=user_email)
    if history_activation_code:
        send_time = make_aware(history_activation_code.first().send_time, timezone=get_current_timezone())
        current_time = make_aware(timezone.now(), timezone=get_current_timezone())

        interval_time = timedelta(minutes=1)
        if current_time < send_time + interval_time:
            return JsonResponse({
                'code': 500,
                'msg': '激活码重复获取',
                'data': {
                    'second_left': (interval_time + send_time - current_time).seconds
                }
            })
        history_activation_code.delete()
    async_send_register_email.delay(user_email=user_email)

    return JsonResponse({
        'code': 200,
        'msg': '激活码发送成功',
    })


@api_view(['POST'])
def user_register(request):
    """
    用户注册函数
    接收前端传递的用户邮箱、密码（已被md5加密）和激活码
    对邮箱和激活码进行检查，如果不存在该表项则返回激活码不正确
    如果存在，检查激活码是否过期，如果超时将超时表项删除，并返回过期错误
    如果激活码不过期，则创建用户数据表项，返回注册成功，并删除该用户所有激活码表项

    用户邮箱、密码合法性交由前端检查
    @param request:
    @return:
    """
    post_data = json.loads(request.body.decode('utf-8'))
    user_email = post_data['user_email']
    user_password = post_data['password']
    activation_code = post_data['activation_code']

    # 检查邮箱和激活码是否正确
    try:
        user_activation_obj = UserActivationModel.objects.get(user_email=user_email,
                                                              activation_code=activation_code)
    except UserActivationModel.DoesNotExist:
        return JsonResponse({
            'code': 500,
            'msg': '邮箱或激活码无效',
        })

    # 检查激活码是否过期
    send_time = make_aware(user_activation_obj.send_time, timezone=get_current_timezone())
    current_time = make_aware(timezone.now(), timezone=get_current_timezone())

    valid_time = timedelta(minutes=5)
    if current_time - send_time > valid_time:
        user_activation_obj.delete()
        return JsonResponse({
            'code': 500,
            'msg': '激活码已过期，请重新获取',
        })

    # 创建用户表项并删除用户所有激活码表项
    UserModel.objects.create(user_email=user_email, password=user_password)
    UserActivationModel.objects.filter(user_email=user_email).delete()

    return JsonResponse({
        'code': 200,
        'msg': '注册成功',
    })


@api_view(['POST'])
def user_login(request):
    """
    用户登录函数
    检查用户账户密码是否在用户表中
    此外会返回用户token
    @param request:
    @return:
    """
    user_login_form = userLoginForm(json.loads(request.body.decode('utf-8')))
    if user_login_form.is_valid():
        user_object = UserModel.objects.filter(**user_login_form.cleaned_data).first()
        if user_object:
            refresh = RefreshToken.for_user(user_object)
            access_token = refresh.access_token
            return JsonResponse({
                'code': 200,
                'msg': '登陆成功',
                'data': {
                    'user': user_object.user_email,
                    'token': str(access_token)
                }
            })

    return JsonResponse({
        'code': 500,
        'msg': '登录失败，账户或密码错误',
    })


@api_view(['POST'])
def send_reset_password_email(request):
    """
    用户密码重置激活码发送视图函数
    需要先检查用户是否已经注册
    如果用户邮箱存在，调用异步发送函数，向前端传递的用户邮箱发送6位随机数字激活码，不检查用户邮箱是否实际存在
    同样在异步函数中，将用户邮箱、激活码、发送时间保存在数据表中
    发送新的激活码邮件的同时，删除历史所有旧激活码（无论是否有效）
    @param request:
    @return:
    """
    post_data = json.loads(request.body.decode('utf-8'))
    user_email = post_data['user_email']
    if not UserModel.objects.filter(user_email=user_email).exists():
        return JsonResponse({
            'code': 500,
            'msg': '用户邮箱无效',
        })

    # 检查是否有历史激活码，如果历史激活码发送间隔过短，则不发送新的激活码，并返回剩余秒数；如果超过间隔，则删除历史激活码，添加新的激活码
    history_activation_code = UserActivationModel.objects.filter(user_email=user_email)
    if history_activation_code:
        send_time = make_aware(history_activation_code.first().send_time, timezone=get_current_timezone())
        current_time = make_aware(timezone.now(), timezone=get_current_timezone())

        interval_time = timedelta(minutes=1)
        if current_time < send_time + interval_time:
            return JsonResponse({
                'code': 500,
                'msg': '激活码重复获取',
                'data': {
                    'second_left': (interval_time + send_time - current_time).seconds
                }
            })
        history_activation_code.delete()
    async_send_reset_password_email.delay(user_email=user_email)

    return JsonResponse({
        'code': 200,
        'msg': '激活码发送成功',
    })


@api_view(['POST'])
def user_reset_password(request):
    """
    用户密码重置函数
    @param request:
    @return:
    """
    post_data = json.loads(request.body.decode('utf-8'))
    user_email = post_data['user_email']
    new_password = post_data['new_password']
    activation_code = post_data['activation_code']

    # 检查邮箱和激活码是否正确
    try:
        user_activation_obj = UserActivationModel.objects.get(user_email=user_email,
                                                              activation_code=activation_code)
    except UserActivationModel.DoesNotExist:
        return JsonResponse({
            'code': 500,
            'msg': '邮箱或激活码无效',
        })

    # 检查激活码是否过期
    send_time = make_aware(user_activation_obj.send_time, timezone=get_current_timezone())
    current_time = make_aware(timezone.now(), timezone=get_current_timezone())

    valid_time = timedelta(minutes=5)
    if current_time - send_time > valid_time:
        user_activation_obj.delete()
        return JsonResponse({
            'code': 500,
            'msg': '激活码已过期，请重新获取',
        })

    user_obj = UserModel.objects.get(user_email=user_email)
    user_obj.password = new_password
    user_obj.save()
    UserActivationModel.objects.filter(user_email=user_email).delete()
    return JsonResponse({
        'code': 200,
        'msg': '密码修改成功',
    })


@api_view(['GET'])
def get_user_num(request):
    try:
        user_num = UserModel.objects.count()
        return JsonResponse({
            'code': 200,
            'msg': '获取用户人数成功',
            'data': {
                'user_num': user_num
            }
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'msg': '获取用户人数失败'
        })
