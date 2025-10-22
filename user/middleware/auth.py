import json

from django.http import JsonResponse
from django.shortcuts import redirect
from django.utils.deprecation import MiddlewareMixin
import jwt
from rest_framework_simplejwt.tokens import AccessToken

from user.models import UserModel
from config import AUTH_WITHOUT_URL_LIST

# class AuthMiddleware(MiddlewareMixin):
#     def process_request(self, request):
#         if request.path_info in AUTH_WITHOUT_URL_LIST:
#             return
#
#         user_dict = request.session.get('user_email')
#         if user_dict:
#             return
#
#         return redirect('/user/login/')


class AuthMiddleware(MiddlewareMixin):
    def process_request(self, request):
        """
        鉴权函数
        框架自带鉴权函数不能很好配合前后端通信需求，因此在中间件层自定义鉴权过程
        除了一些用户无关的操作（例如激活码获取、登录、注册等）其余业务操作都需要用户鉴权
        也就是其他操作都需要传递token，所以空请求或不带token的请求不被允许
        获取token之后，使用AccessToken可以进行解码、有效性认证和时效性认证
        获取decoded_token之后，提取其中的user_email，再在数据库中检查是否存在该用户
        全部检查完毕后完成鉴权
        @param request:
        @return:
        """
        # 跳过用户无关url
        if request.path_info in AUTH_WITHOUT_URL_LIST:
            return

        # 获取post的json
        try:
            post_data = json.loads(request.body.decode('utf-8'))
        except json.decoder.JSONDecodeError:
            return JsonResponse({
                'code': 500,
                'msg': '请求json为空',
            })

        # 获取json中的token
        token = post_data.get('token')
        if token is None:
            return JsonResponse({
                'code': 500,
                'msg': '需要传递token',
            })

        # 检查token有效性
        try:
            decoded_token = AccessToken(token)
        except Exception:
            return JsonResponse({
                'code': 400,
                'msg': 'token无效或过期',
            })

        # 检查token的用户合法性（这个操作可能比较耗时，尤其是在用户多的情况下，检查用户合法性其实不一定必须）
        user_email = decoded_token.payload.get('user_email')
        try:
            UserModel.objects.get(user_email=user_email)
            return
        except UserModel.DoesNotExist:
            return JsonResponse({
                'code': 300,
                'msg': 'token中用户无效',
            })
