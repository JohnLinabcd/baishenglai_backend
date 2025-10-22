# Create your views here.
import csv
import os

from django.http import JsonResponse, HttpResponse
from django.core.cache import cache
from rest_framework.decorators import api_view
from rest_framework_simplejwt.tokens import AccessToken

from task.models import TaskModel
from config import *
from djcelery.celery import app
from django.utils import timezone
import json
from task.utils import *


@api_view(['POST'])
def task_list(request):
    post_data = json.loads(request.body.decode('utf-8'))
    task_type = post_data['task_type']
    token = post_data['token']
    user_email = AccessToken(token).payload.get('user_email')
    if task_type == 0:
        tasks = TaskModel.objects.filter(user_email_id=user_email)
    else:
        tasks = TaskModel.objects.filter(user_email_id=user_email, task_type=task_type)

    task_json_list = []
    for task in tasks:
        task_data = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "submit_time": task.submit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "finish_time": task.finish_time.strftime("%Y-%m-%d %H:%M:%S") if task.finish_time else "",
            "predicted_finish_time": task.predicted_finish_time.strftime(
                "%Y-%m-%d %H:%M:%S") if task.predicted_finish_time else "",
            "task_status": task.task_status
        }
        task_json_list.append(task_data)
    return JsonResponse({
        "code": 200,
        "msg": "任务记录列表获取成功",
        "data": {
            "task_num": len(task_json_list),
            "task_json_list": task_json_list
        }

    })


@api_view(['POST'])
def task_detail(request):
    post_data = json.loads(request.body.decode('utf-8'))
    task_id = post_data['task_id']
    token = post_data['token']
    user_email = AccessToken(token).payload.get('user_email')
    try:
        task_object = TaskModel.objects.get(task_id=task_id, user_email_id=user_email)
    except TaskModel.DoesNotExist:
        return JsonResponse({
            "code": 500,
            "msg": "获取失败，任务不存在",
        })

    # 药物生成算法和其他算法结果处理不相同，单独处理
    task_type = task_object.task_type
    result_path = generate_result_path(user_email, task_type, task_id)
    if result_path is None:
        return JsonResponse({
            "code": 500,
            "msg": "获取失败，任务结果文件丢失",
        })

    if task_type == DRUG_GENERATION:
        header = 'gen_smiles'
        result_list = read_json_file(result_path, "gen_smiles")
    elif task_type == HERB_DRUG_INTERACTION:
        header = ''
        result_list = read_json_file(result_path)
    elif task_type == HERB_HERB_INTERACTION:
        header = ''
        result_list = get_hhi_details(result_path)
    else:
        result_list = []
        # 如果是药物生成路径规划，只取前三列结果
        # 如果是药物细胞系反应，则需要进行换算
        with open(result_path, "r") as f:
            csv_reader = csv.reader(f)
            if task_object.task_type == DRUG_SYNTHESIS_DESIGN:
                header = next(csv_reader)[:3]
            else:
                header = next(csv_reader)
            for i, row in enumerate(csv_reader):
                if i >= NUM_ITEMS_TO_READ:
                    break
                if task_object.task_type == DRUG_SYNTHESIS_DESIGN:
                    result_list.append(row[:3])
                elif task_object.task_type == DRUG_CELL_RESPONSE_PREDICTION:
                    row[-1] = process_drp_results(row[-1])
                    result_list.append(row)
                else:
                    result_list.append(row)

    return JsonResponse({
        "code": 200,
        "msg": "任务详情获取成功",
        "data": {
            "header": header,
            "result_list": result_list,
        }
    })


@api_view(['POST'])
def task_revoke(request):
    """
    任务撤销函数，对传递的指定任务进行撤销；会检查任务id是否存在，任务是否正在等待或运行，
    如果任务等待或者运行，则对任务进行撤销。由于撤销过程不算任务结束，且没有找到合适对应的回调函数，因此也会在该函数中修改
    数据库中任务状态。
    @param request:
    @return:
    """
    post_data = json.loads(request.body.decode('utf-8'))
    task_id = post_data['task_id']
    token = post_data['token']
    user_email = AccessToken(token).payload.get('user_email')
    try:
        task_obj = TaskModel.objects.get(task_id=task_id, user_email_id=user_email)
    except TaskModel.DoesNotExist:
        return JsonResponse({
            "code": 500,
            "msg": "撤销失败，任务不存在",
        })

    if task_obj.task_status != PENDING and task_obj.task_status != RUNNING:
        return JsonResponse({
            "code": 500,
            "msg": "撤销失败，任务已结束或终止",
        })

    app.control.revoke(task_id, terminate=True)
    task_obj.task_status = CANCELLED
    task_obj.finish_time = timezone.now()
    task_obj.save()
    return JsonResponse({
        "code": 200,
        "msg": "任务撤销成功",
    })


@api_view(['POST'])
def task_delete(request):
    """
    任务记录删除函数，对传递的指定任务进行删除；会检查任务id是否存在，会检查任务id是否已经结束运行；
    除了删除任务记录的同时，也会删除任务的结果文件（如果存在的话）
    @param request:
    @return:
    """
    post_data = json.loads(request.body.decode('utf-8'))
    task_id = post_data['task_id']
    token = post_data['token']
    user_email = AccessToken(token).payload.get('user_email')
    try:
        task_obj = TaskModel.objects.get(task_id=task_id, user_email_id=user_email)
    except TaskModel.DoesNotExist:
        return JsonResponse({
            "code": 500,
            "msg": "删除失败，任务不存在",
        })

    if task_obj.task_status == PENDING or task_obj.task_status == RUNNING:
        return JsonResponse({
            "code": 500,
            "msg": "删除失败，任务正在运行",
        })
    task_type = task_obj.task_type
    task_obj.delete()

    task_result_path = generate_result_path(user_email, task_type, task_id)
    if os.path.exists(task_result_path):
        os.remove(task_result_path)

    return JsonResponse({
        "code": 200,
        "msg": "任务删除成功"
    })


@api_view(['POST'])
def task_download(request):
    """
    任务结果下载函数，提供任务csv文件下载
    @param request:
    @return:
    """
    post_data = json.loads(request.body.decode('utf-8'))
    task_id = post_data['task_id']
    token = post_data['token']
    user_email = AccessToken(token).payload.get('user_email')
    try:
        task_obj = TaskModel.objects.get(task_id=task_id, user_email_id=user_email)
    except TaskModel.DoesNotExist:
        return JsonResponse({
            "code": 500,
            "msg": "下载失败，任务不存在",
        })

    if task_obj.task_status == PENDING or task_obj.task_status == RUNNING:
        return JsonResponse({
            "code": 500,
            "msg": "下载失败，任务未结束",
        })

    task_type = task_obj.task_type
    result_path = generate_result_path(user_email, task_type, task_id)
    if result_path is None:
        return JsonResponse({
            "code": 500,
            "msg": "获取失败，任务结果文件丢失",
        })
    filename = os.path.basename(result_path)
    if os.path.exists(result_path):
        with open(result_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response


@api_view(['GET'])
def get_access_records(request):
    return JsonResponse({
        "code": 200,
        "msg": "获取访问记录成功",
        "data": {
            ACCESS_TIMES: cache.get(ACCESS_TIMES),
            TASK_TIMES: cache.get(TASK_TIMES)
        }
    })


@api_view(['GET'])
def add_access_records(request):
    try:
        cache.incr(ACCESS_TIMES)
        return JsonResponse({
            "code": 200,
            "msg": "添加访问记录成功",
            "data": {
                ACCESS_TIMES: cache.get(ACCESS_TIMES)
            }
        })
    except Exception as e:
        return JsonResponse({
            "code": 500,
            "msg": "添加访问记录失败",
        })


@api_view(['GET'])
def get_online_users_num(request):
    return JsonResponse({
        "code": 200,
        "msg": "获取在线人数成功",
        "data": {
            ONLINE_USERS_NUM: cache.get(ONLINE_USERS_NUM)
        }
    })


@api_view(['GET'])
def add_online_users_num(request):
    try:
        cache.incr(ONLINE_USERS_NUM)
        return JsonResponse({
            "code": 200,
            "msg": "添加在线人数成功",
            "data": {
                ONLINE_USERS_NUM: cache.get(ONLINE_USERS_NUM)
            }
        })
    except Exception as e:
        return JsonResponse({
            "code": 500,
            "msg": "添加在线人数失败"
        })


@api_view(['GET'])
def decline_online_users_num(request):
    try:
        if cache.get(ONLINE_USERS_NUM) > 0:
            cache.incr(ONLINE_USERS_NUM, -1)
        return JsonResponse({
            "code": 200,
            "msg": "减少在线人数成功",
            "data": {
                ONLINE_USERS_NUM: cache.get(ONLINE_USERS_NUM)
            }
        })
    except Exception as e:
        return JsonResponse({
            "code": 500,
            "msg": "减少在线人数失败"
        })
