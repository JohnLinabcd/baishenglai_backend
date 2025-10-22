import warnings

import pandas as pd
import pynvml
import torch
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view
from rest_framework_simplejwt.tokens import AccessToken
from algorithm.drug_property.main import drug_property_prediction
from djcelery.sms.tasks import *


# Create your views here.
@api_view(['POST'])
def test(request):
    # print(request.user)
    param_dict = {
        "task_param_dict": {
            # "user_email_id": request.session.get("user_email"),
            "task_type": 1,
        },
        "model_param_dict": {
            "mobile": 114514,
        }
    }

    # result = send_sms.delay(param_dict)
    return HttpResponse("OK")


def time_consuming_task(request):
    param_dict = {
        "task_param_dict": {
            "user_email_id": request.session.get("user_email"),
            "task_type": 1,
        },
        "model_param_dict": {
            "mobile": 114514,
            "sleep_time": 10,
        }
    }
    time_consuming_send_sms.delay(param_dict)
    return HttpResponse("OK")


def allocate_cuda(task_type):
    """
    显卡调度分配函数
    结合显存使用情况智能分配算法模型到显卡
    目前实现 检索所有可用显卡，选择剩余显存最多的显卡分配任务
    """
    if torch.cuda.is_available():
        pynvml.nvmlInit()  # 初始化 NVML
        device_count = pynvml.nvmlDeviceGetCount()

        max_free_mem = 0
        best_gpu_index = 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mem = mem_info.free
                if free_mem > max_free_mem:
                    max_free_mem = free_mem
                    best_gpu_index = i
            except pynvml.NVMLError as e:
                return 'cpu'

        pynvml.nvmlShutdown()  # 清理资源
        return f'cuda:{best_gpu_index}'
    return 'cpu'


def get_target_list(request):
    """
    获取药物靶体列表函数
    前端提供给用户选择药物靶体
    """
    if TARGET_LIST is not None:
        return JsonResponse({
            'code': 200,
            'msg': '药物靶体列表获取成功',
            'data': TARGET_LIST
        })
    else:
        return JsonResponse({
            'code': 500,
            'msg': '药物靶体列表获取失败，数据文件读取错误',
        })


def get_cell_list(request):
    """
    获取细胞系列表函数
    前端提供给用户选择细胞系
    """
    if CELL_LIST is not None:
        return JsonResponse({
            'code': 200,
            'msg': '细胞系列表获取成功',
            'data': CELL_LIST
        })
    else:
        return JsonResponse({
            'code': 500,
            'msg': '细胞系列表获取失败，数据文件读取错误',
        })


def get_herb_name_list(request):
    if HERB_NAME_LIST is not None:
        return JsonResponse({
            'code': 200,
            'msg': '中药材名称获取成功',
            'data': HERB_NAME_LIST
        })
    else:
        return JsonResponse({
            'code': 500,
            'msg': '中药材名称获取失败，数据文件读取错误',
        })
