# celery的任务必须写在tasks.py的文件中，别的文件名称不识别!!!
import csv
import os
import sys
from random import randint
import json

import numpy as np
import yaml

from config import *
from celery import Task
from celery import current_task
from django.utils import timezone
from django.core.mail import send_mail
from baishenglai_backend.settings import EMAIL_FROM

sys.path.append(".")

from djcelery.celery import app
from django.core.cache import cache
from django.utils.timezone import now
import time
import logging
from task.models import TaskModel
from user.models import UserActivationModel, UserModel
from config import MODEL_DICT

log = logging.getLogger("django")


class CustomTask(Task):
    def before_start(self, task_id, args, kwargs):
        """
        celery异步任务的执行前回调函数，在每个异步任务之前执行，用来添加任务表项
        包括添加task_id，task_status（默认为等待中），提交任务的用户名
        @param task_id: 由celery将异步任务的task_id传递，作为异步任务的主键使用
        @param args: 将传进异步任务的全部参数包裹成dict，再套一个[]
        @param kwargs:空
        @return:
        """
        super(CustomTask, self).before_start(task_id, args, kwargs)
        arg_dict = args[0]
        task_param_dict = arg_dict["task_param_dict"]
        task_param_dict["task_id"] = task_id
        task_param_dict["predicted_finish_time"] = predict_finish_time(task_type=task_param_dict["task_type"],
                                                                       model_param_dict=arg_dict["model_param_dict"])
        new_task = TaskModel.objects.create(**task_param_dict)
        new_task.save()
        cache.incr(TASK_TIMES)

    def on_success(self, retval, task_id, args, kwargs):
        super(CustomTask, self).on_success(retval, task_id, args, kwargs)
        change_task_status(task_id=task_id, status=COMPLETED)
        return

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        super(CustomTask, self).on_failure(exc, task_id, args, kwargs, einfo)
        change_task_status(task_id=task_id, status=FAILED)
        return

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """
        该回调函数在任务执行完毕返回前调用，无论任务是否顺利执行，用来
        1. 设置任务结果短信通知功能
        2. 设置任务表中任务结束时间
        @param status:
        @param retval:
        @param task_id:
        @param args:
        @param kwargs:
        @param einfo:
        @return:
        """
        task_model_obj = TaskModel.objects.filter(task_id=task_id).first()
        task_model_obj.finish_time = timezone.now()
        task_model_obj.save()
        send_task_finish_email(task_model_obj)
        return super(CustomTask, self).after_return(status, retval, task_id, args, kwargs, einfo)


def change_task_status(task_id, status):
    task_model_obj = TaskModel.objects.filter(task_id=task_id).first()
    task_model_obj.task_status = status
    task_model_obj.save()
    return


def predict_finish_time(task_type, model_param_dict):
    """
    计算任务预期结束时间
    除了药物生成以外的任务，预期耗时都是5min
    对药物生成任务，基本上符合 预期耗时 = 0.045min * generation_num + 15min，可以给一倍的预期冗余
    """
    if task_type == DRUG_GENERATION:
        task_time_consuming = model_param_dict['timestep'] * 0.1 + 30
        predicted_finish_time = timezone.now() + timezone.timedelta(minutes=task_time_consuming)
    else:
        predicted_finish_time = timezone.now() + timezone.timedelta(minutes=5)
    return predicted_finish_time


@app.task(base=CustomTask)
def send_sms(param_dict):
    """
    测试用函数1
    """
    task_id = current_task.request.id
    change_task_status(task_id=task_id, status=RUNNING)
    model_param_dict = param_dict["model_param_dict"]
    print("向手机号%s发送短信成功!" % model_param_dict["mobile"])
    return "send_sms OK"


@app.task
def time_consuming_send_sms(param_dict):
    """
    测试用函数2
    """
    task_id = current_task.request.id
    # change_task_status(task_id=task_id, status=RUNNING)
    model_param_dict = param_dict["model_param_dict"]
    sleep_time = model_param_dict["sleep_time"]
    time.sleep(sleep_time)
    # time.sleep(sleep_time)
    for i in range(10):
        print(i)
    print("向手机号%s发送短信成功!" % model_param_dict["mobile"])
    return "send_sms OK"


@app.task
def async_send_register_email(user_email):
    """
    注册激活码邮件发送函数
    """
    activation_code_len = 6
    activation_code = ''
    for i in range(activation_code_len):
        activation_code += str(randint(0, 9))

    email_subject = '百生来 | 注册激活码'
    email_content = f'欢迎使用百生来平台，您的注册激活码为：{activation_code}，激活码五分钟内有效，请勿回复。'
    send_mail(email_subject, email_content, EMAIL_FROM, [user_email])

    # 保存发送记录在数据库中
    UserActivationModel.objects.create(user_email=user_email, activation_code=activation_code)
    return


@app.task
def async_send_reset_password_email(user_email):
    """
    密码修改激活码邮件发送函数
    """
    activation_code_len = 6
    activation_code = ''
    for i in range(activation_code_len):
        activation_code += str(randint(0, 9))

    email_subject = '百生来 | 密码重置激活码'
    email_content = f'欢迎使用百生来平台，您的密码重置激活码为：{activation_code}，激活码五分钟内有效，请勿回复。'
    send_mail(email_subject, email_content, EMAIL_FROM, [user_email])

    # 保存发送记录在数据库中
    UserActivationModel.objects.create(user_email=user_email, activation_code=activation_code)
    return


def send_task_finish_email(task_model_obj):
    """
    任务结束通知邮件发送函数
    """
    user_email = task_model_obj.user_email_id
    submit_time = task_model_obj.submit_time.strftime('%Y-%m-%d %H:%M:%S')
    finish_time = task_model_obj.finish_time.strftime('%Y-%m-%d %H:%M:%S')
    task_type = TASK_TYPE_CHOICES[task_model_obj.task_type]
    task_status = TASK_STATUS_CHOICES[task_model_obj.task_status]
    task_url = f"{DOMAIN_HTTP}/#/home/{TASK_TYPE_URL[task_model_obj.task_type]}?key=history"
    email_subject = '百生来 | 任务结束通知'
    email_content = (f'用户{user_email}，您好！\n\n'
                     f'欢迎使用百生来平台，您提交的计算任务已结束。\n\n'
                     f'任务类型：\t{task_type}\n'
                     f'任务编号：\t{task_model_obj.task_id}\n'
                     f'任务提交时间：\t{submit_time}\n'
                     f'任务结束时间：\t{finish_time}\n'
                     f'任务状态：\t{task_status}\n\n'
                     f'访问 {task_url} 以检查任务结果，请勿回复。')
    send_mail(email_subject, email_content, EMAIL_FROM, [user_email])
    return


@app.task(base=CustomTask)
def async_drug_cell_response_regression_predict(param_dict):
    """
    药物-细胞器反应回归预测异步任务
    @param param_dict:
    @return:
    """
    task_id = current_task.request.id
    change_task_status(task_id=task_id, status=RUNNING)
    model_param_dict = param_dict["model_param_dict"]
    task_param_dict = param_dict["task_param_dict"]
    user_email = task_param_dict["user_email_id"]
    drug_smiles = model_param_dict["drug_smiles"]
    cell_name = model_param_dict["cell_name"]

    model_name = param_dict["model_name"]
    task_type = task_param_dict["task_type"]
    drug_target_affinity_regression_predict = MODEL_DICT[task_type][model_name]
    response_value = drug_target_affinity_regression_predict(**model_param_dict)

    result_data = [
        ['drug_smiles', 'cell_name', 'IC50_value']
    ]
    result_data += [[drug_smiles[i], cell_name, response_value[i]] for i in range(len(drug_smiles))]
    save_task_result(user_email=user_email, task_id=task_id, result_data=result_data)

    return


@app.task(base=CustomTask)
def async_drug_target_affinity_regression_predict(param_dict):
    """
    药物-靶体亲和度回归预测异步任务
    @param param_dict:
    @return:
    """
    task_id = current_task.request.id
    change_task_status(task_id=task_id, status=RUNNING)
    model_param_dict = param_dict["model_param_dict"]
    task_param_dict = param_dict["task_param_dict"]
    user_email = task_param_dict["user_email_id"]
    drug_smiles = model_param_dict["drug_smiles"]
    target_seq = model_param_dict["target_seq"]

    model_name = param_dict["model_name"]
    task_type = task_param_dict["task_type"]
    drug_target_affinity_regression_predict = MODEL_DICT[task_type][model_name]
    affinity_value = drug_target_affinity_regression_predict(**model_param_dict)

    result_data = [
        ['drug_smiles', 'target_seq', 'affinity_value']
    ]
    result_data += [[drug_smiles[i], target_seq, affinity_value[i]] for i in range(len(drug_smiles))]
    save_task_result(user_email=user_email, task_id=task_id, result_data=result_data)

    return


@app.task(base=CustomTask)
def async_drug_target_classification_regression_predict(param_dict):
    """
    药物-靶体亲和度分类预测异步任务
    @param param_dict:
    @return:
    """
    task_id = current_task.request.id
    change_task_status(task_id=task_id, status=RUNNING)

    model_param_dict = param_dict["model_param_dict"]
    task_param_dict = param_dict["task_param_dict"]
    user_email = task_param_dict["user_email_id"]
    drug_smiles = model_param_dict["drug_smiles"]
    target_seq = model_param_dict["target_seq"]

    model_name = param_dict["model_name"]
    task_type = task_param_dict["task_type"]
    drug_target_classification_prediction = MODEL_DICT[task_type][model_name]
    affinity_classification = drug_target_classification_prediction(**model_param_dict)

    result_data = [
        ['drug_smiles', 'target_seq', 'affinity_classification']
    ]
    result_data += [[drug_smiles[i], target_seq, affinity_classification[i]] for i in range(len(drug_smiles))]
    save_task_result(user_email=user_email, task_id=task_id, result_data=result_data)

    return


@app.task(base=CustomTask)
def async_general_algorithm_task(param_dict):
    """
    异步通用算法任务
    目前适配的任务有:
        - 药物属性预测异步任务
        - 药物-药物反应预测异步任务
        - 药物生成异步任务
        - 药物逆合成反应预测异步任务
        - 基于药物-药物反应的分子优化异步任务
        - 中药药物反应预测
        - 中药属性预测
        - 中药中药反应预测
    其余不适配的任务一般都是需要对算法结果进行进一步的格式处理
    @param param_dict:
    @return:
    """
    task_id = current_task.request.id
    change_task_status(task_id=task_id, status=RUNNING)
    model_param_dict = param_dict["model_param_dict"]
    task_param_dict = param_dict["task_param_dict"]
    user_email = task_param_dict["user_email_id"]

    model_name = param_dict["model_name"]
    task_type = task_param_dict["task_type"]
    algorithm_function = MODEL_DICT[task_type][model_name]
    result_data = algorithm_function(**model_param_dict)
    save_task_result(user_email=user_email,
                     task_id=task_id,
                     result_data=result_data,
                     model_param_dict=model_param_dict,
                     task_type=task_type)
    return


def save_task_result(user_email, task_id, result_data, model_param_dict=None, task_type=None):
    """
    算法任务结果保存函数
    该任务结果保存函数不适用于药物分子生成功能
    @param user_email:用户邮箱/用户名，用来创建一个以用户名命名的目录保存用户任务结果
    @param task_id:任务号，将任务结果保存到以任务号命名的csv文件中
    @param result_data:
        大部分任务以[[表头1，表头2，表头3], [结果1，结果2，结果3]，[结果1，结果2，结果3]]形式保存的任务结果数据
        特殊：
        -   分子生成、hdi、hhi：用json保存结果
    @param model_param_dict:算法模型的任务参数，保存成同"任务id_config"的json文件，方便用户查看结果对应的任务参数
    @param task_type:任务类型，根据类型不同进行不同的结果处理
    @return:
    """
    result_dir = f'./result/{user_email}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if model_param_dict is not None:
        model_param_path = f'{result_dir}/{task_id}_config.json'
        with open(model_param_path, 'w') as f:
            json.dump(model_param_dict, f, indent=4)

    if task_type in JSON_RESULT_TASKS:
        result_path = f'{result_dir}/{task_id}.json'
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=4)
    else:
        result_path = f'{result_dir}/{task_id}.csv'
        with open(result_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(result_data)
    return


@app.task
def scheduled_records_save():
    """
    定时将新增的访问数据持久化到磁盘上
    同时定期与mysql进行简单交互，方式mysql超时断连
    """
    access_times = cache.get(ACCESS_TIMES)
    task_times = cache.get(TASK_TIMES)

    try:
        num = UserModel.objects.count()
        print(f"heartbeat ping success: user number {num}")
    except Exception as e:
        print(f"heartbeat ping fails: {e}")

    if access_times is not None and task_times is not None:
        with open(ACCESS_RECORDS_PATH, 'w') as f:
            json_content = {
                ACCESS_TIMES: access_times,
                TASK_TIMES: task_times,
                UPDATE_TIME: str(now())
            }
            json.dump(json_content, f, indent=4)

# if __name__ == '__main__':
#     print(sys.path)
