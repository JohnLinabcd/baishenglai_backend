import json

import numpy as np
from rest_framework.exceptions import NotFound
from config import *


def process_drp_results(ic_50_val):
    ic_50_val = np.float64(ic_50_val)
    ic_50_val = -10 * np.log(1 / ic_50_val - 1)
    return round(ic_50_val, 3)


def generate_result_path(user_email, task_type, task_id):
    extension = 'json' if task_type in JSON_RESULT_TASKS else 'csv'
    result_path = f"./result/{user_email}/{task_id}.{extension}"
    if os.path.exists(result_path):
        return result_path
    else:
        return None


def read_json_file(file_path, key=None, limit=NUM_ITEMS_TO_READ):
    with open(file_path, "r") as f:
        data = json.load(f)
        if key:
            return data[key][:limit]
        return data[:limit]


def get_hhi_details(result_path):
    """
    hhi预览结果处理函数
    限制为前5个中药对的前10个结果，最多50个条目
    """
    result = []
    with open(result_path, "r") as f:
        data = json.load(f)
    data = data['details']
    pair_num = min(len(data), 5)
    for i in range(pair_num):
        cur_pair = data[i]
        cur_pair_details = cur_pair['details']
        cur_pair['details'] = cur_pair_details[:min(10, len(cur_pair_details))]
        result.append(cur_pair)
    return result

