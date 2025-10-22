import json
import os.path

from django.apps import AppConfig
from django.core.cache import cache
from config import *
from django.utils.timezone import now


class TaskConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'task'

    def ready(self):
        """
        加载历史访问量到cache
        如果访问量记录文件不存在，则创建；若存在，则加载
        """
        if not os.path.exists(ACCESS_RECORDS_PATH):
            with open(ACCESS_RECORDS_PATH, 'w') as f:
                json_content = {
                    ACCESS_TIMES: 0,
                    TASK_TIMES: 0,
                    UPDATE_TIME: str(now())
                }
                json.dump(json_content, f, indent=4)

        with open(ACCESS_RECORDS_PATH, 'r') as f:
            json_content = json.load(f)
            cache.set(ACCESS_TIMES, json_content[ACCESS_TIMES], timeout=None)
            cache.set(TASK_TIMES, json_content[TASK_TIMES], timeout=None)

        cache.set(ONLINE_USERS_NUM, 0, timeout=None)
