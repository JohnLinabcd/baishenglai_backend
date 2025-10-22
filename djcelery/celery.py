import os
from celery import Celery
from config import *

app = Celery("djcelery")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "baishenglai_backend.settings")
import django

django.setup()
app.config_from_object("djcelery.config")
app.autodiscover_tasks(["djcelery.sms", ])

app.conf.beat_schedule = {
    'save_access_records': {
        'task': 'djcelery.sms.tasks.scheduled_records_save',
        'schedule': SAVE_RECORDS_INTERVAL,  # 定时任务的时间间隔，单位为秒
    },
}

import logging
from logging.handlers import RotatingFileHandler

# 配置 Celery 日志
app.conf.update(
    CELERYD_HIJACK_ROOT_LOGGER=False,
    CELERYD_LOG_COLOR=False,
    CELERYD_LOG_LEVEL='ERROR',  # 设置 Celery 日志级别
)

# 配置 Python 日志模块
log_file = './log/celery.log'
# 检查文件所在目录是否存在，如果不存在则创建
if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))

# 创建空的日志文件
if not os.path.exists(log_file):
    with open(log_file, 'a'):
        os.utime(log_file, None)

logger = logging.getLogger('celery')
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
