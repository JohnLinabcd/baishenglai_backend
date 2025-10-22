from django.db import models
from user.models import UserModel
from config import TASK_TYPE_CHOICES, TASK_STATUS_CHOICES


# Create your models here.
class TaskModel(models.Model):
    task_id = models.CharField(max_length=255, primary_key=True)
    user_email = models.ForeignKey(to=UserModel, to_field="user_email", on_delete=models.CASCADE)
    task_type = models.SmallIntegerField(choices=TASK_TYPE_CHOICES.items())
    submit_time = models.DateTimeField(auto_now_add=True)
    finish_time = models.DateTimeField(null=True, blank=True)
    predicted_finish_time = models.DateTimeField(null=True, blank=True)
    task_status = models.SmallIntegerField(choices=TASK_STATUS_CHOICES.items(), default=1)

