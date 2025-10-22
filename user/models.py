from django.db import models


# Create your models here.
class UserModel(models.Model):
    user_email = models.EmailField(primary_key=True)
    password = models.CharField(max_length=255)


class UserActivationModel(models.Model):
    """
    用于保存用户注册激活码的数据表
    """
    user_email = models.EmailField()
    activation_code = models.CharField(max_length=255)
    send_time = models.DateTimeField(auto_now_add=True)
