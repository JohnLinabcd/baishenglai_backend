"""baishenglai_backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from drug import views as drug_views
from user import views as user_views
from task import views as task_views

trans_base = 'api/'
urlpatterns = [
    # path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    # path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    path(f'{trans_base}user/send-register-email/', user_views.send_register_email),
    path(f'{trans_base}user/register/', user_views.user_register),
    path(f'{trans_base}user/login/', user_views.user_login),
    path(f'{trans_base}user/send_reset_password_email/', user_views.send_reset_password_email),
    path(f'{trans_base}user/user_reset_password/', user_views.user_reset_password),

    # path('email/test/', user_views.send_email_test),

    # path('test/', drug_views.test),
    path(f'{trans_base}function/get-target-list/', drug_views.get_target_list),
    path(f'{trans_base}function/get-cell-list/', drug_views.get_cell_list),
    path(f'{trans_base}function/get-herb-name-list/', drug_views.get_herb_name_list),


    path(f'{trans_base}function/drug-target-affinity-regression/', drug_views.drug_target_affinity_regression),
    path(f'{trans_base}function/drug-target-affinity-classification/', drug_views.drug_target_affinity_classification),
    path(f'{trans_base}function/drug-cell-response-regression/', drug_views.drug_cell_response_regression),
    path(f'{trans_base}function/drug-property/', drug_views.drug_property),
    path(f'{trans_base}function/drug-drug-response/', drug_views.drug_drug_response),
    path(f'{trans_base}function/single-drug-property/', drug_views.single_drug_property),
    path(f'{trans_base}function/drug-generation/', drug_views.drug_cell_response_regression_generation),
    path(f'{trans_base}function/drug-synthesis-design/', drug_views.drug_synthesis_design),
    path(f'{trans_base}function/drug-cell-response-regression-optimization/', drug_views.drug_cell_response_regression_optimization),
    path(f'{trans_base}function/herb-drug-interaction-prediction/', drug_views.herb_drug_interaction_prediction),
    path(f'{trans_base}function/herb-property-prediction/', drug_views.herb_property_prediction),
    path(f'{trans_base}function/herb-herb-interaction-prediction/', drug_views.herb_herb_interaction_prediction),
    # path(f'{trans_base}time-consuming-test/', drug_views.time_consuming_task),

    path(f'{trans_base}task/list/', task_views.task_list),
    path(f'{trans_base}task/detail/', task_views.task_detail),
    path(f'{trans_base}task/revoke/', task_views.task_revoke),
    path(f'{trans_base}task/delete/', task_views.task_delete),
    path(f'{trans_base}task/download/', task_views.task_download),

    # 一些网站需要的统计量，比如任务数量，用户数量等
    path(f'{trans_base}task/get-access-records/', task_views.get_access_records),
    path(f'{trans_base}task/add-access-records/', task_views.add_access_records),
    path(f'{trans_base}task/get-online-users-num/', task_views.get_online_users_num),
    path(f'{trans_base}task/add-online-users-num/', task_views.add_online_users_num),
    path(f'{trans_base}task/decline-online-users-num/', task_views.decline_online_users_num),
    path(f'{trans_base}user/get-user-num/', user_views.get_user_num),

]

