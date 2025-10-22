from utils.utils import *

# 主域名常量
DOMAIN_HTTP = "https://www.baishenglai.net"

# 任务类型号设计
DRUG_GENERATION = 1                                         # 药物生成
DRUG_TARGET_AFFINITY_REGRESSION_PREDICTION = 2              # 药物靶体亲和度回归预测
DRUG_TARGET_AFFINITY_CLASSIFICATION_PREDICTION = 3          # 药物靶体亲和度分类预测
DRUG_CELL_RESPONSE_PREDICTION = 4                           # 药物细胞器反应预测
DRUG_PROPERTY_PREDICTION = 5                                # 药物属性预测
DRUG_SYNTHESIS_DESIGN = 6                                   # 药物合成路径设计
DRUG_DRUG_RESPONSE_PREDICTION = 7                           # 药物药物反应预测
DRUG_CELL_RESPONSE_OPTIMIZATION = 8                         # 基于药物细胞系反应的分子优化
HERB_DRUG_INTERACTION = 9                                   # 中药药物反应预测
HERB_PROPERTY_PREDICTION = 10                               # 中药属性预测
HERB_HERB_INTERACTION = 11                                  # 中药中药反应预测

TASK_TYPE_CHOICES = {
    DRUG_GENERATION: "药物生成",
    DRUG_TARGET_AFFINITY_REGRESSION_PREDICTION: "药物靶点亲和度回归预测",
    DRUG_TARGET_AFFINITY_CLASSIFICATION_PREDICTION: "药物靶点亲和度分类预测",
    DRUG_CELL_RESPONSE_PREDICTION: "药物细胞系反应预测",
    DRUG_PROPERTY_PREDICTION: "药物属性预测",
    DRUG_SYNTHESIS_DESIGN: "药物合成路径规划",
    DRUG_DRUG_RESPONSE_PREDICTION: "药物药物反应预测",
    DRUG_CELL_RESPONSE_OPTIMIZATION: "基于药物细胞系反应的分子优化",
    HERB_DRUG_INTERACTION: "中药药物反应预测",
    HERB_PROPERTY_PREDICTION: "中药属性预测",
    HERB_HERB_INTERACTION: "中药中药反应预测",
}

# 任务类型到对应前端url（在任务结束邮件中使用）
TASK_TYPE_URL = {
    DRUG_GENERATION: "drugGeneration_reactiveCondition",
    DRUG_TARGET_AFFINITY_REGRESSION_PREDICTION: "drugTargetAffinity",
    DRUG_TARGET_AFFINITY_CLASSIFICATION_PREDICTION: "drugTargetAffinity",
    DRUG_CELL_RESPONSE_PREDICTION: "drugCellInteraction",
    DRUG_PROPERTY_PREDICTION: "drugProperties",
    DRUG_SYNTHESIS_DESIGN: "drugPath",
    DRUG_DRUG_RESPONSE_PREDICTION: "drugDrugInteraction",
    DRUG_CELL_RESPONSE_OPTIMIZATION: "drugGeneration_fragmentBasedMolecularOptimization",
    HERB_DRUG_INTERACTION: "herbDrugInteraction",
    HERB_PROPERTY_PREDICTION: "herbDrugProperties",
    HERB_HERB_INTERACTION: "herbHerbInteraction",
}

# 任务状态码
TASK_STATUS_CHOICES = {
    1: "等待中",
    2: "已开始",
    3: "已完成",
    4: "失败",
    5: "已取消",
}

PENDING = 1
RUNNING = 2
COMPLETED = 3
FAILED = 4
CANCELLED = 5

# 随机种子
SEED = 42

# 无需鉴权的url列表
AUTH_WITHOUT_URL_LIST = {
    '/api/user/login/',
    '/api/user/register/',
    '/api//email/test/',
    '/api/user/send-register-email/',
    '/api/user/send_reset_password_email/',
    '/api/user/user_reset_password/',
    '/api/function/get-target-list/',
    '/api/function/get-cell-list/',
    '/api/function/get-herb-name-list/',

    '/api/task/get-access-records/',
    '/api/task/add-access-records/',
    '/api/task/get-online-users-num/',
    '/api/task/add-online-users-num/',
    '/api/task/decline-online-users-num/',
    '/api/user/get-user-num/',

}

# 时间间隔管理
CACHE_TIME = 5  # cache有效时间，单位为s，用来过滤短时间内的重复请求
SAVE_RECORDS_INTERVAL = 1 * 60 * 60  # 单位为s，用来定期持久化网站的访问数据

# 访问统计数据本地文件路径
ACCESS_RECORDS_PATH = './access_records.json'

# 共用cache token
ACCESS_TIMES = "access_times"
TASK_TIMES = "task_times"
UPDATE_TIME = "update_time"
ONLINE_USERS_NUM = "online_users_num"

# 多模型算法的模型选择
from algorithm.drug_generation.main import drug_cell_response_regression_generation as drug_generation_rmcd
from algorithm.drug_target_affinity_regression.main import drug_target_affinity_regression_predict as cl_dta
from algorithm.drug_target_affinity_classification.main import drug_target_classification_prediction as siam_dti
from algorithm.drug_cell_response_regression.main import drug_cell_response_regression_predict as transe_drp
from algorithm.drug_synthesis_design.scripts.main import Retrosynthetic_reaction_pathway_prediction as local_retro
from algorithm.drug_drug_response.main import drug_drug_response_predict as gmpnn_cs
from algorithm.drug_cell_response_regression_optimization.main import drug_cell_response_regression_optimization as fmop
from algorithm.drug_property.main import drug_property_prediction as drug_property_prediction_graph
from algorithm.drug_property_kg.main import drug_property_prediction as drug_property_prediction_kg
from algorithm.herb_drug_interaction.main_hdi import predict_hdi as hdi_predict_kg
from algorithm.drug_property_kg.main_kgpp import herb_property_prediction as herb_property_prediction_kg
from algorithm.herb_drug_interaction.main_hhi import predict_hhi_only as hhi_predict_kg

MODEL_DICT = {
    DRUG_GENERATION: {
        "RMCD": drug_generation_rmcd
    },
    DRUG_TARGET_AFFINITY_REGRESSION_PREDICTION: {
        "CL-DTA": cl_dta
    },
    DRUG_TARGET_AFFINITY_CLASSIFICATION_PREDICTION: {
        "SiamDTI": siam_dti
    },
    DRUG_CELL_RESPONSE_PREDICTION: {
        "TransEDRP": transe_drp
    },
    DRUG_SYNTHESIS_DESIGN: {
        "LocalRetro": local_retro
    },
    DRUG_DRUG_RESPONSE_PREDICTION: {
        "GMPNN-CS": gmpnn_cs
    },
    DRUG_CELL_RESPONSE_OPTIMIZATION: {
        "FMOP": fmop
    },
    DRUG_PROPERTY_PREDICTION: {
        "GraphPP": drug_property_prediction_graph,
        "KGPP": drug_property_prediction_kg
    },
    HERB_DRUG_INTERACTION: {
        "KGHDI": hdi_predict_kg
    },
    HERB_PROPERTY_PREDICTION: {
        "KGPP": herb_property_prediction_kg
    },
    HERB_HERB_INTERACTION: {
        "KGHHI": hhi_predict_kg
    },
}


# 页面初始化数据
TARGET_LIST = read_target_list()
CELL_LIST = read_cell_list()
HERB_NAME_LIST = read_herb_name_list()


# 其他
NUM_ITEMS_TO_READ = 20           # 任务结果详情展示条目显示

# 使用JSON保存结果的任务
JSON_RESULT_TASKS = {
    DRUG_GENERATION,
    HERB_DRUG_INTERACTION,
    HERB_HERB_INTERACTION,
}
