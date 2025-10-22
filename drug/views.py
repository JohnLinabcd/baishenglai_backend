from config import *
from drug.utils import *


@api_view(['POST'])
def drug_cell_response_regression(request):
    """
    药物细胞器反应回归预测的视图函数
    比较重要的任务参数：
    @drug_smiles：以列表形式保存的药物字符串['药物1','药物2','药物3']，中间可能存在无效药物字符串，在算法层面进行检查
    @target_seq：唯一的细胞器名称序列
    @param request:
    @return:
    """
    task_type = DRUG_CELL_RESPONSE_PREDICTION
    return general_task(request, task_type)


@api_view(['POST'])
def drug_target_affinity_regression(request):
    """
    药物靶体亲和度回归预测的视图函数
    比较重要的任务参数：
    @drug_smiles：以列表形式保存的药物字符串['药物1','药物2','药物3']，中间可能存在无效药物字符串，在算法层面进行检查
    @target_seq：唯一的蛋白序列字符串
    @param request:
    @return:
    """
    task_type = DRUG_TARGET_AFFINITY_REGRESSION_PREDICTION
    return general_task(request, task_type)


@api_view(['POST'])
def drug_target_affinity_classification(request):
    """
    药物靶体亲和度分类预测的视图函数
    比较重要的任务参数：
    @drug_smiles：以列表形式保存的药物字符串['药物1','药物2','药物3']，中间可能存在无效药物字符串，在算法层面进行检查
    @target_seq：唯一的蛋白序列字符串
    @param request:
    @return:
    """
    task_type = DRUG_TARGET_AFFINITY_CLASSIFICATION_PREDICTION
    return general_task(request, task_type)


@api_view(['POST'])
def drug_property(request):
    """
    药物属性预测的视图函数
    比较重要的任务参数：
    @drug_smiles：以列表形式保存的药物字符串['药物1','药物2','药物3']，中间可能存在无效药物字符串，在算法层面进行检查
    @task_list：以列表形式保存的任务名，里面包含了用户指定要进行的任务名，实际以数据集名字指代任务名，[BACE, BBBP, ClinTox, SIDER, LIPO]
    @param request:
    @return:
    """
    task_type = DRUG_PROPERTY_PREDICTION
    return general_task(request, task_type)


@api_view(['POST'])
def single_drug_property(request):
    """
    单个药物属性预测的视图函数
    该任务函数是为了满足用户点击某个药物序列后，在药物序列页面中自动计算药物的各项属性
    @param request:
    @return:
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        post_data = json.loads(request.body.decode('utf-8'))
        drug_smiles = post_data['drugSmilesInput']
        if type(drug_smiles) is str:
            drug_smiles = [drug_smiles]
        task_list = ['BACE', 'BBBP', 'ClinTox', 'Tox21', 'ESOL', 'FreeSolv', 'LIPO']

        device = 'cpu'
        batch_size = 128

        model_param_dict = {
            "drug_smiles": drug_smiles,
            "task_list": task_list,
            "batch_size": batch_size,
            "device": device
        }

        task_result = drug_property_prediction(**model_param_dict)
        result_header = task_result[0][1:]
        result_value = [float(x) for x in task_result[1][1:]]
        # print(result_value)
        return JsonResponse({
            'code': 200,
            'msg': '任务提交成功',
            'data': {
                'drug_smiles': task_result[1][0],
                'result_header': result_header,
                'result_value': result_value
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'code': 500,
            'msg': '任务请求失败',
        })


@api_view(['POST'])
def drug_drug_response(request):
    """
    药物-药物反应预测的视图函数
    @param request:
    @return:
    """
    task_type = DRUG_DRUG_RESPONSE_PREDICTION
    return general_task(request, task_type)


@api_view(['POST'])
def drug_cell_response_regression_generation(request):
    """
    药物生成的视图函数
    @param request:
    @return:
    """
    task_type = DRUG_GENERATION
    return general_task(request, task_type)


@api_view(['POST'])
def drug_synthesis_design(request):
    """
    药物合成路径规划的视图函数
    @param request:
    @return:
    """
    task_type = DRUG_SYNTHESIS_DESIGN
    return general_task(request, task_type)


@api_view(['POST'])
def drug_cell_response_regression_optimization(request):
    """
    基于药物细胞系反应的分子片段优化
    @param request:
    @return:
    """
    task_type = DRUG_CELL_RESPONSE_OPTIMIZATION
    return general_task(request, task_type)


@api_view(['POST'])
def herb_drug_interaction_prediction(request):
    """
    中药药物反应预测
    @param request:
    @return:
    """
    task_type = HERB_DRUG_INTERACTION
    return general_task(request, task_type)


@api_view(['POST'])
def herb_property_prediction(request):
    """
    中药属性预测
    @param request:
    @return:
    """
    task_type = HERB_PROPERTY_PREDICTION
    return general_task(request, task_type)


@api_view(['POST'])
def herb_herb_interaction_prediction(request):
    """
    中药中药反应预测
    @param request:
    @return:
    """
    task_type = HERB_HERB_INTERACTION
    return general_task(request, task_type)


def general_task(request, task_type):
    """
    通用任务生成函数
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        post_data = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({
            'code': 500,
            'msg': '请求体不是有效的 JSON 格式',
        })

    token = post_data['token']
    user_email = AccessToken(token).payload.get('user_email')
    model_name = post_data['model_name']
    if task_type not in TASK_TYPE_CHOICES.keys():
        return JsonResponse({
            'code': 500,
            'msg': '任务请求失败，任务类型错误',
        })
    if model_name not in MODEL_DICT[task_type].keys():
        return JsonResponse({
            'code': 500,
            'msg': '任务请求失败，请求模型错误',
        })
    try:
        model_param_dict = get_model_param_dict(post_data, task_type)
    except KeyError as e:
        return JsonResponse({
            'code': 500,
            'msg': f'任务请求失败，算法参数错误: {str(e)}',
        })
    if model_param_dict is None:
        return JsonResponse({
            'code': 500,
            'msg': '任务请求失败，算法参数匹配未实现',
        })
    param_dict = {
        "task_param_dict": {
            "user_email_id": user_email,
            "task_type": task_type,
        },
        "model_param_dict": model_param_dict,
        "model_name": model_name
    }
    if task_type == DRUG_CELL_RESPONSE_PREDICTION:
        task_result = async_drug_cell_response_regression_predict.delay(param_dict)
    elif task_type == DRUG_TARGET_AFFINITY_REGRESSION_PREDICTION:
        task_result = async_drug_target_affinity_regression_predict.delay(param_dict)
    elif task_type == DRUG_TARGET_AFFINITY_CLASSIFICATION_PREDICTION:
        task_result = async_drug_target_classification_regression_predict.delay(param_dict)
    else:
        task_result = async_general_algorithm_task.delay(param_dict)
    return JsonResponse({
        'code': 200,
        'msg': '任务提交成功',
        'data': {
            'task_id': task_result.id
        }
    })


def get_model_param_dict(post_data, task_type):
    """
    根据反应类型从请求中解析算法任务要执行的参数
    """
    device = allocate_cuda(task_type)
    model_param_dict = None
    try:
        if task_type == DRUG_CELL_RESPONSE_PREDICTION:
            drug_smiles = post_data['drugSmilesInput']
            cell_name = post_data['cellNameInput']
            model_type = 'TransEDRP'
            batch_size = 128
            model_param_dict = {
                "model_type": model_type,
                "drug_smiles": drug_smiles,
                "cell_name": cell_name,
                "batch_size": batch_size,
                "device": device
            }
        elif task_type == DRUG_TARGET_AFFINITY_REGRESSION_PREDICTION:
            drug_smiles = post_data['drugSmilesInput']
            target_seq = post_data['targetSeqInput']
            model_type = 'ColdDTA'
            batch_size = 128
            model_param_dict = {
                "model_type": model_type,
                "drug_smiles": drug_smiles,
                "target_seq": target_seq,
                "batch_size": batch_size,
                "device": device
            }
        elif task_type == DRUG_TARGET_AFFINITY_CLASSIFICATION_PREDICTION:
            drug_smiles = post_data['drugSmilesInput']
            target_seq = post_data['targetSeqInput']
            model_type = 'model_DTI'
            batch_size = 128
            model_param_dict = {
                "model_type": model_type,
                "drug_smiles": drug_smiles,
                "target_seq": target_seq,
                "batch_size": batch_size,
                "device": device
            }
        elif task_type == DRUG_PROPERTY_PREDICTION:
            drug_smiles = post_data['drugSmilesInput']
            task_list = post_data['task_list']
            batch_size = 128
            model_param_dict = {
                "drug_smiles": drug_smiles,
                "task_list": task_list,
                "batch_size": batch_size,
                "device": device
            }
        elif task_type == DRUG_DRUG_RESPONSE_PREDICTION:
            smiles_pairs = post_data['smiles_pairs']
            task_list = post_data['task_list']
            batch_size = 128
            model_param_dict = {
                "smiles_pairs": smiles_pairs,
                "task_list": task_list,
                "batch_size": batch_size,
                "device": device
            }
        elif task_type == DRUG_GENERATION:
            model_param_dict = {
                "model_type": 'molgen',
                "condition_strength": float(post_data['condition_strength']),
                "seed": 42,
                "gen_number": int(post_data['gen_number']),
                "timestep": int(post_data['timestep']),
                "ic50": float(post_data['ic50']),
                "cell_line": int(post_data['cell_line']),
                "device": device
            }
        elif task_type == DRUG_SYNTHESIS_DESIGN:
            smiles_list = post_data['smiles_list']
            dataset = post_data['dataset']
            top_k_result = post_data['top_k_result']
            batch_size = 128
            model_param_dict = {
                "smiles_list": smiles_list,
                "dataset": dataset,
                "top_k_result": top_k_result,
                "device": device,
                "batch_size": batch_size
            }
        elif task_type == DRUG_CELL_RESPONSE_OPTIMIZATION:
            model_param_dict = {
                "model_type": 'molgen',
                "condition_strength": float(post_data['condition_strength']),
                "seed": SEED,
                "gen_number": int(post_data['gen_number']),
                "timestep": int(post_data['timestep']),
                "cell_line": str(post_data['cell_line']),
                "mask": post_data['mask'],
                "gt": str(post_data['gt']),
                "ic50": float(post_data['ic50']),
                "device": device
            }
        elif task_type == HERB_DRUG_INTERACTION:
            model_param_dict = {
                "drug_pairs": post_data['drug_pairs'],
                "device": device
            }
        elif task_type == HERB_PROPERTY_PREDICTION:
            batch_size = 128
            model_param_dict = {
                "drugs": post_data['drugs'],
                "task_list": post_data['task_list'],
                "batch_size": batch_size,
                "device": device
            }
        elif task_type == HERB_HERB_INTERACTION:
            model_param_dict = {
                "input_herbs": post_data['input_herbs'],
                "device": device
            }

    except KeyError:
        raise

    return model_param_dict
