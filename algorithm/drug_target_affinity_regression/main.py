import os.path
import pdb
import torch
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from algorithm.drug_target_affinity_regression.models.ColdDTA_model_clip import ColdDTA
# from .models.GraphDTA_model import GraphDTA
# from .models.MSF_DTA_model import MSF_DTA
# from .models.MSGNN_DTA_model import MSGNN_DTA
from algorithm.drug_target_affinity_regression.utils import *


def drug_target_affinity_regression_predict(model_type, drug_smiles, batch_size, target_seq, device='cpu'):
    pred = -1

    total_preds = None
    false_flag = None
    if model_type == 'ColdDTA':
        model = ColdDTA(device=device).to(device)

        model_path = f"{os.path.dirname(__file__)}/pretrained_models/ColdDTA.model"
        # model_path = "pretrained_models/ColdDTA.model"
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        data_list, false_flag = preprocess_ColdDTA(drug_smiles, target_seq)
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, drop_last=False)

        model.eval()
        total_preds = torch.Tensor()
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            pred = model(data)
            pred = pred.cpu().view(-1, 1)
            total_preds = torch.cat((total_preds, pred), 0)
        total_preds = total_preds.detach().numpy().flatten()

    affinity_value = total_preds
    for idx, flag in enumerate(false_flag):
        if not flag:
            affinity_value = np.insert(affinity_value, idx, None)
    return affinity_value


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    import torch

    # 检查系统中可用的CUDA设备数量
    num_devices = torch.cuda.device_count()
    print("可用的CUDA设备数量：", num_devices)

    # 打印每个设备的信息
    for i in range(num_devices):
        print("CUDA 设备 {}: {}".format(i, torch.cuda.get_device_name(i)))

    model_type = 'ColdDTA'
    drug_smiles = [
        'CC(=O)Nc1cc(N)c(C#N)c(-c2ccccc2)n1',
        'COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1',
        'N#CCC(C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1.O=P(O)(O)O',
        '00000',    # 错误示例测试
        'COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1',
        'CC(C)(C)c1cc(NC(=O)Nc2ccc(-c3cn4c(n3)sc3cc(OCCN5CCOCC5)ccc34)cc2)no1'
        ]
    target_seq = ('MERKVLALQARKKRTKAKKDKAQRKSETQHRGSAPHSESDLPEQEEEILGSDDDEQEDPNDYCKGGYHLVKIGDLFNGRYHVIRKLGWGHFSTVWLSWDI'
                  'QGKKFVAMKVVKSAEHYTETALDEIRLLKSVRNSDPNDPNREMVVQLLDDFKISGVNGTHICMVFEVLGHHLLKWIIKSNYQGLPLPCVKKIIQQVLQGL'
                  'DYLHTKCRIIHTDIKPENILLSVNEQYIRRLAAEATEWQRSGAPPPSGSAVSTAPQPKPADKMSKNKKKKLKKKQKRQAELLEKRMQEIEEMEKESGPGQ'
                  'KRPNKQEESESPVERPLKENPPNKMTQEKLEESSTIGQDQTLMERDTEGGAAEINCNGVIEVINYTQNSNNETLRHKEDLHNANDCDVQNLNQESSFLSS'
                  'QNGDSSTSQETDSCTPITSEVSDTMVCQSSSTVGQSFSEQHISQLQESIRAEIPCEDEQEQEHNGPLDNKGKSTAGNFLVNPLEPKNAEKLKVKIADLGN'
                  'ACWVHKHFTEDIQTRQYRSLEVLIGSGYNTPADIWSTACMAFELATGDYLFEPHSGEEYTRDEDHIALIIELLGKVPRKLIVAGKYSKEFFTKKGDLKHI'
                  'TKLKPWGLFEVLVEKYEWSQEEAAGFTDFLLPMLELIPEKRATAAECLRHPWLNS')
    device = 'cuda:2'
    affinity_value, false_flag = drug_target_affinity_regression_predict(model_type=model_type, drug_smiles=drug_smiles,
                                                             target_seq=target_seq, device=device)
    # pdb.set_trace()
    for idx, flag in enumerate(false_flag):
        if not flag:
            affinity_value = np.insert(affinity_value, idx, None)
    print(affinity_value)
