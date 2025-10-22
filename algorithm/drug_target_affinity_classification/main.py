import os

from algorithm.drug_target_affinity_classification.models import DrugBAN
from algorithm.drug_target_affinity_classification.utils import graph_collate_func, drug_preprocess
from algorithm.drug_target_affinity_classification.configs import get_cfg_defaults
from algorithm.drug_target_affinity_classification.dataloader import DTIDataset
from torch.utils.data import DataLoader
from algorithm.drug_target_affinity_classification.predict import test
import torch
import numpy as np
import pdb


def drug_target_classification_prediction(model_type, drug_smiles, target_seq, batch_size, device):
    cur_path = os.path.dirname(__file__)

    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(f'{cur_path}/configs/DrugBAN_LL4.yaml')
    # print(f"Running on: {device}", end="\n")

    valid_drug_list, false_flag = drug_preprocess(drug_smiles)

    test_dataset = DTIDataset(valid_drug_list, target_seq)

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}
    params['shuffle'] = False
    params['drop_last'] = False

    test_generator = DataLoader(test_dataset, **params)

    model = DrugBAN(**cfg).to(device)

    torch.backends.cudnn.benchmark = True
    if model_type == 'model_DTI':
        PATH = f'{cur_path}/pretrained_models/model_DTI.model'
    model.load_state_dict(torch.load(PATH, map_location=device))
    result = test(model, device, test_generator)
    for i, flag in enumerate(false_flag):
        if not flag:
            result = np.insert(result, i, None)
    return result


if __name__ == '__main__':
    model_type = 'model_DTI'
    drug_smiles = [
        "CC(=O)Nc1cc(N)c(C#N)c(-c2ccccc2)n1",
        "COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1",
        "N#CCC(C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1.O=P(O)(O)O",
        " ",  # 错误示例测试
        "COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1",
        "CC(C)(C)c1cc(NC(=O)Nc2ccc(-c3cn4c(n3)sc3cc(OCCN5CCOCC5)ccc34)cc2)no1"
    ]
    target_seq = ('MERKVLALQARKKRTKAKKDKAQRKSETQHRGSAPHSESDLPEQEEEILGSDDDEQEDPNDYCKGGYHLVKIGDLFNGRYHVIRKLGWGHFSTVWLSWDI'
                  'QGKKFVAMKVVKSAEHYTETALDEIRLLKSVRNSDPNDPNREMVVQLLDDFKISGVNGTHICMVFEVLGHHLLKWIIKSNYQGLPLPCVKKIIQQVLQGL'
                  'DYLHTKCRIIHTDIKPENILLSVNEQYIRRLAAEATEWQRSGAPPPSGSAVSTAPQPKPADKMSKNKKKKLKKKQKRQAELLEKRMQEIEEMEKESGPGQ'
                  'KRPNKQEESESPVERPLKENPPNKMTQEKLEESSTIGQDQTLMERDTEGGAAEINCNGVIEVINYTQNSNNETLRHKEDLHNANDCDVQNLNQESSFLSS'
                  'QNGDSSTSQETDSCTPITSEVSDTMVCQSSSTVGQSFSEQHISQLQESIRAEIPCEDEQEQEHNGPLDNKGKSTAGNFLVNPLEPKNAEKLKVKIADLGN'
                  'ACWVHKHFTEDIQTRQYRSLEVLIGSGYNTPADIWSTACMAFELATGDYLFEPHSGEEYTRDEDHIALIIELLGKVPRKLIVAGKYSKEFFTKKGDLKHI'
                  'TKLKPWGLFEVLVEKYEWSQEEAAGFTDFLLPMLELIPEKRATAAECLRHPWLNS')
    batch_size = 64
    device = 'cuda:0'

    result = drug_target_classification_prediction(model_type=model_type, drug_smiles=drug_smiles,
                                                   target_seq=target_seq, batch_size=batch_size, device=device)

    print(result)
