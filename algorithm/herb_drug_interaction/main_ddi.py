# -*- coding: utf-8 -*-
# @Time    : 2024/9/3 上午12:30
# @Author  : Chen Mukun & Wu Wenjie
# @File    : main.py
# @Software: PyCharm
# @desc    : 子算法，预测两个 smile 的相互反应
import json
import pickle
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from random import random

import tqdm
from sklearn import metrics
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import dgl
import pandas as pd

from rdkit import Chem
from torch.utils.data import Dataset
import os

import torch
import torch.distributed as dist

# from utils.logger import initialize_exp, snapshot
# from utils.early_stop import EarlyStopping
from algorithm.herb_drug_interaction.data.mapping import get_mapping
from text2vec import SentenceModel
from dgllife.utils.io import pmap

from algorithm.herb_drug_interaction.model.layer.readout import PairReadout
from algorithm.herb_drug_interaction.model.layer import KGMPNN
import torch.nn.functional as F
import pdb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,4,3,1,0,2"


def do_compute(batch, device, model):
    '''
        *batch: (bg_pos_h, bg_pos_t, bg_pos_cb, pos_rels), (bg_neg_h, bg_neg_t, bg_neg_cb, neg_rels)
    '''
    encoder = model[0]
    readout = model[1]
    probas_pred = []
    pos_tri = batch

    bg_pos_h = encoder(pos_tri[0].to(device))
    bg_pos_t = encoder(pos_tri[1].to(device))
    bg_pos_cb = encoder(pos_tri[2].to(device))

    p_score = readout(pos_tri[0].to(device), bg_pos_h.to(device), pos_tri[1].to(device), bg_pos_t.to(device),
                      pos_tri[2].to(device), bg_pos_cb.to(device), pos_tri[3].to(device))

    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())

    probas_pred = np.concatenate(probas_pred)

    return p_score, probas_pred


def set_seed(seed):
    """
    Freeze every seed for reproducibility.
    torch.cuda.manual_seed_all is useful when using random generation on GPUs.
    e.g. torch.cuda.FloatTensor(100).uniform_()
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Triples:
    def __init__(self, data_dir="triples"):
        self.data = self.load_data(data_dir)
        self.entities, self.entity2id = self.get_entities(self.data)
        self.attributes, self.attribute2id = self.get_attributes(self.data)
        self.relations, self.relation2id = self.get_relations(self.data)
        self.triples = self.read_triple(self.data, self.entity2id, self.relation2id)
        self.h2rt = self.h2rt(self.triples)
        self.t2rh = self.t2rh(self.triples)

    def load_data(self, data_dir):
        with open("%s.txt" % (data_dir), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        relationid = [i for i in range(len(relations))]
        relation2id = dict(zip(relations, relationid))
        return relations, relation2id

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        entityid = [i for i in range(len(entities))]
        entity2id = dict(zip(entities, entityid))
        return entities, entity2id

    def get_attributes(self, data):
        attributes = sorted(list(set([d[0] for d in data])))
        attributeid = [i for i in range(len(attributes))]
        attribute2id = dict(zip(attributes, attributeid))
        return attributes, attribute2id

    def read_triple(self, data, entity2id, relation2id):
        '''
        Read triples and map them into ids.
        '''
        triples = []
        for triple in data:
            h = triple[0]
            r = triple[1]
            t = triple[2]
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples

    def h2rt(self, triples):  # dict: attribute_id  --> list[(rel_id, atom_id)]
        h2rt = defaultdict(list)
        for tri in triples:
            h, r, t = tri
            h2rt[h].append((r, t))
        return h2rt

    def t2rh(self, triples):  # dict: atom_id  --> list[(rel_id, attribute_id)]
        t2rh = defaultdict(list)
        for tri in triples:
            h, r, t = tri
            t2rh[t].append((r, h))
        return t2rh


kg_data = Triples("./algorithm/herb_drug_interaction/resource/triples")
sentence_model = SentenceModel("./algorithm/herb_drug_interaction/text2vec-base-multilingual")
bondtype_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
bond2emb = {}
for idx, bt in enumerate(bondtype_list):
    bond2emb[bt] = idx


def bondtype_features(bond):
    try:
        fbond = bond2emb[str(bond.GetBondType())]
    except:
        fbond = len(bondtype_list)
    return fbond


def combin_smiles_2_kgdgl(mol_a, mol_b):
    cb = Chem.CombineMols(mol_a, mol_b)
    return smiles_2_kgdgl(smiles=cb)[0]


def smiles_2_kgdgl(smiles):
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if mol is None or (type(smiles) == str and smiles == ""):
        return None, None
    connected_atom_list = []
    # for bond in mol.GetBonds():
    #     connected_atom_list.append(bond.GetBeginAtomIdx())
    #     connected_atom_list.append(bond.GetEndAtomIdx())
    for atom in mol.GetAtoms():
        connected_atom_list.append(atom.GetIdx())

    connected_atom_list = sorted(list(set(connected_atom_list)))
    connected_atom_map = {k: v for k, v in zip(connected_atom_list, list(range(len(connected_atom_list))))}
    atoms_feature = [0 for _ in range(len(connected_atom_list))]

    begin_attributes = []
    end_atoms = []
    rel_features = []
    for atom in mol.GetAtoms():
        node_index = atom.GetIdx()
        symbol = atom.GetSymbol()
        atomicnum = atom.GetAtomicNum()
        if node_index not in connected_atom_list:
            continue

        atoms_feature[connected_atom_map[node_index]] = atomicnum

        if symbol in kg_data.entities:
            attribute_id = [h for (r, h) in kg_data.t2rh[kg_data.entity2id[symbol]]]
            rid = [r for (r, h) in kg_data.t2rh[kg_data.entity2id[symbol]]]

            begin_attributes.extend(attribute_id)
            end_atoms.extend([node_index] * len(attribute_id))
            rel_features.extend(i + 4 for i in rid)

    if begin_attributes:
        attribute_id = sorted(list(set(begin_attributes)))
        node_id = [i + len(connected_atom_list) for i in range(len(attribute_id))]
        attrid2nodeid = dict(zip(attribute_id, node_id))
        nodeids = [attrid2nodeid[i] for i in begin_attributes]

        nodes_feature = [i + 118 for i in attribute_id]

    begin_indexes = []
    end_indexes = []
    bonds_feature = []
    edge_type = []

    for bond in mol.GetBonds():
        bond_feature = bondtype_features(bond)
        begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        bonds_feature.append(bond_feature)
        begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        bonds_feature.append(bond_feature)
    edge_type.extend([0] * len(bonds_feature))

    if end_atoms:
        begin_indexes.extend(nodeids)
        end_indexes.extend(end_atoms)
        atoms_feature.extend(nodes_feature)
        bonds_feature.extend(rel_features)
        edge_type.extend([1] * len(rel_features))

    graph = dgl.graph((begin_indexes, end_indexes), idtype=torch.int32)
    graph.edata['e'] = torch.tensor(bonds_feature, dtype=torch.long)
    graph.ndata['h'] = torch.tensor(atoms_feature, dtype=torch.long)
    graph.edata['etype'] = torch.tensor(edge_type, dtype=torch.long)

    return (graph, mol)


def generate_graph(smiles, n_jobs=1):
    if n_jobs > 1:
        graphs = pmap(smiles_2_kgdgl, smiles, n_jobs=n_jobs)
    else:
        graphs = []
        for i, s in enumerate(smiles):
            graphs.append(smiles_2_kgdgl(s))
    return graphs


def sentence_to_emb(id_sentence):
    id, sentence = id_sentence[0], id_sentence[1]
    return (id, sentence_model.encode(sentence))


def generate_sentence_emb(id_sentence, n_jobs=1):
    if n_jobs > 1:
        sentence_emb = pmap(sentence_to_emb, id_sentence, n_jobs=n_jobs)
    else:
        sentence_emb = []
        for i, s in enumerate(id_sentence):
            sentence_emb.append(sentence_to_emb(s))
    return sentence_emb


class PairDataset(Dataset):
    def __init__(self, valid_triple):
        self.valid_triplets = valid_triple

    def __getitem__(self, item):
        return self.valid_triplets[item]

    def __len__(self):
        return len(self.valid_triplets)


def collate_pair(batch, id_sentence_emb):
    pos_rels = []
    pos_h_samples = []
    pos_t_samples = []
    pos_cb_samples = []

    for pos_triplet in batch:
        try:
            dgl1, mol1 = smiles_2_kgdgl(pos_triplet[0])
            dgl2, mol2 = smiles_2_kgdgl(pos_triplet[1])

            pos_h_samples.append(dgl1)
            pos_t_samples.append(dgl2)
            pos_rels.append(id_sentence_emb[int(pos_triplet[2])])
            pos_cb_samples.append(combin_smiles_2_kgdgl(mol1, mol2))
        except:
            print(id_sentence_emb.keys())
            print(pos_triplet)
            raise Exception()

    bg_pos_h = dgl.batch(pos_h_samples)
    bg_pos_h.set_n_initializer(dgl.init.zero_initializer)
    bg_pos_h.set_e_initializer(dgl.init.zero_initializer)

    bg_pos_t = dgl.batch(pos_t_samples)
    bg_pos_t.set_n_initializer(dgl.init.zero_initializer)
    bg_pos_t.set_e_initializer(dgl.init.zero_initializer)

    # for i, sample in enumerate(pos_cb_samples):
    #     if not isinstance(sample, dgl.DGLGraph):
    #         print(f"Unexpected type at index {i}: {type(sample)}")
    
    bg_pos_cb = dgl.batch(pos_cb_samples)
    bg_pos_cb.set_n_initializer(dgl.init.zero_initializer)
    bg_pos_cb.set_e_initializer(dgl.init.zero_initializer)

    pos_rels = torch.FloatTensor(np.array(pos_rels))

    return (bg_pos_h, bg_pos_t, bg_pos_cb, pos_rels)


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--featurizer_type', type=str, default='random')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--node_indim', type=int, default=128)
    parser.add_argument('--edge_indim', type=int, default=64)
    parser.add_argument('--hidden_feats', type=int, default=64)
    parser.add_argument('--node_hidden_feats', type=int, default=64)
    parser.add_argument('--edge_hidden_feats', type=int, default=128)
    parser.add_argument('--num_step_message_passing', type=int, default=6)
    parser.add_argument('--gnn_norm', type=str, default=None)  # None, both, right
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--batchnorm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument('--eval', type=str, default='nonfreeze')

    parser.add_argument('--initial_path', type=str, default='./algorithm/herb_drug_interaction/pretrained/RotatE_128_64_emb.pkl')
    parser.add_argument('--pretarin_path', type=str, default='./algorithm/herb_drug_interaction/pretrained/KGMPNN_pretrain.pkl')

    parser.add_argument('--data_name', type=str, default='hdi')
    parser.add_argument("--dump_path", default="./dump", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_name", default="finetune", type=str,
                        help="Experiment name")
    parser.add_argument("--exp_id", default="data_name", type=str,
                        help="Experiment ID")
    parser.add_argument('--patience', type=int, default=50)

    return parser.parse_known_args()[0].__dict__


def get_batch(smis, batch_size=24, n_job=16, name='drugbank'):
    id_smi, id_desc = get_mapping(name)
    id_sentence_emb = {}
    valid_triplets = []
    invalid = []
    for pair in smis:
        mol1 = Chem.MolFromSmiles(pair[0])
        mol2 = Chem.MolFromSmiles(pair[1])
        if mol1 is None or mol2 is None:
            invalid.append(pair)
            continue
        for rel in id_desc.keys():
            valid_triplets.append([pair[0], pair[1], rel])

    for id_sentence in generate_sentence_emb(list(id_desc.items()), n_job):
        id_sentence_emb[id_sentence[0]] = id_sentence[1]

    print('get batch data')

    dataset = PairDataset(valid_triplets)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        collate_fn=lambda x: collate_pair(x, id_sentence_emb))

    return loader, invalid, valid_triplets, id_desc


all_tasks = ['drugbank', "twosides", "hdi"]


class Inference():
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")
        self.args["device"] = self.device

        self.loaded_dict = pickle.load(open(args['initial_path'], 'rb'))
        self.entity_emb, self.relation_emb = self.loaded_dict['entity_emb'], self.loaded_dict['relation_emb']

    def infer_all(self, smis, tasks, batch_size, device):
        if device is None:
            device = self.args["device"]
        tasks = set(tasks) & set(all_tasks)
        # smiles = list(set(smis))
        smiles = list(set([tuple(pair) for pair in smis]))

        valid_triplets = None
        tasks_results = {}
        invalid = []
        id_desc_dict = {}
        for task_name in tasks:
            # pdb.set_trace()
            loader, invalid, valid_triplets, id_desc = get_batch(smiles, batch_size=batch_size, n_job=1,
                                                                 name=task_name)
            tasks_results[task_name] = self.infer_one_task(task_name, device, loader)
            id_desc_dict[task_name] = id_desc
        return self.format(valid_triplets, tasks_results, invalid, id_desc_dict)

    def format(self, smiles_valid, tasks_results, invalid_smiles, id_desc_dict):
        arr = []
        # pdb.set_trace()
        smiles_pairs = [[pair[0], pair[1]] for pair in smiles_valid]
        for i, (smiles1, smiles2) in enumerate(smiles_pairs):  # 使用 enumerate 获取索引 i 和解包元组
            tasks = []

            # 遍历 tasks_results 处理每个任务
            for task_name, results in tasks_results.items():
                task_keys = list(id_desc_dict[task_name])

                # 根据结果筛选
                task_result = [float(results[i][0])]

                tasks.append({
                    "task_name": task_name,
                    "result": task_result
                })

            # 构建每个 `smiles` 的字典
            arr.append({
                "smiles1": smiles1,  # 第一个元素
                "smiles2": smiles2,  # 第二个元素
                "valid": True,
                "result": 0 if smiles1 == smiles2 else float(results[i][0])
            })
            # pdb.set_trace()

        for i in range(len(invalid_smiles)):
            arr.append({
                "smiles1": invalid_smiles[i][0],
                "smiles2": invalid_smiles[i][1],
                "valid": False
            })
        # print(json.dumps(arr))
        return arr

    def infer_one_task(self, task_name, device, dataloader):

        # Load checkpoint
        checkpoint = torch.load("./algorithm/herb_drug_interaction/checkpoints/hdi_best_model.pth")

        encoder = KGMPNN(self.args, self.entity_emb, self.relation_emb).to(device)
        # encoder.load_state_dict(torch.load(f"./pretrained/{task_name}/KGMPNN.model", map_location=device))
        # Strip "module." prefix from encoder state_dict if necessary
        encoder_state_dict = checkpoint['encoder']
        new_encoder_state_dict = {k.replace("module.", ""): v for k, v in encoder_state_dict.items()}

        # Load the modified encoder state dictionary
        encoder.load_state_dict(new_encoder_state_dict)

        rel_dim = sentence_to_emb((0, "aaa"))[1].shape[0]
        predictor = PairReadout(encoder.out_dim, n_iters=6, n_layers=3, rel_dim=rel_dim).to(device)
        # predictor.load_state_dict(torch.load(f"./pretrained/{task_name}/Predictor.model", map_location=device))

        # Strip "module." prefix from readout state_dict if necessary
        readout_state_dict = checkpoint['readout']
        new_readout_state_dict = {k.replace("module.", ""): v for k, v in readout_state_dict.items()}

        # Load the modified readout state dictionary
        predictor.load_state_dict(new_readout_state_dict)

        encoder.eval()
        predictor.eval()
        probas_pred = []

        with torch.no_grad():
            for batch_id, batch_data in enumerate(dataloader):
                pos_tri = batch_data
                bg_pos_h = encoder(pos_tri[0].to(device))
                bg_pos_t = encoder(pos_tri[1].to(device))
                bg_pos_cb = encoder(pos_tri[2].to(device))

                p_score = predictor(pos_tri[0].to(device), bg_pos_h.to(device), pos_tri[1].to(device),
                                    bg_pos_t.to(device),
                                    pos_tri[2].to(device), bg_pos_cb.to(device), pos_tri[3].to(device))
                probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        # pdb.set_trace()
        result = np.concatenate(probas_pred)
        result = result.reshape(len(dataloader.dataset), -1)
        return result
    
def predict_ddi(input_pairs, device):
    args = get_args()
    i = Inference(args=args)
    tmp_result = i.infer_all(input_pairs, ['hdi'], batch_size=24, device=device)
    final_results = []
    # Iterate over input pairs and match with API output
    for pair in input_pairs:
        for output in tmp_result:
            # If the SMILES strings from input match the output, append the result to final_results
            # pdb.set_trace()
            if (pair[0] == output['smiles1'] and pair[1] == output['smiles2']) or \
               (pair[0] == output['smiles2'] and pair[1] == output['smiles1']):
                final_results.append(output['result'])
                break

    return (final_results)


if __name__ == '__main__':
    input_pairs = [["COC1=C2OC(=O)C=CC2=CC2=C1OC=C2",
                  "COC(=O)CCC1=C2NC(\\C=C3/N=C(/C=C4\\N\\C(=C/C5=N/C(=C\\2)/C(CCC(O)=O)=C5C)C(C=C)=C4C)C(\\C(=O)OC)=C3C)=C1C"],
                  ["O=C1N(CC2=CC=CC=C2)C2C[S+]3CCCC3C2N1CC1=CC=CC=C1", "CN1CCN2C(C1)C1=CC=CC=C1CC1=CC=CC=C21"],
                 ["C[C@H](CC1=CC=CC=C1)N(C)CC#C", "C[C@H](CC1=CC=CC=C1)N(C)CC#C"]]
    device = 'cuda:0'
    final_results = predict_ddi(input_pairs, device)
    print(final_results)

