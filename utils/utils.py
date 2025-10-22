import csv
import os

import pandas as pd


def read_target_list():
    target_list_dataset_path = './dataset/sorted_target_dataset.csv'
    if not os.path.exists(target_list_dataset_path):
        return None
    with open(target_list_dataset_path, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        rows = list(csv_reader)
    return {
        'header': header,
        'target_list': rows
    }


def read_cell_list():
    cell_list_dataset_path = './dataset/sorted_cell_dataset.csv'
    if not os.path.exists(cell_list_dataset_path):
        return None
    with open(cell_list_dataset_path, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        rows = list(csv_reader)
    return {
        'header': header,
        'cell_list': rows
    }


def read_herb_name_list():
    herb_name_dataset_path = './dataset/herb_name_dataset.csv'
    if not os.path.exists(herb_name_dataset_path):
        return None
    herb_name_dataset = pd.read_csv(herb_name_dataset_path)
    herb_name_list = []
    for row in herb_name_dataset.itertuples(index=False):
        herb_name_list.append(
            {
                'name': row.name,
                'pinyin': row.pinyin
            }
        )
    return herb_name_list
