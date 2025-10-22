from algorithm.herb_drug_interaction.main_ddi import predict_ddi
import os
import pdb
import json


# Helper function to check if the input is a valid SMILES
def is_smile(drug):
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(drug)
        return mol is not None
    except:
        return False


# Helper function to check if the input is an available herb from the CSV file
def is_herb(drug):
    import pandas as pd
    available_herb_csv_path = './algorithm/herb_drug_interaction/data/herb2cids.csv'
    herbs_df = pd.read_csv(available_herb_csv_path)
    herbs_list = herbs_df.iloc[:, 0].str.lower().tolist()  # assuming herbs are in the first column
    return drug.lower() in herbs_list


# Helper function to get SMILES from a herb
def get_smiles_list_from_herb(herb):
    import pandas as pd
    herb_2_cid = './algorithm/herb_drug_interaction/data/herb2cids.csv'
    cid_2_smile = './algorithm/herb_drug_interaction/data/cid2smiles.csv'
    # pdb.set_trace()
    # Get CIDs for the herb
    herb_cid_df = pd.read_csv(herb_2_cid)
    cids_list = herb_cid_df[herb_cid_df.iloc[:, 0].str.lower() == herb.lower()].iloc[:, -1].values[0].split('|')

    # Get SMILES for the CIDs
    cid_smile_df = pd.read_csv(cid_2_smile)
    # Ensure the 'CID' column is a string and strip any extra spaces
    cid_smile_df['CID'] = cid_smile_df['CID'].astype(str).str.strip()

    # Ensure all CIDs in cids_list are strings as well
    cids_list = [str(cid).strip() for cid in cids_list]

    # Check if there are any missing CIDs and log them for debugging
    missing_cids = [cid for cid in cids_list if cid not in cid_smile_df['CID'].values]
    if missing_cids:
        print(f"Warning: The following CIDs are missing from cid2smile.csv: {missing_cids}")

    # Now perform the filtering to get the SMILES
    smiles_list = cid_smile_df[cid_smile_df['CID'].isin(cids_list)]['IsomericSMILES'].tolist()

    # print(smiles_list)
    # pdb.set_trace()
    return smiles_list


# Main function to predict herb-drug interaction
def predict_hdi(drug_pairs, device):
    final_results = []

    for i, (drug1, drug2) in enumerate(drug_pairs):
        if is_smile(drug1) and is_smile(drug2):
            # Both are SMILES
            result = predict_ddi([[drug1, drug2]], device)
            final_results.append({
                "drug1": drug1,
                "drug2": drug2,
                "valid": True,
                "result": result,
                "details": None
            })

        elif is_herb(drug1) and is_smile(drug2):
            # drug1 is herb, drug2 is SMILES
            smiles1_list = get_smiles_list_from_herb(drug1)
            smile_pairs = [[smile1, drug2] for smile1 in smiles1_list]
            tmp_results = predict_ddi(smile_pairs, device)
            result = max(tmp_results)  # Get the maximum interaction probability
            details = [{"smiles1": smiles1_list[i], "smiles2": drug2, "result": tmp_result}
                       for i, tmp_result in enumerate(tmp_results)]
            details.sort(key=lambda x: x['result'], reverse=True)
            final_results.append({
                "drug1": drug1,
                "drug2": drug2,
                "valid": True,
                "result": result,
                "details": details
            })

        elif is_smile(drug1) and is_herb(drug2):
            # drug1 is SMILES, drug2 is herb
            smiles2_list = get_smiles_list_from_herb(drug2)
            smile_pairs = [[drug1, smile2] for smile2 in smiles2_list]
            # pdb.set_trace()
            tmp_results = predict_ddi(smile_pairs, device)
            result = max(tmp_results)
            details = [{"smiles1": drug1, "smiles2": smiles2_list[i], "result": tmp_result}
                       for i, tmp_result in enumerate(tmp_results)]
            details.sort(key=lambda x: x['result'], reverse=True)
            final_results.append({
                "drug1": drug1,
                "drug2": drug2,
                "valid": True,
                "result": result,
                "details": details
            })

        elif is_herb(drug1) and is_herb(drug2):
            # Both are herbs
            # Create a cache key by sorting the herbs to ensure consistent key regardless of order
            sorted_herbs = sorted([drug1.lower(), drug2.lower()])
            cache_file = f"./algorithm/herb_drug_interaction/cache/hhi_{sorted_herbs[0]}_{sorted_herbs[1]}.json"

            # Try to load from cache first
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_result = json.load(f)
                final_results.append(cached_result)
            else:
                # Calculate if not in cache
                smiles1_list = get_smiles_list_from_herb(drug1)
                smiles2_list = get_smiles_list_from_herb(drug2)
                smile_pairs = [[smile1, smile2] for smile1 in smiles1_list for smile2 in smiles2_list]
                tmp_results = predict_ddi(smile_pairs, device)
                result = max(tmp_results)
                details = [{"smiles1": smiles1_list[i // len(smiles2_list)],
                            "smiles2": smiles2_list[i % len(smiles2_list)],
                            "result": tmp_result}
                           for i, tmp_result in enumerate(tmp_results)]
                details.sort(key=lambda x: x['result'], reverse=True)

                result_dict = {
                    "drug1": drug1,
                    "drug2": drug2,
                    "valid": True,
                    "result": result,
                    "details": details
                }

                # Save to cache
                os.makedirs("./algorithm/herb_drug_interaction/cache", exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(result_dict, f)

                final_results.append(result_dict)

        else:
            # Invalid input
            final_results.append({
                "drug1": drug1,
                "drug2": drug2,
                "valid": False
            })

    return final_results


if __name__ == '__main__':
    # Test cases
    drug_pairs = [
        # ["O=C1N(CC2=CC=CC=C2)C2C[S+]3CCCC3C2N1CC1=CC=CC=C1", "CN1CCN2C(C1)C1=CC=CC=C1CC1=CC=CC=C21"],
        ["COC1=C2OC(=O)C=CC2=CC2=C1OC=C2", "shi jue ming"],
        # ["HUA XU GENG BAI MAI GEN", "KU DING CHA"],
        ["invalid", "smiles"]
    ]
    device = 'cuda:0'
    results = predict_hdi(drug_pairs, device)
    for result in results:
        print(result)

# 示例输出
"""
[
# {'drug1': 'O=C1N(CC2=CC=CC=C2)C2C[S+]3CCCC3C2N1CC1=CC=CC=C1', 'drug2': 'CN1CCN2C(C1)C1=CC=CC=C1CC1=CC=CC=C21', 'valid': True, 'result': [2.1045141693321057e-05], 'details': None},
{'drug1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'drug2': 'shi jue ming', 'valid': True, 'result': 0.5477972030639648, 'details': [{'smiles1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'smiles2': 'CCC(=O)O[C@@](CC1=CC=CC=C1)(C2=CC=CC=C2)[C@H](C)CN(C)C.CC(=O)OC1=CC=CC=C1C(=O)O.CN1C=NC2=C1C(=O)N(C(=O)N2C)C.Cl', 'result': 0.5477972030639648}, {'smiles1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'smiles2': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'result': 0.35037943720817566}, {'smiles1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'smiles2': 'C[C@]12CC[C@](C[C@H]1C3=CC(=O)[C@@H]4[C@]5(CC[C@@H](C([C@@H]5CC[C@]4([C@@]3(CC2)C)C)(C)C)O)C)(C)C(=O)O', 'result': 0.3169381618499756}, {'smiles1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'smiles2': 'C(=O)([O-])[O-].[Ca+2]', 'result': 0.1641145497560501}, {'smiles1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'smiles2': 'C/C=C(/C)\\C(=O)O[C@H]1C/C(=C\\CC/C(=C/[C@@H]2[C@H]1C(=C)C(=O)O2)/C)/C(=O)OC', 'result': 0.15211258828639984}, {'smiles1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'smiles2': 'C1=CC(=CN=C1)CC(O)(P(=O)(O)O)P(=O)(O)O.C(=O)([O-])[O-].[Ca+2]', 'result': 0.09224428236484528}, {'smiles1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'smiles2': 'CCCN(CC)C(=O)C1=CN=C2C(=C1)C=CC3=C2NC=C(C3=O)[N+](=O)[O-]', 'result': 0.04439326375722885}, {'smiles1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'smiles2': 'CCCCO[14C](=O)C1=CC=CC=C1[14C](=O)OCCCC', 'result': 0.023744523525238037}, {'smiles1': 'COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'smiles2': 'CCCCOC(=O)C1=CC=CC=C1C(=O)OCCCC', 'result': 0.023744523525238037}]},
{'drug1': 'invalid', 'drug2': 'smiles', 'valid': False}
]
""" 