from algorithm.herb_drug_interaction.main_hdi import predict_hdi, is_herb


def predict_hhi_only(input_herbs, device):
    for herb in input_herbs:
        if not is_herb(herb):
            raise ValueError(f"{herb} is not a herb")
    # Create pairs of herbs
    herb_pairs = []
    for i in range(len(input_herbs)):
        for j in range(i + 1, len(input_herbs)):
            herb_pairs.append([input_herbs[i], input_herbs[j]])

    # Pass the pairs to predict_hdi
    results = predict_hdi(herb_pairs, device)
    # Sort results by the 'result' value in descending order
    results = sorted(results, key=lambda x: x['result'], reverse=True)
    value_max = max([result['result'] for result in results])
    final_results = {'herbs': input_herbs, 'value_max': value_max, 'details': results}
    return final_results


if __name__ == '__main__':
    import json
    # input_herbs = [["COC1=C2OC(=O)C=CC2=CC2=C1OC=C2", "COC(=O)CCC1=C2NC(\\C=C3/N=C(/C=C4\\N\\C(=C/C5=N/C(=C\\2)/C(CCC(O)=O)=C5C)C(C=C)=C4C)C(\\C(=O)OC)=C3C)=C1C"]]
    input_herbs = ["HUA XU GENG BAI MAI GEN", "KU DING CHA", "shi jue ming"]
    # input_herbs = ["HUA XU GENG BAI MAI GEN", "KU DING CHA", "Invalid herb"]
    device = 'cuda:0'
    results = predict_hhi_only(input_herbs, device)
    with open("/home/junqi/project/baishenglai_backend/temp/hhi_temp.json", "w") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)
    print("HHI results:")
    for result in results['details']:
        print(result)
    print(f"Value results: {results['value_max']}")

"""
{'herbs': ["HUA XU GENG BAI MAI GEN", "KU DING CHA", "shi jue ming"],
 'value_max': 0.9998929500579834,
 'details': [
                {'drug1': 'HUA XU GENG BAI MAI GEN', 'drug2': 'shi jue ming', 'valid': True, 
                'result': 0.9825770258903503, 'details': [{'smiles1': 'CS(=O)/C=C/CCN=C=S', 'smiles2': 'CCC(=O)O[C@@](CC1=CC=CC=C1)(C2=CC=CC=C2)[C@H](C)CN(C)C.CC(=O)OC1=CC=CC=C1C(=O)O.CN1C=NC2=C1C(=O)N(C(=O)N2C)C.Cl', 'result': 0.9825770258903503}, {'smiles1': 'COC1=C(C2=C(C=C1)C3=C(C4=C(O3)C=C(C=C4)O)C(=O)O2)O', 'smiles2': 'CCC(=O)O[C@@](CC1=CC=CC=C1)(C2=CC=CC=C2)[C@H](C)CN(C)C.CC(=O)OC1=CC=CC=C1C(=O)O.CN1C=NC2=C1C(=O)N(C(=O)N2C)C.Cl', 'result': 0.9752030372619629}, {'smiles1': 'CS(=O)/C=C/CCN=C=S', 'smiles2': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'result': 0.6389641761779785}, {'smiles1': 'CS(=O)/C=C/CCN=C=S', 'smiles2': 'C/C=C(/C)\\C(=O)O[C@H]1C/C(=C\\CC/C(=C/[C@@H]2[C@H]1C(=C)C(=O)O2)/C)/C(=O)OC', 'result': 0.6234221458435059}, {'smiles1': 'C([C@@H](C(=O)O)N)S/C(=C/Cl)/Cl', 'smiles2': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'result': 0.5930671691894531}, {'smiles1': 'COC1=C(C2=C(C=C1)C3=C(C4=C(O3)C=C(C=C4)O)C(=O)O2)O', 'smiles2': 'C/C=C(/C)\\C(=O)O[C@H]1C/C(=C\\CC/C(=C/[C@@H]2[C@H]1C(=C)C(=O)O2)/C)/C(=O)OC', 'result': 0.5435184240341187}, {'smiles1': 'C([C@@H](C(=O)O)N)S/C(=C/Cl)/Cl', 'smiles2': 'CCC(=O)O[C@@](CC1=CC=CC=C1)(C2=CC=CC=C2)[C@H](C)CN(C)C.CC(=O)OC1=CC=CC=C1C(=O)O.CN1C=NC2=C1C(=O)N(C(=O)N2C)C.Cl', 'result': 0.4861294627189636}, {'smiles1': 'COC1=C(C2=C(C=C1)C3=C(C4=C(O3)C=C(C=C4)O)C(=O)O2)O', 'smiles2': 'CCCN(CC)C(=O)C1=CN=C2C(=C1)C=CC3=C2NC=C(C3=O)[N+](=O)[O-]', 'result': 0.38868287205696106}, {'smiles1': 'CS(=O)/C=C/CCN=C=S', 'smiles2': 'CCCN(CC)C(=O)C1=CN=C2C(=C1)C=CC3=C2NC=C(C3=O)[N+](=O)[O-]', 'result': 0.3733540177345276}, {'smiles1': 'COC1=C(C2=C(C=C1)C3=C(C4=C(O3)C=C(C=C4)O)C(=O)O2)O', 'smiles2': 'C[C@]12CC[C@](C[C@H]1C3=CC(=O)[C@@H]4[C@]5(CC[C@@H](C([C@@H]5CC[C@]4([C@@]3(CC2)C)C)(C)C)O)C)(C)C(=O)O', 'result': 0.3697476387023926}, {'smiles1': 'C([C@@H](C(=O)O)N)S/C(=C/Cl)/Cl', 'smiles2': 'C/C=C(/C)\\C(=O)O[C@H]1C/C(=C\\CC/C(=C/[C@@H]2[C@H]1C(=C)C(=O)O2)/C)/C(=O)OC', 'result': 0.35408422350883484}, {'smiles1': 'COC1=C(C2=C(C=C1)C3=C(C4=C(O3)C=C(C=C4)O)C(=O)O2)O', 'smiles2': 'C1=CC(=CN=C1)CC(O)(P(=O)(O)O)P(=O)(O)O.C(=O)([O-])[O-].[Ca+2]', 'result': 0.3236284852027893}, {'smiles1': 'COC1=C(C2=C(C=C1)C3=C(C4=C(O3)C=C(C=C4)O)C(=O)O2)O', 'smiles2': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'result': 0.27413424849510193}, {'smiles1': 'CS(=O)/C=C/CCN=C=S', 'smiles2': 'C[C@]12CC[C@](C[C@H]1C3=CC(=O)[C@@H]4[C@]5(CC[C@@H](C([C@@H]5CC[C@]4([C@@]3(CC2)C)C)(C)C)O)C)(C)C(=O)O', 'result': 0.2605663537979126}, {'smiles1': 'CS(=O)/C=C/CCN=C=S', 'smiles2': 'CCCCO[14C](=O)C1=CC=CC=C1[14C](=O)OCCCC', 'result': 0.21757563948631287}, {'smiles1': 'CS(=O)/C=C/CCN=C=S', 'smiles2': 'CCCCOC(=O)C1=CC=CC=C1C(=O)OCCCC', 'result': 0.21757563948631287}, {'smiles1': 'COC1=C(C2=C(C=C1)C3=C(C4=C(O3)C=C(C=C4)O)C(=O)O2)O', 'smiles2': 'CCCCO[14C](=O)C1=CC=CC=C1[14C](=O)OCCCC', 'result': 0.10307257622480392}, {'smiles1': 'COC1=C(C2=C(C=C1)C3=C(C4=C(O3)C=C(C=C4)O)C(=O)O2)O', 'smiles2': 'CCCCOC(=O)C1=CC=CC=C1C(=O)OCCCC', 'result': 0.10307248681783676}, {'smiles1': 'C([C@@H](C(=O)O)N)S/C(=C/Cl)/Cl', 'smiles2': 'C[C@]12CC[C@](C[C@H]1C3=CC(=O)[C@@H]4[C@]5(CC[C@@H](C([C@@H]5CC[C@]4([C@@]3(CC2)C)C)(C)C)O)C)(C)C(=O)O', 'result': 0.09501258283853531}, {'smiles1': 'C([C@@H](C(=O)O)N)S/C(=C/Cl)/Cl', 'smiles2': 'CCCCO[14C](=O)C1=CC=CC=C1[14C](=O)OCCCC', 'result': 0.08691968023777008}, {'smiles1': 'C([C@@H](C(=O)O)N)S/C(=C/Cl)/Cl', 'smiles2': 'CCCCOC(=O)C1=CC=CC=C1C(=O)OCCCC', 'result': 0.08691968023777008}, {'smiles1': 'C([C@@H](C(=O)O)N)S/C(=C/Cl)/Cl', 'smiles2': 'CCCN(CC)C(=O)C1=CN=C2C(=C1)C=CC3=C2NC=C(C3=O)[N+](=O)[O-]', 'result': 0.055580366402864456}, {'smiles1': 'COC1=C(C2=C(C=C1)C3=C(C4=C(O3)C=C(C=C4)O)C(=O)O2)O', 'smiles2': 'C(=O)([O-])[O-].[Ca+2]', 'result': 0.04715418443083763}, {'smiles1': 'CS(=O)/C=C/CCN=C=S', 'smiles2': 'C1=CC(=CN=C1)CC(O)(P(=O)(O)O)P(=O)(O)O.C(=O)([O-])[O-].[Ca+2]', 'result': 0.025929396972060204}, {'smiles1': 'CS(=O)/C=C/CCN=C=S', 'smiles2': 'C(=O)([O-])[O-].[Ca+2]', 'result': 0.02311028353869915}, {'smiles1': 'C([C@@H](C(=O)O)N)S/C(=C/Cl)/Cl', 'smiles2': 'C(=O)([O-])[O-].[Ca+2]', 'result': 0.005479243118315935}, {'smiles1': 'C([C@@H](C(=O)O)N)S/C(=C/Cl)/Cl', 'smiles2': 'C1=CC(=CN=C1)CC(O)(P(=O)(O)O)P(=O)(O)O.C(=O)([O-])[O-].[Ca+2]', 'result': 0.00289302715100348}]}
                {第二对}
                {第三对}
            ]
}
"""