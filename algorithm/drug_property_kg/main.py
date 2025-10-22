# -*- coding: utf-8 -*-
# @Time    : 2024/9/2 上午9:40
# @Author  : Chen Mukun
# @File    : main.py
# @Software: PyCharm
# @desc    :
import pickle
from argparse import ArgumentParser

import numpy as np
from torch.utils.data.dataloader import DataLoader
import dgl
from torch.utils.data import Dataset
import torch
from dgllife.utils.io import pmap
from rdkit import Chem
import json

from tqdm import tqdm

from algorithm.drug_property_kg.model.layer.mpnn import KGMPNN
from algorithm.drug_property_kg.model.layer.readout import Set2Set
from algorithm.drug_property_kg.model.predictor.predictor import Predictor
from collections import defaultdict

cls_tasks = ['BBBP', 'Tox21', 'SIDER', 'ClinTox', 'BACE', 'MUV', 'HIV']
task_properties = {
    'BACE': ['Class'],
    'BBBP': ['p_np'],
    'ClinTox': ['FDA_APPROVED', 'CT_TOX'],
    'SIDER': ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
              'Investigations', 'Musculoskeletal and connective tissue disorders', 'Gastrointestinal disorders',
              'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders',
              'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
              'General disorders and administration site conditions', 'Endocrine disorders',
              'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
              'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders',
              'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
              'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions',
              'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders',
              'Injury, poisoning and procedural complications'],
    'Tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
              'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
    'ToxCast': ['ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive', 'APR_HepG2_CellCycleArrest_24h_dn',
                'APR_HepG2_CellCycleArrest_24h_up', 'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn',
                'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_24h_dn', 'APR_HepG2_MicrotubuleCSK_24h_up',
                'APR_HepG2_MicrotubuleCSK_72h_dn', 'APR_HepG2_MicrotubuleCSK_72h_up', 'APR_HepG2_MitoMass_24h_dn',
                'APR_HepG2_MitoMass_24h_up', 'APR_HepG2_MitoMass_72h_dn', 'APR_HepG2_MitoMass_72h_up',
                'APR_HepG2_MitoMembPot_1h_dn', 'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn',
                'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up', 'APR_HepG2_NuclearSize_24h_dn',
                'APR_HepG2_NuclearSize_72h_dn', 'APR_HepG2_NuclearSize_72h_up', 'APR_HepG2_OxidativeStress_24h_up',
                'APR_HepG2_OxidativeStress_72h_up', 'APR_HepG2_StressKinase_1h_up', 'APR_HepG2_StressKinase_24h_up',
                'APR_HepG2_StressKinase_72h_up', 'APR_HepG2_p53Act_24h_up', 'APR_HepG2_p53Act_72h_up',
                'APR_Hepat_Apoptosis_24hr_up', 'APR_Hepat_Apoptosis_48hr_up', 'APR_Hepat_CellLoss_24hr_dn',
                'APR_Hepat_CellLoss_48hr_dn', 'APR_Hepat_DNADamage_24hr_up', 'APR_Hepat_DNADamage_48hr_up',
                'APR_Hepat_DNATexture_24hr_up', 'APR_Hepat_DNATexture_48hr_up', 'APR_Hepat_MitoFxnI_1hr_dn',
                'APR_Hepat_MitoFxnI_24hr_dn', 'APR_Hepat_MitoFxnI_48hr_dn', 'APR_Hepat_NuclearSize_24hr_dn',
                'APR_Hepat_NuclearSize_48hr_dn', 'APR_Hepat_Steatosis_24hr_up', 'APR_Hepat_Steatosis_48hr_up',
                'ATG_AP_1_CIS_dn', 'ATG_AP_1_CIS_up', 'ATG_AP_2_CIS_dn', 'ATG_AP_2_CIS_up', 'ATG_AR_TRANS_dn',
                'ATG_AR_TRANS_up', 'ATG_Ahr_CIS_dn', 'ATG_Ahr_CIS_up', 'ATG_BRE_CIS_dn', 'ATG_BRE_CIS_up',
                'ATG_CAR_TRANS_dn', 'ATG_CAR_TRANS_up', 'ATG_CMV_CIS_dn', 'ATG_CMV_CIS_up', 'ATG_CRE_CIS_dn',
                'ATG_CRE_CIS_up', 'ATG_C_EBP_CIS_dn', 'ATG_C_EBP_CIS_up', 'ATG_DR4_LXR_CIS_dn', 'ATG_DR4_LXR_CIS_up',
                'ATG_DR5_CIS_dn', 'ATG_DR5_CIS_up', 'ATG_E2F_CIS_dn', 'ATG_E2F_CIS_up', 'ATG_EGR_CIS_up',
                'ATG_ERE_CIS_dn', 'ATG_ERE_CIS_up', 'ATG_ERRa_TRANS_dn', 'ATG_ERRg_TRANS_dn', 'ATG_ERRg_TRANS_up',
                'ATG_ERa_TRANS_up', 'ATG_E_Box_CIS_dn', 'ATG_E_Box_CIS_up', 'ATG_Ets_CIS_dn', 'ATG_Ets_CIS_up',
                'ATG_FXR_TRANS_up', 'ATG_FoxA2_CIS_dn', 'ATG_FoxA2_CIS_up', 'ATG_FoxO_CIS_dn', 'ATG_FoxO_CIS_up',
                'ATG_GAL4_TRANS_dn', 'ATG_GATA_CIS_dn', 'ATG_GATA_CIS_up', 'ATG_GLI_CIS_dn', 'ATG_GLI_CIS_up',
                'ATG_GRE_CIS_dn', 'ATG_GRE_CIS_up', 'ATG_GR_TRANS_dn', 'ATG_GR_TRANS_up', 'ATG_HIF1a_CIS_dn',
                'ATG_HIF1a_CIS_up', 'ATG_HNF4a_TRANS_dn', 'ATG_HNF4a_TRANS_up', 'ATG_HNF6_CIS_dn', 'ATG_HNF6_CIS_up',
                'ATG_HSE_CIS_dn', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn', 'ATG_IR1_CIS_up', 'ATG_ISRE_CIS_dn',
                'ATG_ISRE_CIS_up', 'ATG_LXRa_TRANS_dn', 'ATG_LXRa_TRANS_up', 'ATG_LXRb_TRANS_dn', 'ATG_LXRb_TRANS_up',
                'ATG_MRE_CIS_up', 'ATG_M_06_TRANS_up', 'ATG_M_19_CIS_dn', 'ATG_M_19_TRANS_dn', 'ATG_M_19_TRANS_up',
                'ATG_M_32_CIS_dn', 'ATG_M_32_CIS_up', 'ATG_M_32_TRANS_dn', 'ATG_M_32_TRANS_up', 'ATG_M_61_TRANS_up',
                'ATG_Myb_CIS_dn', 'ATG_Myb_CIS_up', 'ATG_Myc_CIS_dn', 'ATG_Myc_CIS_up', 'ATG_NFI_CIS_dn',
                'ATG_NFI_CIS_up', 'ATG_NF_kB_CIS_dn', 'ATG_NF_kB_CIS_up', 'ATG_NRF1_CIS_dn', 'ATG_NRF1_CIS_up',
                'ATG_NRF2_ARE_CIS_dn', 'ATG_NRF2_ARE_CIS_up', 'ATG_NURR1_TRANS_dn', 'ATG_NURR1_TRANS_up',
                'ATG_Oct_MLP_CIS_dn', 'ATG_Oct_MLP_CIS_up', 'ATG_PBREM_CIS_dn', 'ATG_PBREM_CIS_up',
                'ATG_PPARa_TRANS_dn', 'ATG_PPARa_TRANS_up', 'ATG_PPARd_TRANS_up', 'ATG_PPARg_TRANS_up',
                'ATG_PPRE_CIS_dn', 'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn', 'ATG_PXRE_CIS_up', 'ATG_PXR_TRANS_dn',
                'ATG_PXR_TRANS_up', 'ATG_Pax6_CIS_up', 'ATG_RARa_TRANS_dn', 'ATG_RARa_TRANS_up', 'ATG_RARb_TRANS_dn',
                'ATG_RARb_TRANS_up', 'ATG_RARg_TRANS_dn', 'ATG_RARg_TRANS_up', 'ATG_RORE_CIS_dn', 'ATG_RORE_CIS_up',
                'ATG_RORb_TRANS_dn', 'ATG_RORg_TRANS_dn', 'ATG_RORg_TRANS_up', 'ATG_RXRa_TRANS_dn', 'ATG_RXRa_TRANS_up',
                'ATG_RXRb_TRANS_dn', 'ATG_RXRb_TRANS_up', 'ATG_SREBP_CIS_dn', 'ATG_SREBP_CIS_up', 'ATG_STAT3_CIS_dn',
                'ATG_STAT3_CIS_up', 'ATG_Sox_CIS_dn', 'ATG_Sox_CIS_up', 'ATG_Sp1_CIS_dn', 'ATG_Sp1_CIS_up',
                'ATG_TAL_CIS_dn', 'ATG_TAL_CIS_up', 'ATG_TA_CIS_dn', 'ATG_TA_CIS_up', 'ATG_TCF_b_cat_CIS_dn',
                'ATG_TCF_b_cat_CIS_up', 'ATG_TGFb_CIS_dn', 'ATG_TGFb_CIS_up', 'ATG_THRa1_TRANS_dn',
                'ATG_THRa1_TRANS_up', 'ATG_VDRE_CIS_dn', 'ATG_VDRE_CIS_up', 'ATG_VDR_TRANS_dn', 'ATG_VDR_TRANS_up',
                'ATG_XTT_Cytotoxicity_up', 'ATG_Xbp1_CIS_dn', 'ATG_Xbp1_CIS_up', 'ATG_p53_CIS_dn', 'ATG_p53_CIS_up',
                'BSK_3C_Eselectin_down', 'BSK_3C_HLADR_down', 'BSK_3C_ICAM1_down', 'BSK_3C_IL8_down',
                'BSK_3C_MCP1_down', 'BSK_3C_MIG_down', 'BSK_3C_Proliferation_down', 'BSK_3C_SRB_down',
                'BSK_3C_Thrombomodulin_down', 'BSK_3C_Thrombomodulin_up', 'BSK_3C_TissueFactor_down',
                'BSK_3C_TissueFactor_up', 'BSK_3C_VCAM1_down', 'BSK_3C_Vis_down', 'BSK_3C_uPAR_down',
                'BSK_4H_Eotaxin3_down', 'BSK_4H_MCP1_down', 'BSK_4H_Pselectin_down', 'BSK_4H_Pselectin_up',
                'BSK_4H_SRB_down', 'BSK_4H_VCAM1_down', 'BSK_4H_VEGFRII_down', 'BSK_4H_uPAR_down', 'BSK_4H_uPAR_up',
                'BSK_BE3C_HLADR_down', 'BSK_BE3C_IL1a_down', 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down',
                'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up', 'BSK_BE3C_PAI1_down', 'BSK_BE3C_SRB_down',
                'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down', 'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPAR_up',
                'BSK_BE3C_uPA_down', 'BSK_CASM3C_HLADR_down', 'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL6_up',
                'BSK_CASM3C_IL8_down', 'BSK_CASM3C_LDLR_down', 'BSK_CASM3C_LDLR_up', 'BSK_CASM3C_MCP1_down',
                'BSK_CASM3C_MCP1_up', 'BSK_CASM3C_MCSF_down', 'BSK_CASM3C_MCSF_up', 'BSK_CASM3C_MIG_down',
                'BSK_CASM3C_Proliferation_down', 'BSK_CASM3C_Proliferation_up', 'BSK_CASM3C_SAA_down',
                'BSK_CASM3C_SAA_up', 'BSK_CASM3C_SRB_down', 'BSK_CASM3C_Thrombomodulin_down',
                'BSK_CASM3C_Thrombomodulin_up', 'BSK_CASM3C_TissueFactor_down', 'BSK_CASM3C_VCAM1_down',
                'BSK_CASM3C_VCAM1_up', 'BSK_CASM3C_uPAR_down', 'BSK_CASM3C_uPAR_up', 'BSK_KF3CT_ICAM1_down',
                'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down', 'BSK_KF3CT_IP10_up', 'BSK_KF3CT_MCP1_down',
                'BSK_KF3CT_MCP1_up', 'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down', 'BSK_KF3CT_TGFb1_down',
                'BSK_KF3CT_TIMP2_down', 'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down', 'BSK_LPS_Eselectin_down',
                'BSK_LPS_Eselectin_up', 'BSK_LPS_IL1a_down', 'BSK_LPS_IL1a_up', 'BSK_LPS_IL8_down', 'BSK_LPS_IL8_up',
                'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down', 'BSK_LPS_PGE2_down', 'BSK_LPS_PGE2_up', 'BSK_LPS_SRB_down',
                'BSK_LPS_TNFa_down', 'BSK_LPS_TNFa_up', 'BSK_LPS_TissueFactor_down', 'BSK_LPS_TissueFactor_up',
                'BSK_LPS_VCAM1_down', 'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down', 'BSK_SAg_CD69_down',
                'BSK_SAg_Eselectin_down', 'BSK_SAg_Eselectin_up', 'BSK_SAg_IL8_down', 'BSK_SAg_IL8_up',
                'BSK_SAg_MCP1_down', 'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down', 'BSK_SAg_PBMCCytotoxicity_up',
                'BSK_SAg_Proliferation_down', 'BSK_SAg_SRB_down', 'BSK_hDFCGF_CollagenIII_down', 'BSK_hDFCGF_EGFR_down',
                'BSK_hDFCGF_EGFR_up', 'BSK_hDFCGF_IL8_down', 'BSK_hDFCGF_IP10_down', 'BSK_hDFCGF_MCSF_down',
                'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down', 'BSK_hDFCGF_MMP1_up', 'BSK_hDFCGF_PAI1_down',
                'BSK_hDFCGF_Proliferation_down', 'BSK_hDFCGF_SRB_down', 'BSK_hDFCGF_TIMP1_down',
                'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn', 'CEETOX_H295R_ANDR_dn', 'CEETOX_H295R_CORTISOL_dn',
                'CEETOX_H295R_DOC_dn', 'CEETOX_H295R_DOC_up', 'CEETOX_H295R_ESTRADIOL_dn', 'CEETOX_H295R_ESTRADIOL_up',
                'CEETOX_H295R_ESTRONE_dn', 'CEETOX_H295R_ESTRONE_up', 'CEETOX_H295R_OHPREG_up',
                'CEETOX_H295R_OHPROG_dn', 'CEETOX_H295R_OHPROG_up', 'CEETOX_H295R_PROG_up', 'CEETOX_H295R_TESTO_dn',
                'CLD_ABCB1_48hr', 'CLD_ABCG2_48hr', 'CLD_CYP1A1_24hr', 'CLD_CYP1A1_48hr', 'CLD_CYP1A1_6hr',
                'CLD_CYP1A2_24hr', 'CLD_CYP1A2_48hr', 'CLD_CYP1A2_6hr', 'CLD_CYP2B6_24hr', 'CLD_CYP2B6_48hr',
                'CLD_CYP2B6_6hr', 'CLD_CYP3A4_24hr', 'CLD_CYP3A4_48hr', 'CLD_CYP3A4_6hr', 'CLD_GSTA2_48hr',
                'CLD_SULT2A_24hr', 'CLD_SULT2A_48hr', 'CLD_UGT1A1_24hr', 'CLD_UGT1A1_48hr', 'NCCT_HEK293T_CellTiterGLO',
                'NCCT_QuantiLum_inhib_2_dn', 'NCCT_QuantiLum_inhib_dn', 'NCCT_TPO_AUR_dn', 'NCCT_TPO_GUA_dn',
                'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_ADME_hCYP19A1', 'NVS_ADME_hCYP1A1', 'NVS_ADME_hCYP1A2',
                'NVS_ADME_hCYP2A6', 'NVS_ADME_hCYP2B6', 'NVS_ADME_hCYP2C19', 'NVS_ADME_hCYP2C9', 'NVS_ADME_hCYP2D6',
                'NVS_ADME_hCYP3A4', 'NVS_ADME_hCYP4F12', 'NVS_ADME_rCYP2C12', 'NVS_ENZ_hAChE', 'NVS_ENZ_hAMPKa1',
                'NVS_ENZ_hAurA', 'NVS_ENZ_hBACE', 'NVS_ENZ_hCASP5', 'NVS_ENZ_hCK1D', 'NVS_ENZ_hDUSP3', 'NVS_ENZ_hES',
                'NVS_ENZ_hElastase', 'NVS_ENZ_hFGFR1', 'NVS_ENZ_hGSK3b', 'NVS_ENZ_hMMP1', 'NVS_ENZ_hMMP13',
                'NVS_ENZ_hMMP2', 'NVS_ENZ_hMMP3', 'NVS_ENZ_hMMP7', 'NVS_ENZ_hMMP9', 'NVS_ENZ_hPDE10', 'NVS_ENZ_hPDE4A1',
                'NVS_ENZ_hPDE5', 'NVS_ENZ_hPI3Ka', 'NVS_ENZ_hPTEN', 'NVS_ENZ_hPTPN11', 'NVS_ENZ_hPTPN12',
                'NVS_ENZ_hPTPN13', 'NVS_ENZ_hPTPN9', 'NVS_ENZ_hPTPRC', 'NVS_ENZ_hSIRT1', 'NVS_ENZ_hSIRT2',
                'NVS_ENZ_hTrkA', 'NVS_ENZ_hVEGFR2', 'NVS_ENZ_oCOX1', 'NVS_ENZ_oCOX2', 'NVS_ENZ_rAChE', 'NVS_ENZ_rCNOS',
                'NVS_ENZ_rMAOAC', 'NVS_ENZ_rMAOAP', 'NVS_ENZ_rMAOBC', 'NVS_ENZ_rMAOBP', 'NVS_ENZ_rabI2C',
                'NVS_GPCR_bAdoR_NonSelective', 'NVS_GPCR_bDR_NonSelective', 'NVS_GPCR_g5HT4', 'NVS_GPCR_gH2',
                'NVS_GPCR_gLTB4', 'NVS_GPCR_gLTD4', 'NVS_GPCR_gMPeripheral_NonSelective', 'NVS_GPCR_gOpiateK',
                'NVS_GPCR_h5HT2A', 'NVS_GPCR_h5HT5A', 'NVS_GPCR_h5HT6', 'NVS_GPCR_h5HT7', 'NVS_GPCR_hAT1',
                'NVS_GPCR_hAdoRA1', 'NVS_GPCR_hAdoRA2a', 'NVS_GPCR_hAdra2A', 'NVS_GPCR_hAdra2C', 'NVS_GPCR_hAdrb1',
                'NVS_GPCR_hAdrb2', 'NVS_GPCR_hAdrb3', 'NVS_GPCR_hDRD1', 'NVS_GPCR_hDRD2s', 'NVS_GPCR_hDRD4.4',
                'NVS_GPCR_hH1', 'NVS_GPCR_hLTB4_BLT1', 'NVS_GPCR_hM1', 'NVS_GPCR_hM2', 'NVS_GPCR_hM3', 'NVS_GPCR_hM4',
                'NVS_GPCR_hNK2', 'NVS_GPCR_hOpiate_D1', 'NVS_GPCR_hOpiate_mu', 'NVS_GPCR_hTXA2', 'NVS_GPCR_p5HT2C',
                'NVS_GPCR_r5HT1_NonSelective', 'NVS_GPCR_r5HT_NonSelective', 'NVS_GPCR_rAdra1B',
                'NVS_GPCR_rAdra1_NonSelective', 'NVS_GPCR_rAdra2_NonSelective', 'NVS_GPCR_rAdrb_NonSelective',
                'NVS_GPCR_rNK1', 'NVS_GPCR_rNK3', 'NVS_GPCR_rOpiate_NonSelective', 'NVS_GPCR_rOpiate_NonSelectiveNa',
                'NVS_GPCR_rSST', 'NVS_GPCR_rTRH', 'NVS_GPCR_rV1', 'NVS_GPCR_rabPAF', 'NVS_GPCR_rmAdra2B',
                'NVS_IC_hKhERGCh', 'NVS_IC_rCaBTZCHL', 'NVS_IC_rCaDHPRCh_L', 'NVS_IC_rNaCh_site2', 'NVS_LGIC_bGABARa1',
                'NVS_LGIC_h5HT3', 'NVS_LGIC_hNNR_NBungSens', 'NVS_LGIC_rGABAR_NonSelective', 'NVS_LGIC_rNNR_BungSens',
                'NVS_MP_hPBR', 'NVS_MP_rPBR', 'NVS_NR_bER', 'NVS_NR_bPR', 'NVS_NR_cAR', 'NVS_NR_hAR',
                'NVS_NR_hCAR_Antagonist', 'NVS_NR_hER', 'NVS_NR_hFXR_Agonist', 'NVS_NR_hFXR_Antagonist', 'NVS_NR_hGR',
                'NVS_NR_hPPARa', 'NVS_NR_hPPARg', 'NVS_NR_hPR', 'NVS_NR_hPXR', 'NVS_NR_hRAR_Antagonist',
                'NVS_NR_hRARa_Agonist', 'NVS_NR_hTRa_Antagonist', 'NVS_NR_mERa', 'NVS_NR_rAR', 'NVS_NR_rMR',
                'NVS_OR_gSIGMA_NonSelective', 'NVS_TR_gDAT', 'NVS_TR_hAdoT', 'NVS_TR_hDAT', 'NVS_TR_hNET',
                'NVS_TR_hSERT', 'NVS_TR_rNET', 'NVS_TR_rSERT', 'NVS_TR_rVMAT2', 'OT_AR_ARELUC_AG_1440',
                'OT_AR_ARSRC1_0480', 'OT_AR_ARSRC1_0960', 'OT_ER_ERaERa_0480', 'OT_ER_ERaERa_1440', 'OT_ER_ERaERb_0480',
                'OT_ER_ERaERb_1440', 'OT_ER_ERbERb_0480', 'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120',
                'OT_ERa_EREGFP_0480', 'OT_FXR_FXRSRC1_0480', 'OT_FXR_FXRSRC1_1440', 'OT_NURR1_NURR1RXRa_0480',
                'OT_NURR1_NURR1RXRa_1440', 'TOX21_ARE_BLA_Agonist_ch1', 'TOX21_ARE_BLA_Agonist_ch2',
                'TOX21_ARE_BLA_agonist_ratio', 'TOX21_ARE_BLA_agonist_viability', 'TOX21_AR_BLA_Agonist_ch1',
                'TOX21_AR_BLA_Agonist_ch2', 'TOX21_AR_BLA_Agonist_ratio', 'TOX21_AR_BLA_Antagonist_ch1',
                'TOX21_AR_BLA_Antagonist_ch2', 'TOX21_AR_BLA_Antagonist_ratio', 'TOX21_AR_BLA_Antagonist_viability',
                'TOX21_AR_LUC_MDAKB2_Agonist', 'TOX21_AR_LUC_MDAKB2_Antagonist', 'TOX21_AR_LUC_MDAKB2_Antagonist2',
                'TOX21_AhR_LUC_Agonist', 'TOX21_Aromatase_Inhibition', 'TOX21_AutoFluor_HEK293_Cell_blue',
                'TOX21_AutoFluor_HEK293_Media_blue', 'TOX21_AutoFluor_HEPG2_Cell_blue',
                'TOX21_AutoFluor_HEPG2_Cell_green', 'TOX21_AutoFluor_HEPG2_Media_blue',
                'TOX21_AutoFluor_HEPG2_Media_green', 'TOX21_ELG1_LUC_Agonist', 'TOX21_ERa_BLA_Agonist_ch1',
                'TOX21_ERa_BLA_Agonist_ch2', 'TOX21_ERa_BLA_Agonist_ratio', 'TOX21_ERa_BLA_Antagonist_ch1',
                'TOX21_ERa_BLA_Antagonist_ch2', 'TOX21_ERa_BLA_Antagonist_ratio', 'TOX21_ERa_BLA_Antagonist_viability',
                'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_ERa_LUC_BG1_Antagonist', 'TOX21_ESRE_BLA_ch1', 'TOX21_ESRE_BLA_ch2',
                'TOX21_ESRE_BLA_ratio', 'TOX21_ESRE_BLA_viability', 'TOX21_FXR_BLA_Antagonist_ch1',
                'TOX21_FXR_BLA_Antagonist_ch2', 'TOX21_FXR_BLA_agonist_ch2', 'TOX21_FXR_BLA_agonist_ratio',
                'TOX21_FXR_BLA_antagonist_ratio', 'TOX21_FXR_BLA_antagonist_viability', 'TOX21_GR_BLA_Agonist_ch1',
                'TOX21_GR_BLA_Agonist_ch2', 'TOX21_GR_BLA_Agonist_ratio', 'TOX21_GR_BLA_Antagonist_ch2',
                'TOX21_GR_BLA_Antagonist_ratio', 'TOX21_GR_BLA_Antagonist_viability', 'TOX21_HSE_BLA_agonist_ch1',
                'TOX21_HSE_BLA_agonist_ch2', 'TOX21_HSE_BLA_agonist_ratio', 'TOX21_HSE_BLA_agonist_viability',
                'TOX21_MMP_ratio_down', 'TOX21_MMP_ratio_up', 'TOX21_MMP_viability', 'TOX21_NFkB_BLA_agonist_ch1',
                'TOX21_NFkB_BLA_agonist_ch2', 'TOX21_NFkB_BLA_agonist_ratio', 'TOX21_NFkB_BLA_agonist_viability',
                'TOX21_PPARd_BLA_Agonist_viability', 'TOX21_PPARd_BLA_Antagonist_ch1', 'TOX21_PPARd_BLA_agonist_ch1',
                'TOX21_PPARd_BLA_agonist_ch2', 'TOX21_PPARd_BLA_agonist_ratio', 'TOX21_PPARd_BLA_antagonist_ratio',
                'TOX21_PPARd_BLA_antagonist_viability', 'TOX21_PPARg_BLA_Agonist_ch1', 'TOX21_PPARg_BLA_Agonist_ch2',
                'TOX21_PPARg_BLA_Agonist_ratio', 'TOX21_PPARg_BLA_Antagonist_ch1', 'TOX21_PPARg_BLA_antagonist_ratio',
                'TOX21_PPARg_BLA_antagonist_viability', 'TOX21_TR_LUC_GH3_Agonist', 'TOX21_TR_LUC_GH3_Antagonist',
                'TOX21_VDR_BLA_Agonist_viability', 'TOX21_VDR_BLA_Antagonist_ch1', 'TOX21_VDR_BLA_agonist_ch2',
                'TOX21_VDR_BLA_agonist_ratio', 'TOX21_VDR_BLA_antagonist_ratio', 'TOX21_VDR_BLA_antagonist_viability',
                'TOX21_p53_BLA_p1_ch1', 'TOX21_p53_BLA_p1_ch2', 'TOX21_p53_BLA_p1_ratio', 'TOX21_p53_BLA_p1_viability',
                'TOX21_p53_BLA_p2_ch1', 'TOX21_p53_BLA_p2_ch2', 'TOX21_p53_BLA_p2_ratio', 'TOX21_p53_BLA_p2_viability',
                'TOX21_p53_BLA_p3_ch1', 'TOX21_p53_BLA_p3_ch2', 'TOX21_p53_BLA_p3_ratio', 'TOX21_p53_BLA_p3_viability',
                'TOX21_p53_BLA_p4_ch1', 'TOX21_p53_BLA_p4_ch2', 'TOX21_p53_BLA_p4_ratio', 'TOX21_p53_BLA_p4_viability',
                'TOX21_p53_BLA_p5_ch1', 'TOX21_p53_BLA_p5_ch2', 'TOX21_p53_BLA_p5_ratio', 'TOX21_p53_BLA_p5_viability',
                'Tanguay_ZF_120hpf_AXIS_up', 'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_BRAI_up',
                'Tanguay_ZF_120hpf_CFIN_up', 'Tanguay_ZF_120hpf_CIRC_up', 'Tanguay_ZF_120hpf_EYE_up',
                'Tanguay_ZF_120hpf_JAW_up', 'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_OTIC_up',
                'Tanguay_ZF_120hpf_PE_up', 'Tanguay_ZF_120hpf_PFIN_up', 'Tanguay_ZF_120hpf_PIG_up',
                'Tanguay_ZF_120hpf_SNOU_up', 'Tanguay_ZF_120hpf_SOMI_up', 'Tanguay_ZF_120hpf_SWIM_up',
                'Tanguay_ZF_120hpf_TRUN_up', 'Tanguay_ZF_120hpf_TR_up', 'Tanguay_ZF_120hpf_YSE_up'],
    'ESOL': ['measured log solubility in mols per litre'],
    'FreeSolv': ['expt'],
    'LIPO': ['exp'],
    'QM7': ['u0_atom'],
    'QM8': ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM',
            'f1-CAM', 'f2-CAM'],
    'QM9': ['homo', 'lumo', 'gap'],
    'HIV': ['HIV_active'],
    'MUV': ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689', 'MUV-692', 'MUV-712', 'MUV-713',
            'MUV-733', 'MUV-737', 'MUV-810', 'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
}

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


def smiles_2_kgdgl(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if smiles.strip() == "" or mol is None:
        return None
    connected_atom_list = []
    for bond in mol.GetBonds():
        connected_atom_list.append(bond.GetBeginAtomIdx())
        connected_atom_list.append(bond.GetEndAtomIdx())

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
            rel_features.extend(
                i + 4 for i in rid)

    if begin_attributes:
        attribute_id = sorted(list(set(begin_attributes)))
        node_id = [i + len(connected_atom_list) for i in range(len(attribute_id))]
        attrid2nodeid = dict(zip(attribute_id, node_id))
        nodeids = [attrid2nodeid[i] for i in begin_attributes]

        nodes_feature = [i + 118 for i in
                         attribute_id]

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
    return graph


def collate_molgraphs(data):
    smiles, graphs = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    return smiles, bg


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


kg_data = Triples("./algorithm/drug_property_kg/resource/triples")


class MoleculeDataset(Dataset):
    def __init__(self, smiles, graphs):
        self.smiles = smiles
        self.graphs = graphs

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item]

    def __len__(self):
        return len(self.smiles)


def genarate_graph(smiles, n_jobs=1):
    if n_jobs > 1:
        graphs = pmap(smiles_2_kgdgl, smiles, n_jobs=n_jobs)
    else:
        graphs = []
        for i, s in enumerate(smiles):
            graphs.append(smiles_2_kgdgl(s))
    valid_ids = []
    invalid_smiles = []
    graphs_valid = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            graphs_valid.append(g)
        else:
            invalid_smiles.append(smiles[i])
    smiles_valid = [smiles[i] for i in valid_ids]

    return invalid_smiles, smiles_valid, graphs_valid


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
        tasks = set(tasks)
        tasks = tasks & set(task_properties.keys())
        smiles = list(set(smis))
        invalid_smiles, smiles_valid, graphs_valid = genarate_graph(smiles)
        dataset = MoleculeDataset(smiles_valid, graphs_valid)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_molgraphs, num_workers=8)
        tasks_results = {}
        for task_name in tqdm(tasks):
            tasks_results[task_name] = self.infer_one_task(task_name, device, dataloader)
        return self.format(smiles_valid, tasks_results, invalid_smiles)

    def format(self, smiles_valid, tasks_results, invalid_smiles):
        arr = []
        for i in range(len(smiles_valid)):
            arr.append({
                "smile": smiles_valid[i],
                "valid": True,
                "tasks": [{"task_name": task_name,
                           "property_names": task_properties[task_name],
                           "result": {
                               task_properties[task_name][property_id]: tasks_results[task_name][i][property_id] for
                               property_id in range(len(task_properties[task_name]))
                           }
                           } for task_name in tasks_results
                          ]
            })
        for i in range(len(invalid_smiles)):
            arr.append({
                "smile": invalid_smiles[i],
                "valid": False
            })
        # print(json.dumps(arr))
        """
        drug_smiles, dataset-property
        """

        head = ["drug_smiles"]

        head_property_name_list = []
        for task_name in tasks_results:
            property_list = task_properties[task_name]
            for property_id in property_list:
                head_property_name_list.append(f"{task_name}-{property_id}")
        property_idx_dict = {}
        for property_idx, head_property_name in enumerate(head_property_name_list):
            task_name = head_property_name.split('-')[0]
            property_name = head_property_name[len(task_name) + 1:]
            property_idx_dict[property_name] = property_idx + 1

        head.extend(head_property_name_list)
        row_num = len(smiles_valid) + len(invalid_smiles) + 1
        col_num = len(head)
        results = np.full((row_num, col_num), np.nan).tolist()
        results[0] = head
        for row, result_item_dict in enumerate(arr):
            results[row + 1][0] = result_item_dict["smile"]
            if not result_item_dict["valid"]:
                continue
            for task in result_item_dict["tasks"]:
                for key, value in task["result"].items():
                    col = property_idx_dict[key]
                    results[row + 1][col] = value

        # print(results)
        # print(arr)
        return results

    def infer_one_task(self, task_name, device, dataloader):

        encoder = KGMPNN(self.args, self.entity_emb, self.relation_emb).to(device)
        encoder.load_state_dict(
            torch.load(f"./algorithm/drug_property_kg/pretrained/{task_name}/KGMPNN.model", map_location=device))

        readout = Set2Set(encoder.out_dim, n_iters=6, n_layers=3).to(device)
        readout.load_state_dict(
            torch.load(f"./algorithm/drug_property_kg/pretrained/{task_name}/Set2Set.model", map_location=device))

        predictor = Predictor(readout.out_dim, len(task_properties[task_name]), self.args).to(device)
        predictor.load_state_dict(
            torch.load(f"./algorithm/drug_property_kg/pretrained/{task_name}/Predictor.model", map_location=device))

        encoder.eval()
        predictor.eval()
        result = []
        with torch.no_grad():
            for batch_id, batch_data in enumerate(dataloader):
                smiles, bg = batch_data
                bg = bg.to(device)
                graph_embedding = readout(bg, encoder(bg))
                logits = predictor(graph_embedding)
                if task_name in cls_tasks:
                    logits = torch.sigmoid(logits.clone().detach())
                result.extend(logits.tolist())

        return result


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=2)
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

    # predictor
    parser.add_argument('--predictor_dropout', type=float, default=0.0)
    parser.add_argument('--predictor_hidden_feats', type=int, default=256)

    parser.add_argument('--initial_path', type=str,
                        default='./algorithm/drug_property_kg/pretrained/RotatE_128_64_emb.pkl')
    return parser.parse_known_args()[0].__dict__


def drug_property_prediction(drug_smiles, task_list, batch_size, device):
    args = get_args()
    i = Inference(args=args)
    return i.infer_all(drug_smiles, task_list, batch_size=batch_size, device=device)


if __name__ == '__main__':
    result = drug_property_prediction(["CCCN1CCC[C@@H]2Cc3n[nH]cc3C[C@H]21", "Test"] * 10, ["Tox21", "LIPO"],
                                      batch_size=8,
                                      device='cuda:0')

    print(result)
