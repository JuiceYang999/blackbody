import pandas as pd
import numpy as np

def build_features(df):
    """
    构建多维度特征集。
    [cite_start]这个理念借鉴了 Jones et al. 的工作，将标量度量升级为多维向量 [cite: 600, 789]。
    """
    features = pd.DataFrame(index=df.index)

    # [cite_start]核心物理量：比色法 [cite: 603]
    features['Ratio'] = df['V1_filt'] / df['V2_filt']
    
    # [cite_start]线性化转换：与 1/T 呈近似线性关系 [cite: 605]
    features['Log_Ratio'] = np.log(features['Ratio'])
    
    # [cite_start]能量信息补充：反映总辐射能量 [cite: 605]
    features['Energy_Sum'] = df['V1_filt'] + df['V2_filt']
    
    # [cite_start]数值稳定性增强 [cite: 605]
    features['Norm_Diff'] = (df['V1_filt'] - df['V2_filt']) / (df['V1_filt'] + df['V2_filt'])
    
    # [cite_start]交互特征 [cite: 605]
    features['Interaction'] = df['V1_filt'] * df['V2_filt']
    
    # 原始特征（用于 M2 模型）
    features['V1'] = df['V1_filt']
    features['V2'] = df['V2_filt']
    
    # 定义目标变量
    target = df['T_true']
    
    print("特征工程完成。")
    return features, target

def get_feature_sets():
    """
    [cite_start]定义消融实验所需的特征集 [cite: 621-638]。
    """
    feature_sets = {
        [cite_start]'M0_M1': ['Ratio'], # M0 和 M1 使用 [cite: 624, 628]
        [cite_start]'M2': ['V1', 'V2'], # M2 使用 [cite: 633]
        [cite_start]'M3': ['Ratio', 'Log_Ratio', 'Energy_Sum', 'Norm_Diff', 'Interaction'] # M3 使用 [cite: 637]
    }
    return feature_sets