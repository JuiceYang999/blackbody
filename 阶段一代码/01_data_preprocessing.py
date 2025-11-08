import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """加载原始数据文件，假设为CSV格式，包含 T_true, V1, V2 列"""
    return pd.read_csv(filepath)

def denoise_signals(df, window_length=11, polyorder=2):
    """对V1和V2信号应用Savitzky-Golay滤波器"""
    df['V1_denoised'] = savgol_filter(df['V1'], window_length, polyorder)
    df['V2_denoised'] = savgol_filter(df['V2'], window_length, polyorder)
    return df

def feature_engineering(df):
    """根据3.4节定义，构建多维度特征"""
    # 核心物理特征
    df['Ratio'] = df['V1_denoised'] / df['V2_denoised']
    
    # 深化物理特征
    df['Log_Ratio'] = np.log(df['Ratio'])
    df['Energy_Sum'] = df['V1_denoised'] + df['V2_denoised']
    df['Norm_Diff'] = (df['V1_denoised'] - df['V2_denoised']) / (df['V1_denoised'] + df['V2_denoised'])
    
    # 最终用于模型的列
    features = ['V1_denoised', 'V2_denoised', 'Ratio', 'Log_Ratio', 'Energy_Sum', 'Norm_Diff']
    target = 'T_true'
    
    return df[features], df[target]

def preprocess_pipeline(filepath):
    """整合完整的数据预处理流程"""
    # 1. 加载数据
    raw_df = load_data(filepath)
    
    # 2. 去噪
    denoised_df = denoise_signals(raw_df)
    
    # 3. 特征工程
    X, y = feature_engineering(denoised_df)
    
    # 4. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 将 scaled arrays 转换回 DataFrame 以便后续使用
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 主程序入口
if __name__ == '__main__':
    # 假设你的数据文件叫 'calibration_data.csv'
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline('calibration_data.csv')
    
    # 打印处理后的数据形状以作检查
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    
    # 可以将处理好的数据保存下来，方便后续模型训练脚本直接调用
    # X_train.to_csv('X_train.csv', index=False)