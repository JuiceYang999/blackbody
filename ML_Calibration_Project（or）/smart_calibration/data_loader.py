import pandas as pd
from scipy.signal import savitzky_golay

def load_and_preprocess_data(file_path, window_length=5, polyorder=2):
    """
    [cite_start]加载原始数据并应用 Savitzky-Golay 滤波器进行去噪 [cite: 597]。
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {file_path}")
        return None
    
    # [cite_start]3.3.2 信号去噪：Savitzky-Golay 滤波器 [cite: 597]
    df['V1_filt'] = savitzky_golay(df['V1'], window_length, polyorder)
    df['V2_filt'] = savitzky_golay(df['V2'], window_length, polyorder)
    
    print("数据加载和去噪完成。")
    return df