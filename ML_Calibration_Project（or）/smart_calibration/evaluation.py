import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_ablation_models(models, X_test, y_test, feature_sets):
    """
    [cite_start]在测试集上评估所有模型，使用 RMSE, MAE, R² [cite: 616-619]。
    """
    results = {}
    for name, model in models.items():
        if name in ['M0', 'M1']:
            feature_set = feature_sets['M0_M1']
        elif name == 'M2':
            feature_set = feature_sets['M2']
        else: # M3
            feature_set = feature_sets['M3']
        
        y_pred = model.predict(X_test[feature_set])
        
        results[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
    return pd.DataFrame(results).T

def plot_predictions_vs_actual(y_test, models, X_test, feature_sets, save_path):
    """
    [cite_start]生成并保存 "图2: 预测-真实对比图" [cite: 673]。
    """
    y_pred_m0 = models['M0'].predict(X_test[feature_sets['M0_M1']])
    y_pred_m3 = models['M3'].predict(X_test[feature_sets['M3']])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1.scatter(y_test, y_pred_m0, alpha=0.6, s=10)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_title(f"M0 (基线)")
    ax1.set_xlabel("真实温度 T_true")
    ax1.set_ylabel("预测温度 T_pred")

    ax2.scatter(y_test, y_pred_m3, alpha=0.6, s=10)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_title(f"M3 (完整模型)")
    ax2.set_xlabel("真实温度 T_true")
    ax2.set_ylabel("预测温度 T_pred")
    
    plt.suptitle("图2: 预测-真实对比图")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"图2 已保存至: {save_path}")

def plot_residuals(y_test, models, X_test, feature_sets, save_path):
    """
    [cite_start]生成并保存 "图3: 残差分析图" [cite: 717]。
    """
    y_pred_m0 = models['M0'].predict(X_test[feature_sets['M0_M1']])
    y_pred_m3 = models['M3'].predict(X_test[feature_sets['M3']])
    residuals_m0 = y_pred_m0 - y_test
    residuals_m3 = y_pred_m3 - y_test

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    ax1.scatter(y_test, residuals_m0, alpha=0.6, s=10)
    ax1.axhline(0, color='r', linestyle='--', lw=2)
    ax1.set_title("M0 (基线) 残差图")
    ax1.set_xlabel("真实温度 T_true")
    ax1.set_ylabel("残差 (T_pred - T_true)")

    ax2.scatter(y_test, residuals_m3, alpha=0.6, s=10)
    ax2.axhline(0, color='r', linestyle='--', lw=2)
    ax2.set_title("M3 (完整模型) 残差图")
    ax2.set_xlabel("真实温度 T_true")
    
    plt.suptitle("图3: 残差分析图")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"图3 已保存至: {save_path}")

def plot_data_efficiency(results, save_path):
    """
    [cite_start]生成并保存 "图1: 数据效率对比图" [cite: 643]。
    """
    train_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.figure(figsize=(10, 6))
    plt.plot(train_fractions, results['M0'], 'bo-', label='M0 (多项式)')
    plt.plot(train_fractions, results['M3'], 'rs-', label='M3 (XGBoost 完整特征)')
    plt.xlabel("用于训练的数据百分比")
    plt.ylabel("测试集 RMSE")
    plt.title("图1: 数据效率对比图")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(save_path)
    plt.close()
    print(f"图1 已保存至: {save_path}")

def evaluate_and_plot_ensemble(ensemble_models, X_test, y_test, save_path):
    """
    [cite_start]评估集成模型并生成 "图4: 带不确定性量化的预测图" [cite: 731, 734]。
    """
    predictions_ensemble = np.array([model.predict(X_test) for model in ensemble_models])
    
    y_pred_ensemble_mean = np.mean(predictions_ensemble, axis=0)
    y_pred_ensemble_std = np.std(predictions_ensemble, axis=0)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(y_test, y_pred_ensemble_mean, c=y_pred_ensemble_std, cmap='viridis', alpha=0.7, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.colorbar(sc, label='预测标准差 (不确定性)')
    plt.title("图4: 带不确定性量化的预测图")
    plt.xlabel("真实温度 T_true")
    plt.ylabel("集成预测温度 T_pred")
    plt.savefig(save_path)
    plt.close()
    print(f"图4 已保存至: {save_path}")