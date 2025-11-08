import os
import smart_calibration.data_loader as dl
import smart_calibration.feature_engineering as fe
import smart_calibration.modeling as modeling
import smart_calibration.evaluation as ev
import smart_calibration.interpretation as interp

# --- 1. 配置 ---
DATA_FILE_PATH = 'data/data.csv'
RESULTS_DIR = 'results'

# 创建结果目录
os.makedirs(RESULTS_DIR, exist_ok=True)

def main_pipeline():
    """
    执行完整的智能标定流程。
    """
    print("--- 流程开始：智能温度标定 ---")

    # --- 2. 加载和预处理数据 ---
    df = dl.load_and_preprocess_data(DATA_FILE_PATH)
    if df is None:
        return

    # --- 3. 特征工程 ---
    features, target = fe.build_features(df)
    feature_sets = fe.get_feature_sets()
    all_features = list(features.columns)

    # --- 4. 数据划分与标准化 ---
    X_train, X_test, y_train, y_test, _ = modeling.split_and_scale_data(
        features, target
    )

    # --- 5. 消融实验模型训练 ---
    models = modeling.train_ablation_models(
        X_train[all_features], y_train, feature_sets
    )
    m3_best_params = models['M3'].get_params()

    # --- 6. 模型评估与可视化 ---
    results_df = ev.evaluate_ablation_models(models, X_test, y_test, feature_sets)
    print("\n--- 消融实验评估结果 (测试集) ---")
    print(results_df)

    ev.plot_predictions_vs_actual(y_test, models, X_test, feature_sets, 
                                  save_path=f"{RESULTS_DIR}/Fig2_Pred_vs_Actual.png")
    
    ev.plot_residuals(y_test, models, X_test, feature_sets, 
                      save_path=f"{RESULTS_DIR}/Fig3_Residuals.png")

    # --- 7. 数据效率分析 ---
    efficiency_results = modeling.perform_data_efficiency_study(
        X_train, y_train, X_test, y_test, feature_sets, 
        best_params=m3_best_params
    )
    ev.plot_data_efficiency(efficiency_results, 
                            save_path=f"{RESULTS_DIR}/Fig1_Data_Efficiency.png")

    # --- 8. 不确定性量化 ---
    ensemble_models = modeling.train_ensemble(
        X_train[feature_sets['M3']], y_train, m3_best_params
    )
    ev.evaluate_and_plot_ensemble(ensemble_models, X_test[feature_sets['M3']], y_test,
                                  save_path=f"{RESULTS_DIR}/Fig4_Uncertainty_Plot.png")

    # --- 9. 模型解释性 ---
    interp.run_shap_analysis(
        models['M3'], X_test[feature_sets['M3']], 
        feature_names=feature_sets['M3'],
        save_path_summary=f"{RESULTS_DIR}/Fig5_SHAP_Summary.png",
        save_path_dependence=f"{RESULTS_DIR}/Fig6_SHAP_Dependence.png"
    )

    print("\n--- 流程结束：所有分析和绘图已完成 ---")

if __name__ == "__main__":
    main_pipeline()