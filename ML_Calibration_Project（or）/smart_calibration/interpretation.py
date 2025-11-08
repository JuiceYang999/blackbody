import shap
import matplotlib.pyplot as plt

def run_shap_analysis(model, X_test, feature_names, save_path_summary, save_path_dependence):
    """
    [cite_start]运行 SHAP 分析并保存 "图5" 和 "图6" [cite: 762]。
    """
    print("正在运行 SHAP 分析...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # [cite_start]"图5: SHAP 特征重要性摘要图" [cite: 763]
    plt.figure()
    plt.title("图5: SHAP 特征重要性摘要图")
    shap.summary_plot(shap_values, X_test, plot_type="beeswarm", show=False)
    plt.savefig(save_path_summary, bbox_inches='tight')
    plt.close()
    print(f"图5 SHAP 摘要图已保存至: {save_path_summary}")

    # [cite_start]"图6: SHAP 依赖图" [cite: 769]
    # 验证 Log_Ratio 与 Energy_Sum 的交互
    plt.figure()
    shap.dependence_plot(
        "Log_Ratio",
        shap_values.values,
        X_test,
        feature_names=feature_names,
        interaction_index="Energy_Sum",
        show=False
    )
    plt.savefig(save_path_dependence, bbox_inches='tight')
    plt.close()
    print(f"图6 SHAP 依赖图已保存至: {save_path_dependence}")