import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def split_and_scale_data(features, target, test_size=0.2, random_state=42):
    """
    [cite_start]划分训练集/测试集，并应用 Z-score 标准化 [cite: 598]。
    """
    all_feature_names = list(features.columns)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features[all_feature_names], target, test_size=test_size, random_state=random_state
    )
    
    scalers = {}
    X_train_scaled = X_train_raw.copy()
    X_test_scaled = X_test_raw.copy()

    for col in all_feature_names:
        scaler = StandardScaler()
        X_train_scaled[col] = scaler.fit_transform(X_train_raw[[col]])
        X_test_scaled[col] = scaler.transform(X_test_raw[[col]])
        scalers[col] = scaler
        
    print(f"数据划分为: {X_train_scaled.shape[0]} 训练样本, {X_test_scaled.shape[0]} 测试样本。")
    return X_train_scaled, X_test_scaled, y_train, y_test, scalers

def train_ablation_models(X_train, y_train, feature_sets, poly_degree=3):
    """
    [cite_start]训练消融实验中的所有模型 (M0, M1, M2, M3) [cite: 621-638]。
    """
    models = {}

    # [cite_start]M0: 基线 - 多项式回归 [Ratio] [cite: 608, 622]
    models['M0'] = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    models['M0'].fit(X_train[feature_sets['M0_M1']], y_train)

    # [cite_start]M1: 基础ML - XGBoost [Ratio] [cite: 626]
    models['M1'] = xgb.XGBRegressor(random_state=42)
    models['M1'].fit(X_train[feature_sets['M0_M1']], y_train)

    # [cite_start]M2: 原始信息 - XGBoost [V1, V2] [cite: 631]
    models['M2'] = xgb.XGBRegressor(random_state=42)
    models['M2'].fit(X_train[feature_sets['M2']], y_train)

    # [cite_start]M3: 完整特征模型 - XGBoost [Full Features] [cite: 635]
    # [cite_start]使用 K-折交叉验证和网格搜索进行超参数调优 [cite: 615]
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
    }
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(random_state=42),
        param_grid=param_grid,
        cv=kfold,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train[feature_sets['M3']], y_train)
    models['M3'] = grid_search.best_estimator_
    
    print("消融实验模型训练完成。")
    print(f"M3 最佳参数: {grid_search.best_params_}")
    return models

def perform_data_efficiency_study(X_train, y_train, X_test, y_test, feature_sets, poly_degree=3, best_params={}):
    """
    [cite_start]执行数据效率分析，借鉴 Jones et al. (2022) [cite: 641, 7]。
    """
    train_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rmse_results = {'M0': [], 'M3': []}

    for frac in train_fractions:
        n_samples = int(len(X_train) * frac)
        if n_samples < 1: n_samples = 1
        
        X_train_subset = X_train.sample(n=n_samples, random_state=42)
        y_train_subset = y_train.loc[X_train_subset.index]
        
        # 训练 M0
        model_m0 = Pipeline([
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
        model_m0.fit(X_train_subset[feature_sets['M0_M1']], y_train_subset)
        y_pred_m0 = model_m0.predict(X_test[feature_sets['M0_M1']])
        rmse_results['M0'].append(np.sqrt(mean_squared_error(y_test, y_pred_m0)))
        
        # 训练 M3
        model_m3 = xgb.XGBRegressor(**best_params, random_state=42)
        model_m3.fit(X_train_subset[feature_sets['M3']], y_train_subset)
        y_pred_m3 = model_m3.predict(X_test[feature_sets['M3']])
        rmse_results['M3'].append(np.sqrt(mean_squared_error(y_test, y_pred_m3)))
        
    print("数据效率分析完成。")
    return rmse_results

def train_ensemble(X_train, y_train, best_params, n_splits=10):
    """
    [cite_start]使用 K-折模型构建集成，用于不确定性量化 [cite: 728, 729]。
    """
    ensemble_models = []
    kfold_ens = KFold(n_splits=n_splits, shuffle=True, random_state=1337)

    for fold, (train_idx, val_idx) in enumerate(kfold_ens.split(X_train, y_train)):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        
        model = xgb.XGBRegressor(**best_params, random_state=fold)
        model.fit(X_train_fold, y_train_fold)
        ensemble_models.append(model)
        
    print(f"创建了 {len(ensemble_models)} 个模型的集成。")
    return ensemble_models