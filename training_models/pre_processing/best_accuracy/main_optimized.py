import os
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

# Suprimir warnings de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# abrindo os metadados
df = pd.read_csv("./../../../meta_dados/input_data/experiment_0/best_accuracy_sample_0.csv")
X = df.drop(columns=["target"]).to_numpy()
y = df["target"].to_numpy()

# criando KFold
kf = KFold(n_splits=60, shuffle=True, random_state=42) 

# Grids de parâmetros otimizados para reduzir tempo
ada_boost_param_grid = {
    'regressor__n_estimators': [30, 50, 70],
    'regressor__learning_rate': [0.1, 0.5, 1.0],
    'regressor__loss': ['linear', 'square'],
    'regressor__estimator__max_depth': [1, 2, 3],
    'regressor__estimator__min_samples_split': [2, 5],
    'regressor__estimator__min_samples_leaf': [1, 2],
    'regressor__estimator__criterion': ['squared_error']
}

knn_param_grid = {
    'regressor__n_neighbors': [3, 5, 10],  
    'regressor__weights': ['uniform', 'distance'],
    'regressor__algorithm': ['auto'],  
    'regressor__p': [1, 2]
}

lr_param_grid = {
    'regressor__positive': [True, False]
}

mlp_param_grid = {
    'regressor__activation': ['relu', 'tanh'],  
    'regressor__solver': ['adam'],
    'regressor__learning_rate': ['constant'],
    'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'regressor__alpha': [0.0001, 0.001],
    'regressor__max_iter': [100, 200],  
    'regressor__early_stopping': [True]
}

svr_param_grid = {
    'regressor__kernel': ['linear', 'rbf'],
    'regressor__gamma': ['scale', 'auto'],
    'regressor__C': [0.1, 1.0, 10.0],
    'regressor__epsilon': [0.1, 0.5]
}

# salvar resultados
def save_results(random_search, pipeline_name, filename="./resultados_individual_features_optimized.csv"):
    model_name = random_search.best_estimator_.named_steps['regressor'].__class__.__name__
    metric_score = random_search.best_score_
    new_data = pd.DataFrame({
        'model': [model_name],
        'metric_score': [metric_score],
        'pipeline': [pipeline_name]
    })
    if not os.path.exists(filename):
        new_data.to_csv(filename, index=False)
    else:
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(filename, index=False)

models_config = {
    'AdaBoostRegressor': {
        'model': AdaBoostRegressor(random_state=42, estimator=DecisionTreeRegressor(random_state=42)),
        'params': ada_boost_param_grid
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(),
        'params': knn_param_grid
    },
    'LinearRegression': {
        'model': LinearRegression(),
        'params': lr_param_grid
    },
    'MLPRegressor': {
        'model': MLPRegressor(random_state=42),
        'params': mlp_param_grid
    },
    'SVR': {
        'model': SVR(),
        'params': svr_param_grid
    }
}

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler()
}

selectors = {
    'VarianceThreshold': VarianceThreshold(threshold=0.01),
    'RFE': RFE(estimator=LinearRegression()),
    'SelectFromModel': SelectFromModel(estimator=DecisionTreeRegressor(random_state=42))
}

print(f"Dataset possui {X.shape[1]} features")
print("Iniciando treinamento com pipelines de normalização e seleção de features...")

for scaler_name, scaler in scalers.items():
    for selector_name, selector in selectors.items():
        for model_name, config in models_config.items():
            pipeline = Pipeline([
                ('scaler', scaler),
                ('selector', selector),
                ('regressor', config['model'])
            ])
            param_grid = config['params']
            print(f"Treinando {model_name} | {scaler_name} | {selector_name}...", end=" ")
            try:
                random_search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    cv=kf,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0,
                    random_state=42,
                    n_iter=40
                )
                random_search.fit(X, y)
                pipeline_name = f"{model_name}_{scaler_name}_{selector_name}"
                save_results(random_search, pipeline_name)
                model_filename = f"./saved_models/{pipeline_name}.joblib"
                os.makedirs("./saved_models", exist_ok=True)
                joblib.dump(random_search.best_estimator_, model_filename)
                print(f"MSE: {-random_search.best_score_:.6f}")
            except Exception as e:
                print(f"Erro: {str(e)}")
                continue

print("\nTreinamento concluído! Resultados salvos em 'resultados_individual_features_optimized.csv'")

if os.path.exists("./resultados_individual_features_optimized.csv"):
    results_df = pd.read_csv("./resultados_individual_features_optimized.csv")
    print("\nResumo dos resultados:")
    print(results_df.groupby('model')['metric_score'].agg(['mean', 'std', 'min', 'max']).round(6))
    print("\nMelhor pipeline por modelo:")
    best_by_model = results_df.loc[results_df.groupby('model')['metric_score'].idxmax()]
    print(best_by_model[['pipeline', 'model', 'metric_score']].round(6))
    print("\nTop 10 pipelines com melhor performance:")
    top_pipelines = best_by_model.nsmallest(10, 'metric_score')
    print(top_pipelines[['pipeline', 'model', 'metric_score']].round(6))
