from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os

# Suprimir warnings de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# abrindo os metadados
df = pd.read_csv("./../../../meta_dados/input_data/experiment_4/best_accuracy_sample_4.csv")
X = df.drop(columns=["target"]).to_numpy()
y = df["target"].to_numpy()

# criando kfold (ajustar número de splits para dataset pequeno)
kf = KFold(n_splits=60, shuffle=True, random_state=42)

# Função para salvar resultados
def save_results(random_search, filename="./resultados_pipeline_optimized.csv"):
    model_name = random_search.best_estimator_.named_steps['regressor'].__class__.__name__
    metric_score = random_search.best_score_
    new_data = pd.DataFrame({
        'model': [model_name],
        'metric_score': [metric_score]
    })
    if not os.path.exists(filename):
        new_data.to_csv(filename, index=False)
    else:
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(filename, index=False)




param_grid_ada_boost_regressor = {
    'n_estimators': [30, 50, 70, 100],
    'learning_rate': [0.1, 0.3 ,0.5, 0.7, 1.0],
    'loss': ['linear', 'square'],  # compatível com regressão
    'estimator__max_depth': [1, 2, 3, None],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__criterion': ['squared_error', 'friedman_mse']
}

param_grid_decision_tree_regressor = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
    'min_samples_leaf': [1, 2, 4, 5, 10, 15, 20, 30, 35, 40],
    'criterion': ['squared_error', 'friedman_mse']
}

param_grid_knn_regressor = {
    'n_neighbors': [3, 5, 10, 20],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    'leaf_size': [15, 30, 60],
    'p': [1, 2],
    'metric': ['euclidean', 'manhattan']
}

param_grid_linear_regressor = {
    'positive': [True, False]
}

param_grid_mlp_regressor = [
    {
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'learning_rate': ['constant', 'adaptive'],
        'hidden_layer_sizes': [
            (50,), (100,), (150,),
            (50, 50), (100, 100), (150, 150),
            (50, 50, 50), (100, 100, 100)
        ],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [100, 200, 300],
        'early_stopping': [True]
    },
    {
        'activation': ['tanh', 'relu'],
        'solver': ['sgd'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'hidden_layer_sizes': [
            (50,), (100,), (150,),
            (50, 50), (100, 100), (150, 150),
            (50, 50, 50), (100, 100, 100)
        ],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [100, 200, 300],
        'nesterovs_momentum': [True],
        'early_stopping': [True]
    }
]

param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 0.5, 1.0],
    'tol': [1e-4, 1e-3, 1e-2],
    'C': [0.1, 1.0, 10.0],
    'epsilon': [0.1, 0.5]
}

param_grid_random_forest_regressor = {
    'n_estimators': [50, 100, 200],
    'criterion': ['squared_error', 'friedman_mse', 'poisson', 'absolute_error'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

models_config = {
    'AdaBoostRegressor': {
        'regressor': AdaBoostRegressor(random_state=42, estimator=DecisionTreeRegressor(random_state=42)),
        'params': {f'regressor__{k}': v for k, v in param_grid_ada_boost_regressor.items()}
    },
    'DecisionTreeRegressor': {
        'regressor': DecisionTreeRegressor(random_state=42),
        'params': {f'regressor__{k}': v for k, v in param_grid_decision_tree_regressor.items()}
    },
    'KNeighborsRegressor': {
        'regressor': KNeighborsRegressor(),
        'params': {f'regressor__{k}': v for k, v in param_grid_knn_regressor.items()}
    },
    'LinearRegression': {
        'regressor': LinearRegression(),
        'params': {f'regressor__{k}': v for k, v in param_grid_linear_regressor.items()}
    },
    'MLPRegressor': {
        'regressor': MLPRegressor(random_state=42),
        'params': [{f'regressor__{k}': v for k, v in grid.items()} for grid in param_grid_mlp_regressor]
    },
    'SVR': {
        'regressor': SVR(),
        'params': {f'regressor__{k}': v for k, v in param_grid_svr.items()}
    },
    'RandomForestRegressor': {
        'regressor': RandomForestRegressor(random_state=42),
        'params': {f'regressor__{k}': v for k, v in param_grid_random_forest_regressor.items()}
    }
}

# -------------------------
# TREINAMENTO
# -------------------------

scalers = [StandardScaler(), MinMaxScaler()]

results = []
print("Iniciando treinamento com todas as features, pipelines e diferentes métodos de scaling...")

for scaler in scalers:
    for model_name, config in models_config.items():
        print(f"\nTreinando {model_name} com {scaler.__class__.__name__}...", end=" ")
        try:
            pipe = Pipeline([
                ('scaler', scaler),
                ('regressor', config['regressor'])
            ])
            param_grids = config['params'] if isinstance(config['params'], list) else [config['params']]
            for param_grid in param_grids:
                grid = param_grid.copy()
                grid['scaler'] = [scaler]

                random_search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=grid,
                    cv=kf,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0,
                    random_state=42,
                    n_iter=50
                )
                random_search.fit(X, y)
                best_score = random_search.best_score_
                best_model = random_search.best_estimator_
                results.append({
                    'model': model_name,
                    'metric_score': best_score,
                    'scaler': scaler.__class__.__name__,
                    'best_model': best_model
                })
                print(f"MSE: {-best_score:.6f}")
        except Exception as e:
            print(f"Erro: {str(e)}")
            continue

# Salvar apenas o melhor resultado global para cada modelo
results_df = pd.DataFrame(results)
best_results = results_df.loc[results_df.groupby('model')['metric_score'].idxmax()]

csv_data = best_results[['model', 'metric_score', 'scaler']]
csv_data.to_csv("./resultados.csv", index=False)

# Salvar os melhores modelos
os.makedirs("./saved_models", exist_ok=True)
for idx, row in best_results.iterrows():
    model_filename = f"./saved_models/{row['model']}_best_all_features.joblib"
    joblib.dump(row['best_model'], model_filename)

print("\nTreinamento concluído! Resultados salvos em 'resultados_pipeline_best.csv'")
print(csv_data)
