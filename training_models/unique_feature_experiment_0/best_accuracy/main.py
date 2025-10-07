import os
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Suprimir warnings de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# abrindo os metadados
df = pd.read_csv("./../../../meta_dados/input_data/experiment_0/best_accuracy_sample_0.csv")
X = df.drop(columns=["target"]).to_numpy()
y = df["target"].to_numpy()

# criando KFold
kf = KFold(n_splits=60, shuffle=True, random_state=42) 

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

# salvar resultados
def save_results(random_search, feature_index, filename="./resultados_individual_features_optimized.csv"):
    """
    Verifica se o arquivo CSV existe, cria se não existir, e adiciona uma nova linha com model, best_score_ e feature.
    
    Args:
        random_search (RandomizedSearchCV): Objeto RandomizedSearchCV treinado
        feature_index (int): Índice da feature utilizada
        filename (str): Nome do arquivo CSV (padrão: './resultados_individual_features_optimized.csv')
    """
    # Obtém o nome do modelo
    model_name = random_search.estimator.__class__.__name__
    # Obtém o melhor score (neg_mean_squared_error, convertido para positivo se necessário)
    metric_score = random_search.best_score_
    
    # Dados da nova linha
    new_data = pd.DataFrame({
        'model': [model_name],
        'metric_score': [metric_score],
        'feature_index': [feature_index]
    })
    
    # Verifica se o arquivo existe
    if not os.path.exists(filename):
        # Cria o arquivo com as colunas model, metric_score e feature_index
        new_data.to_csv(filename, index=False)
    else:
        # Adiciona a nova linha ao arquivo existente
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(filename, index=False)

# Dicionário com todos os modelos e seus parâmetros - CORRIGIDO
models_config = {
    'AdaBoostRegressor': {
        'model': AdaBoostRegressor(random_state=42, estimator=DecisionTreeRegressor(random_state=42)),
        'params': param_grid_ada_boost_regressor  # CORRIGIDO
    },
    'DecisionTreeRegressor': {  # ADICIONADO
        'model': DecisionTreeRegressor(random_state=42),
        'params': param_grid_decision_tree_regressor
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(),
        'params': param_grid_knn_regressor  # CORRIGIDO
    },
    'LinearRegression': {
        'model': LinearRegression(),
        'params': param_grid_linear_regressor  # CORRIGIDO
    },
    'MLPRegressor': {
        'model': MLPRegressor(random_state=42),
        'params': param_grid_mlp_regressor  # CORRIGIDO
    },
    'SVR': {
        'model': SVR(),
        'params': param_grid_svr  # CORRIGIDO
    },
    'RandomForestRegressor': {  # ADICIONADO
        'model': RandomForestRegressor(random_state=42),
        'params': param_grid_random_forest_regressor
    }
}

# Treinamento com features individuais
print(f"Dataset possui {X.shape[1]} features")
print("Iniciando treinamento com features individuais (versão otimizada)...")

for feature_idx in range(X.shape[1]):
    print(f"\nTreinando com feature {feature_idx + 1}/{X.shape[1]}")
    
    # Selecionar apenas uma feature
    X_single_feature = X[:, feature_idx].reshape(-1, 1)
    
    # Treinar cada modelo
    for model_name, config in models_config.items():
        print(f"  Treinando {model_name}...", end=" ")
        
        try:
            # Tratar MLPRegressor que tem lista de param_grids
            if isinstance(config['params'], list):
                # Para MLPRegressor que tem múltiplos param_grids
                best_score_overall = float('-inf')
                best_model_overall = None
                
                for param_grid in config['params']:
                    random_search = RandomizedSearchCV(
                        estimator=config['model'],
                        param_distributions=param_grid,
                        cv=kf,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0,
                        random_state=42,
                        n_iter=20  # Reduzido por param_grid
                    )
                    
                    random_search.fit(X_single_feature, y)
                    
                    if random_search.best_score_ > best_score_overall:
                        best_score_overall = random_search.best_score_
                        best_model_overall = random_search.best_estimator_
                
                # Criar um objeto mock para save_results
                class MockRandomSearch:
                    def __init__(self, estimator, best_score):
                        self.estimator = estimator
                        self.best_score_ = best_score
                        self.best_estimator_ = best_model_overall
                
                mock_search = MockRandomSearch(config['model'], best_score_overall)
                save_results(mock_search, feature_idx)
                
                # Salvar o melhor modelo
                model_filename = f"./saved_models/{model_name}_feature_{feature_idx}.joblib"
                os.makedirs("./saved_models", exist_ok=True)
                joblib.dump(best_model_overall, model_filename)
                
                print(f"MSE: {-best_score_overall:.6f}")
                
            else:
                # Para modelos com param_grid único
                random_search = RandomizedSearchCV(
                    estimator=config['model'],
                    param_distributions=config['params'],
                    cv=kf,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0,
                    random_state=42,
                    n_iter=40
                )
                
                # Executar o RandomizedSearchCV
                random_search.fit(X_single_feature, y)
                
                # Salvar resultados
                save_results(random_search, feature_idx)
                
                # Salvar o melhor modelo
                model_filename = f"./saved_models/{model_name}_feature_{feature_idx}.joblib"
                os.makedirs("./saved_models", exist_ok=True)
                joblib.dump(random_search.best_estimator_, model_filename)
                
                print(f"MSE: {-random_search.best_score_:.6f}")
            
        except Exception as e:
            print(f"Erro: {str(e)}")
            continue

print("\nTreinamento concluído! Resultados salvos em 'resultados_individual_features_optimized.csv'")

# Análise dos resultados
if os.path.exists("./resultados_individual_features_optimized.csv"):
    results_df = pd.read_csv("./resultados_individual_features_optimized.csv")
    print("\nResumo dos resultados:")
    print(results_df.groupby('model')['metric_score'].agg(['mean', 'std', 'min', 'max']).round(6))
    
    # Melhor resultado por feature
    print("\nMelhor modelo por feature:")
    best_by_feature = results_df.loc[results_df.groupby('feature_index')['metric_score'].idxmax()]
    print(best_by_feature[['feature_index', 'model', 'metric_score']].round(6))
    
    # Features mais importantes (baseado no melhor MSE)
    print("\nTop 10 features com melhor performance:")
    top_features = best_by_feature.nsmallest(10, 'metric_score')
    print(top_features[['feature_index', 'model', 'metric_score']].round(6))
