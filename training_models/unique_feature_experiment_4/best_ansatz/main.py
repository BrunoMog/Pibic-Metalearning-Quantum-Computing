import os
import numpy as np
import pandas as pd
import joblib
import warnings
import pickle
import sys

# Importar implementações customizadas
sys.path.append('../../')  # Adicionar caminho para os módulos customizados
from CustomRandomizedSearchCV import CustomRandomizedSearchCV
from CustomCrossValScore import custom_cross_val_score

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Suprimir warnings de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# abrindo os metadados
df = pd.read_csv("./../../../meta_dados/input_data/experiment_4/best_ansatz_sample_4.csv")
X = df.drop(columns=["original_index", "target"]).to_numpy()
X_original_index = df["original_index"].to_numpy()
y = df["target"].to_numpy()

# carregar o mapping de índices originais
with open("./../../../meta_dados/input_data/mapping_ansatz_target.pkl", "rb") as f:
    mapping_y_custom = pickle.load(f)

# experimento 0 é apenas o primeiro dicionario
mapping_y_custom = mapping_y_custom[4]

# criando KFold (usando o mesmo que o arquivo de todas as features)
kf = LeaveOneOut()

# Grids de parâmetros (usando os mesmos do arquivo principal)
param_grid_ada_boost_classifier = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3],
    'estimator__min_samples_split': [2, 5],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__criterion': ['gini', 'entropy'],
    'estimator__max_features': ['sqrt', 'log2'],
    'estimator__max_leaf_nodes': [None, 10, 20]
}

param_grid_bg_bagging_classifier = {
    'n_estimators': [30, 50, 70, 100],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
    'estimator__max_depth': [1, 2, 3, None],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__criterion': ['gini', 'entropy'],
}

param_grid_dt_classifier = {
    'max_depth': [1, 2, 3, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2', None],
    'max_leaf_nodes': [None, 10, 20, 30]
}

param_grid_gb_classifier = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [30, 50, 100, 200],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_depth': [3, 5, 7],
    'min_impurity_decrease': [0.0, 0.1],
    'max_features': ['sqrt', 'log2', None]
}

param_grid_knn_classifier = {
    'n_neighbors': [3, 5, 7, 10, 15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [15, 30, 45, 60],
    'p': [1, 2],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
}

param_grid_logistic_regression = [
    {
        'solver': ['lbfgs', 'sag', 'newton-cholesky', 'newton-cg'],
        'penalty': ['l2', None],
        'C': [0.01, 0.1, 1, 3],
        'fit_intercept': [True, False]
    },
    {
        'solver': ['saga'],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': [0.01, 0.1, 1, 3],
        'fit_intercept': [True, False],
    }
]

param_grid_mlp_classifier = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'hidden_layer_sizes': [
        (50,), (100,), (150,), (200,),
        (50, 50), (100, 100), (150, 150), (200, 200),
        (50, 50, 50), (100, 100, 100)
    ],
    'alpha': [0.00001, 0.0001, 0.001],
    'max_iter': [200, 300]
}

param_grid_nearest_centroid = {
    'metric': ['euclidean', 'manhattan']
}

param_grid_random_forest_classifier = {
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3],
}

# Função para verificar se um experimento já foi executado
def is_experiment_completed(feature_index, model_name, oversampler_name, scaler_name, filename="./resultados.csv"):
    """
    Verifica se um experimento específico já foi executado.
    """
    if not os.path.exists(filename):
        return False
    
    try:
        existing_data = pd.read_csv(filename)
        mask = ((existing_data['feature_index'] == feature_index) & 
                (existing_data['model'] == model_name) &
                (existing_data['oversampling'] == oversampler_name) &
                (existing_data['scaler'] == scaler_name))
        return mask.any()
    except Exception:
        return False

def get_completed_experiments(filename="./resultados.csv"):
    """
    Retorna um conjunto de tuplas dos experimentos já completados.
    """
    if not os.path.exists(filename):
        return set()
    
    try:
        existing_data = pd.read_csv(filename)
        completed = set()
        for _, row in existing_data.iterrows():
            completed.add((row['feature_index'], row['model'], row['oversampling'], row['scaler']))
        return completed
    except Exception:
        return set()

def save_results(best_score, best_model, best_params, feature_index, model_name, oversampler_name, scaler_name, 
                filename="./resultados.csv"):
    """
    Salva os resultados do experimento.
    """
    # Dados da nova linha
    new_data = pd.DataFrame({
        'model': [model_name],
        'metric_score': [best_score],
        'feature_index': [feature_index],
        'oversampling': [oversampler_name],
        'scaler': [scaler_name]
    })
    
    # Verifica se o arquivo existe
    if not os.path.exists(filename):
        new_data.to_csv(filename, index=False)
    else:
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(filename, index=False)

# Dicionário com todos os modelos e seus parâmetros
models_config = {
    'AdaBoostClassifier': {
        'classifier': AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)),
        'params': param_grid_ada_boost_classifier
    },
    'BaggingClassifier': {
        'classifier': BaggingClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)),
        'params': param_grid_bg_bagging_classifier
    },
    'DecisionTreeClassifier': {
        'classifier': DecisionTreeClassifier(random_state=42),
        'params': param_grid_dt_classifier
    },
    'GradientBoostingClassifier': {
        'classifier': GradientBoostingClassifier(random_state=42),
        'params': param_grid_gb_classifier
    },
    'KNeighborsClassifier': {
        'classifier': KNeighborsClassifier(),
        'params': param_grid_knn_classifier
    },
    'LogisticRegression': {
        'classifier': LogisticRegression(max_iter=500, random_state=42),
        'params': param_grid_logistic_regression
    },
    'MLPClassifier': {
        'classifier': MLPClassifier(random_state=42),
        'params': param_grid_mlp_classifier
    },
    'NearestCentroid': {
        'classifier': NearestCentroid(),
        'params': param_grid_nearest_centroid
    },
    'RandomForestClassifier': {
        'classifier': RandomForestClassifier(random_state=42),
        'params': param_grid_random_forest_classifier
    },
    'SVC': {
        'classifier': SVC(probability=True, random_state=42),
        'params': param_grid_svm
    }
}

# Configurações de preprocessing (igual ao arquivo principal)
scalers = [StandardScaler(), MinMaxScaler()]
oversamplers = {
    'RandomOverSampler': RandomOverSampler(random_state=42)
}

# Verificar experimentos já completados
completed_experiments = get_completed_experiments()
total_experiments = len(models_config) * X.shape[1] * len(scalers) * len(oversamplers)
completed_count = len(completed_experiments)

print(f"Dataset possui {X.shape[1]} features")
print(f"Total de experimentos: {total_experiments}")
print(f"Experimentos já completados: {completed_count}")
print(f"Experimentos restantes: {total_experiments - completed_count}")
print("Iniciando treinamento com features individuais para classificação (implementação customizada)...")

if completed_count > 0:
    print("Continuando de onde parou...")

# Treinamento com features individuais
for feature_idx in range(X.shape[1]):
    print(f"\nTreinando com feature {feature_idx + 1}/{X.shape[1]}")
    
    # Selecionar apenas uma feature
    X_single_feature = X[:, feature_idx].reshape(-1, 1)
    # Para features individuais, precisamos ajustar o X_original_index também
    X_original_index_single = X_original_index  # Manter o mesmo mapeamento
    
    for oversampler_name, oversampler in oversamplers.items():
        for scaler in scalers:
            for model_name, config in models_config.items():
                
                # Verificar se este experimento já foi executado
                if (feature_idx, model_name, oversampler_name, scaler.__class__.__name__) in completed_experiments:
                    print(f"  {model_name} + {scaler.__class__.__name__} + {oversampler_name} - JÁ EXECUTADO")
                    continue
                
                print(f"  Treinando {model_name} + {scaler.__class__.__name__} + {oversampler_name}...", end=" ")
                
                try:
                    # Tratar LogisticRegression que tem lista de param_grids
                    param_grids = config['params'] if isinstance(config['params'], list) else [config['params']]
                    
                    best_score_overall = float('-inf')
                    best_model_overall = None
                    best_params_overall = None
                    
                    for param_grid in param_grids:
                        # Usar CustomRandomizedSearchCV com scaler e oversampler integrados
                        custom_search = CustomRandomizedSearchCV(
                            estimator=config['classifier'],
                            param_distributions=param_grid,
                            cv=kf,
                            scoring='accuracy',  # Custom accuracy será usado automaticamente
                            n_iter=50,  
                            n_jobs=-1,  # Usar todos os núcleos disponíveis
                            verbose=0,
                            random_state=42,
                            scaler=scaler,
                            oversampler=oversampler,
                            original_index=X_original_index_single,
                            use_y_custom_for_scoring=True,
                            y_custom=mapping_y_custom,
                            mapping_y_custom=mapping_y_custom
                        )
                        
                        custom_search.fit(X_single_feature, y)
                        
                        if custom_search.best_score_ > best_score_overall:
                            best_score_overall = custom_search.best_score_
                            best_model_overall = custom_search.best_estimator_
                            best_params_overall = custom_search.best_params_
                    
                    # Verificar se os resultados são válidos
                    if best_score_overall is not None and not np.isnan(best_score_overall) and best_model_overall is not None:
                        # Salvar resultados
                        save_results(best_score_overall, best_model_overall, best_params_overall, 
                                   feature_idx, model_name, oversampler_name, scaler.__class__.__name__)
                        
                        # Salvar o melhor modelo
                        model_filename = f"./saved_models/{model_name}_{scaler.__class__.__name__}_{oversampler_name}_feature_{feature_idx}.joblib"
                        os.makedirs("./saved_models", exist_ok=True)
                        joblib.dump(best_model_overall, model_filename)
                        
                        print(f"Accuracy: {best_score_overall:.6f}")
                        
                        # Atualizar contador
                        completed_count += 1
                        print(f"    Progresso: {completed_count}/{total_experiments} ({(completed_count/total_experiments)*100:.1f}%)")
                    else:
                        print("❌ Resultado inválido")
                        
                except Exception as e:
                    print(f"Erro: {str(e)}")
                    continue

print("\nTreinamento concluído! Resultados salvos em 'resultados.csv'")

# Análise dos resultados
if os.path.exists("./resultados.csv"):
    results_df = pd.read_csv("./resultados.csv")
    print("\nResumo dos resultados:")
    print(results_df.groupby('model')['metric_score'].agg(['mean', 'std', 'min', 'max']).round(6))
    
    # Melhor resultado por feature
    print("\nMelhor configuração por feature:")
    best_by_feature = results_df.loc[results_df.groupby('feature_index')['metric_score'].idxmax()]
    print(best_by_feature[['feature_index', 'model', 'scaler', 'oversampling', 'metric_score']].round(6))
    
    # Features mais importantes (baseado na melhor accuracy)
    print("\nTop 10 features com melhor performance:")
    top_features = best_by_feature.nlargest(10, 'metric_score')
    print(top_features[['feature_index', 'model', 'scaler', 'oversampling', 'metric_score']].round(6))
    
    # Análise por scaler e oversampler
    print("\nDesempenho médio por configuração:")
    config_performance = results_df.groupby(['model', 'scaler', 'oversampling'])['metric_score'].agg(['mean', 'std']).round(6)
    print(config_performance.sort_values('mean', ascending=False).head(20))
    
    print(f"\nTotal de resultados salvos: {len(results_df)}")
