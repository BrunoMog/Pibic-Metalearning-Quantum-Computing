import os
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

# Suprimir warnings de convergência
warnings.filterwarnings("ignore", category=UserWarning)

# abrindo os metadados
df = pd.read_csv("./../../../meta_dados/input_data/experiment_0/best_ansatz_sample_0.csv")
X = df.drop(columns=["target"]).to_numpy()
y = df["target"].to_numpy()

# criando KFold
kf = KFold(n_splits=60, shuffle=True, random_state=42)  

# Grids de parâmetros otimizados para modelos de classificação
ada_boost_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3],
    'estimator__min_samples_split': [2, 5],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__criterion': ['gini', 'entropy']
}

bagging_param_grid = {
    'n_estimators': [30, 50, 70],
    'estimator__max_depth': [1, 2, 3, None],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__criterion': ['gini', 'entropy']
}

dt_param_grid = {
    'max_depth': [1, 2, 3, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}

gb_param_grid = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_depth': [3, 5, 7]
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto'],
    'p': [1, 2]
}

lr_param_grid = {
    'solver': ['lbfgs', 'saga'],
    'penalty': ['l2', None],
    'C': [0.1, 1, 10],
    'max_iter': [100, 200]
}

mlp_param_grid = {
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'learning_rate': ['constant'],
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'alpha': [0.0001, 0.001],
    'max_iter': [100, 200],
    'early_stopping': [True]
}

nearest_centroid_param_grid = {
    'metric': ['euclidean', 'manhattan']
}

rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Função para verificar se um experimento já foi executado
def is_experiment_completed(feature_index, model_name, filename="./resultados_individual_features_classification.csv"):
    """
    Verifica se um experimento específico (feature + modelo) já foi executado.
    
    Args:
        feature_index (int): Índice da feature
        model_name (str): Nome do modelo
        filename (str): Nome do arquivo CSV
    
    Returns:
        bool: True se o experimento já foi executado, False caso contrário
    """
    if not os.path.exists(filename):
        return False
    
    try:
        existing_data = pd.read_csv(filename)
        # Verifica se existe uma linha com essa feature e modelo
        mask = (existing_data['feature_index'] == feature_index) & (existing_data['model'] == model_name)
        return mask.any()
    except Exception:
        return False

# Função para obter experimentos já completados
def get_completed_experiments(filename="./resultados_individual_features_classification.csv"):
    """
    Retorna um conjunto de tuplas (feature_index, model_name) dos experimentos já completados.
    """
    if not os.path.exists(filename):
        return set()
    
    try:
        existing_data = pd.read_csv(filename)
        completed = set()
        for _, row in existing_data.iterrows():
            completed.add((row['feature_index'], row['model']))
        return completed
    except Exception:
        return set()

# salvar resultados
def save_results(random_search, feature_index, filename="./resultados_individual_features_classification.csv"):
    """
    Verifica se o arquivo CSV existe, cria se não existir, e adiciona uma nova linha com model, best_score_ e feature.
    
    Args:
        random_search (RandomizedSearchCV): Objeto RandomizedSearchCV treinado
        feature_index (int): Índice da feature utilizada
        filename (str): Nome do arquivo CSV (padrão: './resultados_individual_features_classification.csv')
    """
    # Obtém o nome do modelo
    model_name = random_search.estimator.__class__.__name__
    # Obtém o melhor score (accuracy para classificação)
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

# Dicionário com todos os modelos e seus parâmetros
models_config = {
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)),
        'params': ada_boost_param_grid
    },
    'BaggingClassifier': {
        'model': BaggingClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)),
        'params': bagging_param_grid
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': dt_param_grid
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': gb_param_grid
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': knn_param_grid
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42),
        'params': lr_param_grid
    },
    'MLPClassifier': {
        'model': MLPClassifier(random_state=42),
        'params': mlp_param_grid
    },
    'NearestCentroid': {
        'model': NearestCentroid(),
        'params': nearest_centroid_param_grid
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': rf_param_grid
    },
    'SVC': {
        'model': SVC(random_state=42),
        'params': svc_param_grid
    }
}

# Verificar experimentos já completados
completed_experiments = get_completed_experiments()
total_experiments = len(models_config) * X.shape[1]
completed_count = len(completed_experiments)

print(f"Dataset possui {X.shape[1]} features")
print(f"Total de experimentos: {total_experiments}")
print(f"Experimentos já completados: {completed_count}")
print(f"Experimentos restantes: {total_experiments - completed_count}")
print("Iniciando treinamento com features individuais para classificação (versão otimizada)...")

if completed_count > 0:
    print("Continuando de onde parou...")

# Treinamento com features individuais
for feature_idx in range(X.shape[1]):
    print(f"\nTreinando com feature {feature_idx + 1}/{X.shape[1]}")
    
    # Selecionar apenas uma feature
    X_single_feature = X[:, feature_idx].reshape(-1, 1)
    
    # Treinar cada modelo
    for model_name, config in models_config.items():
        # Verificar se este experimento já foi executado
        if (feature_idx, model_name) in completed_experiments:
            print(f"  {model_name} - JÁ EXECUTADO (pulando)")
            continue
            
        print(f"  Treinando {model_name}...", end=" ")
        
        try:
            # Configurar o RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=config['model'],
                param_distributions=config['params'],
                cv=kf,
                scoring='accuracy',  # Métrica de classificação
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
            
            print(f"Accuracy: {random_search.best_score_:.6f}")
            
            # Atualizar contador
            completed_count += 1
            print(f"  Progresso: {completed_count}/{total_experiments} ({(completed_count/total_experiments)*100:.1f}%)")
            
        except Exception as e:
            print(f"Erro: {str(e)}")
            continue

print("\nTreinamento concluído! Resultados salvos em 'resultados_individual_features_classification.csv'")

# Análise dos resultados
if os.path.exists("./resultados_individual_features_classification.csv"):
    results_df = pd.read_csv("./resultados_individual_features_classification.csv")
    print("\nResumo dos resultados:")
    print(results_df.groupby('model')['metric_score'].agg(['mean', 'std', 'min', 'max']).round(6))
    
    # Melhor resultado por feature
    print("\nMelhor modelo por feature:")
    best_by_feature = results_df.loc[results_df.groupby('feature_index')['metric_score'].idxmax()]
    print(best_by_feature[['feature_index', 'model', 'metric_score']].round(6))
    
    # Features mais importantes (baseado na melhor accuracy)
    print("\nTop 10 features com melhor performance:")
    top_features = best_by_feature.nlargest(10, 'metric_score')  # nlargest para accuracy
    print(top_features[['feature_index', 'model', 'metric_score']].round(6))
    
    print(f"\nTotal de resultados salvos: {len(results_df)}")
