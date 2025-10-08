from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
import joblib
import warnings
import os
import pickle
import numpy as np  

# Importar implementações customizadas
import sys
sys.path.append('../../')  # Adicionar caminho para os módulos customizados
from CustomRandomizedSearchCV import CustomRandomizedSearchCV
from CustomCrossValScore import custom_cross_val_score

# Suprimir warnings
warnings.filterwarnings("ignore", category=UserWarning)

# carregar dados
df = pd.read_csv("./../../../meta_dados/input_data/experiment_3/best_optimizer_sample_3.csv")
X = df.drop(columns=["original_index","target"]).to_numpy()
X_original_index = df["original_index"].to_numpy()
y = df["target"].to_numpy()

# carregar o mapping de índices originais
with open("./../../../meta_dados/input_data/mapping_embedding_target.pkl", "rb") as f:
    mapping_y_custom = pickle.load(f)

mapping_y_custom = mapping_y_custom[3]

kf = LeaveOneOut()

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


scalers = [StandardScaler(), MinMaxScaler()]
oversamplers = {
    'RandomOverSampler': RandomOverSampler(random_state=42)
}

results = []

for oversampler_name, oversampler in list(oversamplers.items()):  
    for scaler in scalers:  
        for model_name, config in list(models_config.items()):  
            print(f"\nTreinando {model_name} com {oversampler_name} e {scaler.__class__.__name__}...", end=" ")
            try:
                # Usar implementação customizada - sem pipeline
                param_grids = config['params'] if isinstance(config['params'], list) else [config['params']]
                
                for param_grid in param_grids:

                    # Usar CustomRandomizedSearchCV com scaler e oversampler integrados
                    custom_search = CustomRandomizedSearchCV(
                        estimator=config['classifier'],
                        param_distributions=param_grid,
                        cv=kf,
                        scoring='accuracy',  # Usar accuracy padrão - custom_accuracy será usado automaticamente se y_custom estiver ativo
                        n_iter=50,  # Reduzido para testes mais rápidos
                        n_jobs=-1,  # Paralelização com 18 processos
                        verbose=0,  # Verbose mais detalhado para ver paralelização
                        random_state=42,
                        scaler=scaler,  # Scaler integrado
                        oversampler=oversampler,  # Oversampler integrado
                        original_index=X_original_index,  # Índices originais
                        use_y_custom_for_scoring=True,  # Ativar y_custom para scoring
                        y_custom=mapping_y_custom, 
                        mapping_y_custom=mapping_y_custom  # Usar mapping_y_custom para o mapeamento
                    )
                    
                    custom_search.fit(X, y)
                    best_score = custom_search.best_score_
                    best_model = custom_search.best_estimator_
                    
                    # Verificar se os resultados são válidos antes de adicionar
                    if best_score is not None and not np.isnan(best_score) and best_model is not None:
                        results.append({
                            'model': model_name,
                            'metric_score': best_score,
                            'oversampling': oversampler_name,
                            'scaler': scaler.__class__.__name__,
                            'best_model': best_model,
                            'best_params': custom_search.best_params_
                        })

            except Exception as e:
                print(f"Erro: {str(e)}")
                import traceback
                print(f"Traceback completo: {traceback.format_exc()}")
                continue

# Salvar melhores resultados
if results:
    results_df = pd.DataFrame(results)
    
    # Filtrar apenas resultados válidos (sem NaN)
    valid_results = results_df.dropna(subset=['metric_score'])
    
    if not valid_results.empty:
        # Obter melhores resultados por modelo
        best_results = valid_results.loc[valid_results.groupby('model')['metric_score'].idxmax()]
        
        # Salvar CSV
        csv_data = best_results[['model', 'metric_score', 'oversampling', 'scaler']]
        csv_data.to_csv("./resultados.csv", index=False)
        
        # Salvar os melhores modelos
        os.makedirs("./saved_models", exist_ok=True)
        for idx, row in best_results.iterrows():
            if row['best_model'] is not None:  # Verificar se o modelo não é None
                model_filename = f"./saved_models/{row['model']}_best.joblib"
                joblib.dump(row['best_model'], model_filename)
                print(f"Modelo salvo: {model_filename}")
        
        print(f"\n✅ Resultados salvos com sucesso!")
        print(f"Total de modelos válidos: {len(best_results)}")
        print(f"Arquivo CSV: ./resultados.csv")
    else:
        print("❌ Nenhum resultado válido encontrado.")
else:
    print("❌ Nenhum resultado para salvar.")
