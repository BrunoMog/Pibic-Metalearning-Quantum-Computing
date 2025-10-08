import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
import pickle

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

warnings.filterwarnings("ignore", category=UserWarning)

# Carregar dados
df = pd.read_csv("./../../../meta_dados/input_data/experiment_0/best_optimizer_sample_0.csv")
X = df.drop(columns=["original_index", "target"]).to_numpy()
X_original_index = df["original_index"].to_numpy()
y = df["target"].to_numpy()

# carregar o mapping de índices originais
with open("./../../../meta_dados/input_data/mapping_ansatz_target.pkl", "rb") as f:
    mapping_y_custom = pickle.load(f)
mapping_y_custom = mapping_y_custom[0]  # experimento 0

kf = LeaveOneOut()

# parâmetros (mantive seus grids reduzidos para execução razoável)
param_grid_ada_boost_classifier = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2],
}
param_grid_bg_bagging_classifier = {
    'n_estimators': [30, 50],
    'max_samples': [0.5, 1.0],
}
param_grid_dt_classifier = {
    'max_depth': [1, 2, None],
    'min_samples_split': [2, 5],
}
param_grid_gb_classifier = {
    'loss': ['log_loss'],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [30, 50],
}
param_grid_knn_classifier = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
}
param_grid_logistic_regression = [
    {'solver': ['lbfgs'], 'penalty': ['l2', None], 'C': [0.01, 0.1, 1.0]},
    {'solver': ['saga'], 'penalty': ['l1', 'l2', None], 'C': [0.01, 0.1, 1.0]}
]
param_grid_mlp_classifier = {
    'hidden_layer_sizes': [(50,), (100,)],

    'activation': ['relu', 'tanh'],
    'alpha': [1e-4, 1e-3]
}
param_grid_nearest_centroid = {'metric': ['euclidean', 'manhattan']}
param_grid_random_forest_classifier = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
param_grid_svm = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}

models_config = {
    'AdaBoostClassifier': {'classifier': AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)), 'params': param_grid_ada_boost_classifier},
    'BaggingClassifier': {'classifier': BaggingClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)), 'params': param_grid_bg_bagging_classifier},
    'DecisionTreeClassifier': {'classifier': DecisionTreeClassifier(random_state=42), 'params': param_grid_dt_classifier},
    'GradientBoostingClassifier': {'classifier': GradientBoostingClassifier(random_state=42), 'params': param_grid_gb_classifier},
    'KNeighborsClassifier': {'classifier': KNeighborsClassifier(), 'params': param_grid_knn_classifier},
    'LogisticRegression': {'classifier': LogisticRegression(max_iter=500, random_state=42), 'params': param_grid_logistic_regression},
    'MLPClassifier': {'classifier': MLPClassifier(random_state=42), 'params': param_grid_mlp_classifier},
    'NearestCentroid': {'classifier': NearestCentroid(), 'params': param_grid_nearest_centroid},
    'RandomForestClassifier': {'classifier': RandomForestClassifier(random_state=42), 'params': param_grid_random_forest_classifier},
    'SVC': {'classifier': SVC(probability=True, random_state=42), 'params': param_grid_svm}
}

scalers = [StandardScaler(), MinMaxScaler()]
oversamplers = {'RandomOverSampler': RandomOverSampler(random_state=42)}

RESULTS_FILE = "./resultados.csv"
SAVED_MODELS_DIR = "./saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def upsert_best_result_classification(best_score, best_model_obj, best_params, feature_index, model_name, oversampler_name, scaler_name, filename=RESULTS_FILE):
    """
    Insere/atualiza o melhor resultado por (model_name, feature_index).
    Salva o modelo em saved_models/{model_name}_feature_{feature_index}.joblib apenas se melhorar.
    """
    if os.path.exists(filename):
        try:
            existing = pd.read_csv(filename)
        except Exception:
            existing = pd.DataFrame(columns=['model','metric_score','feature_index','oversampling','scaler','params'])
    else:
        existing = pd.DataFrame(columns=['model','metric_score','feature_index','oversampling','scaler','params'])

    mask = (existing['model'] == model_name) & (existing['feature_index'] == feature_index)
    if mask.any():
        current_score = existing.loc[mask, 'metric_score'].astype(float).max()
        if float(best_score) > float(current_score):
            existing = existing[~mask]
            new_row = {'model': model_name, 'metric_score': best_score, 'feature_index': feature_index, 'oversampling': oversampler_name, 'scaler': scaler_name, 'params': str(best_params)}
            existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        else:
            return
    else:
        new_row = {'model': model_name, 'metric_score': best_score, 'feature_index': feature_index, 'oversampling': oversampler_name, 'scaler': scaler_name, 'params': str(best_params)}
        existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)

    existing.to_csv(filename, index=False)

    # salvar modelo único por model+feature (sem scaler/oversampler no nome)
    model_filename = os.path.join(SAVED_MODELS_DIR, f"{model_name}_feature_{feature_index}.joblib")
    try:
        joblib.dump(best_model_obj, model_filename)
    except Exception:
        with open(model_filename + ".pkl", "wb") as f:
            pickle.dump(best_model_obj, f)

print(f"Dataset possui {X.shape[1]} features")
print("Iniciando treinamento de ansatz por feature (mantendo apenas melhor por modelo+feature)...")

for feature_idx in range(X.shape[1]):
    print(f"\nTreinando com feature {feature_idx + 1}/{X.shape[1]}")
    X_single_feature = X[:, feature_idx].reshape(-1, 1)
    X_original_index_single = X_original_index

    for oversampler_name, oversampler in oversamplers.items():
        for scaler in scalers:
            for model_name, config in models_config.items():
                print(f"  Treinando {model_name} + {scaler.__class__.__name__} + {oversampler_name}...", end=" ")
                try:
                    param_grids = config['params'] if isinstance(config['params'], list) else [config['params']]

                    best_score_overall = float('-inf')
                    best_model_overall = None
                    best_params_overall = None

                    for param_grid in param_grids:
                        custom_search = CustomRandomizedSearchCV(
                            estimator=config['classifier'],
                            param_distributions=param_grid,
                            cv=kf,
                            scoring='accuracy',
                            n_iter=50,
                            n_jobs=-1,
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

                    if best_score_overall is not None and not np.isnan(best_score_overall) and best_model_overall is not None:
                        upsert_best_result_classification(best_score_overall, best_model_overall, best_params_overall,
                                                         feature_idx, model_name, oversampler_name, scaler.__class__.__name__, filename=RESULTS_FILE)

                        print(f"Accuracy: {best_score_overall:.6f}")
                    else:
                        print("❌ Resultado inválido")

                except Exception as e:
                    print(f"Erro: {e}")
                    continue

print("\nTreinamento concluído! Resultados salvos em 'resultados.csv'")

# Análise dos resultados
if os.path.exists(RESULTS_FILE):
    results_df = pd.read_csv(RESULTS_FILE)
    print("\nResumo dos resultados:")
    print(results_df.groupby('model')['metric_score'].agg(['mean', 'std', 'min', 'max']).round(6))

    print("\nMelhor configuração por feature:")
    best_by_feature = results_df.loc[results_df.groupby('feature_index')['metric_score'].idxmax()]
    print(best_by_feature[['feature_index', 'model', 'scaler', 'oversampling', 'metric_score']].round(6))

    print("\nTop 10 features com melhor performance:")
    top_features = best_by_feature.nlargest(10, 'metric_score')
    print(top_features[['feature_index', 'model', 'scaler', 'oversampling', 'metric_score']].round(6))

    print("\nDesempenho médio por configuração:")
    config_performance = results_df.groupby(['model', 'scaler', 'oversampling'])['metric_score'].agg(['mean', 'std']).round(6)
    print(config_performance.sort_values('mean', ascending=False).head(20))

    print(f"\nTotal de resultados salvos: {len(results_df)}")
