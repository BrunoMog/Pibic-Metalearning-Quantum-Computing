import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
import pickle

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)

# Caminho do CSV de entrada (ajuste se necessário)
INPUT_CSV = "./../../../meta_dados/input_data/experiment_0/best_accuracy_sample_0.csv"

# Ler dados
df = pd.read_csv(INPUT_CSV)
# remover coluna original_index se existir
df = df.drop(columns=["original_index"], errors='ignore')
X = df.drop(columns=["target"]).to_numpy()
y = df["target"].to_numpy()

print(f"DEBUG: df.shape={df.shape} | X.shape={X.shape} | n_classes={(np.unique(y)).size}")

# KFold para classificação com features individuais
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# modelos e grids (grids pequenos para execução razoável)
models_config = {
    'AdaBoostClassifier': {
        'classifier': AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.5, 1.0]
        }
    },
    'BaggingClassifier': {
        'classifier': BaggingClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)),
        'params': {
            'n_estimators': [30, 50],
            'max_samples': [0.5, 1.0]
        }
    },
    'DecisionTreeClassifier': {
        'classifier': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [None, 3, 5],
            'min_samples_split': [2, 5]
        }
    },
    'GradientBoostingClassifier': {
        'classifier': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1]
        }
    },
    'KNeighborsClassifier': {
        'classifier': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    },
    'LogisticRegression': {
        'classifier': LogisticRegression(max_iter=500, random_state=42),
        'params': {
            'C': [0.01, 0.1, 1.0],
            'penalty': ['l2', None],
            'solver': ['lbfgs']
        }
    },
    'MLPClassifier': {
        'classifier': MLPClassifier(random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
            'alpha': [1e-4, 1e-3]
        }
    },
    'NearestCentroid': {
        'classifier': NearestCentroid(),
        'params': {
            'metric': ['euclidean', 'manhattan']
        }
    },
    'RandomForestClassifier': {
        'classifier': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10]
        }
    },
    'SVC': {
        'classifier': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }
}

scalers = [None, StandardScaler(), MinMaxScaler()]
oversamplers = {'none': None, 'RandomOverSampler': RandomOverSampler(random_state=42)}

RESULTS_FILE = "./resultados.csv"
SAVED_MODELS_DIR = "./saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def prefix_params(params, prefix="clf__"):
    """Adiciona prefixo aos nomes de hyperparams para usar dentro do pipeline ('clf__param')."""
    pref = {}
    for k, v in params.items():
        pref[f"{prefix}{k}"] = v
    return pref

def upsert_best_result_classification(best_score, pipeline_obj, feature_index, model_name, oversampler_name, scaler_name, filename=RESULTS_FILE):
    """
    Insere ou atualiza o melhor resultado (maior score) para (model_name, feature_index).
    Salva pipeline_obj em saved_models/{model_name}_feature_{feature_index}.joblib quando melhora.
    """
    if os.path.exists(filename):
        try:
            existing = pd.read_csv(filename)
        except Exception:
            existing = pd.DataFrame(columns=['model','metric_score','feature_index','oversampling','scaler'])
    else:
        existing = pd.DataFrame(columns=['model','metric_score','feature_index','oversampling','scaler'])

    mask = (existing['model'] == model_name) & (existing['feature_index'] == feature_index)
    if mask.any():
        current_score = existing.loc[mask, 'metric_score'].astype(float).max()
        if float(best_score) > float(current_score):
            existing = existing[~mask]
            new_row = {'model': model_name, 'metric_score': best_score, 'feature_index': feature_index, 'oversampling': oversampler_name, 'scaler': scaler_name}
            existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        else:
            return
    else:
        new_row = {'model': model_name, 'metric_score': best_score, 'feature_index': feature_index, 'oversampling': oversampler_name, 'scaler': scaler_name}
        existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)

    existing.to_csv(filename, index=False)

    # salvar pipeline que gerou o resultado (nome sem scaler/oversampler para consistência)
    model_filename = os.path.join(SAVED_MODELS_DIR, f"{model_name}_feature_{feature_index}.joblib")
    try:
        joblib.dump(pipeline_obj, model_filename)
    except Exception:
        with open(model_filename + ".pkl", "wb") as f:
            pickle.dump(pipeline_obj, f)

print(f"Dataset possui {X.shape[1]} features")
print("Iniciando busca de melhor acurácia por feature (isolado)...")

for feature_idx in range(X.shape[1]):
    print(f"\n=== FEATURE {feature_idx+1}/{X.shape[1]} ===")
    X_single_feature = X[:, feature_idx].reshape(-1, 1)

    for scaler in scalers:
        scaler_name = "None" if scaler is None else scaler.__class__.__name__
        for overs_name, overs in oversamplers.items():
            for model_name, cfg in models_config.items():
                print(f"  Testando {model_name} | scaler={scaler_name} | oversampler={overs_name} ...", end=" ")
                try:
                    steps = []
                    if scaler is not None:
                        steps.append(("scaler", scaler))
                    if overs is not None:
                        steps.append(("oversampler", overs))
                    steps.append(("clf", cfg['classifier']))
                    pipeline = ImbPipeline(steps=steps)

                    # ajustar param_grid para o pipeline
                    param_grid = prefix_params(cfg['params'], prefix="clf__")

                    random_search = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=param_grid,
                        cv=kf,
                        scoring='accuracy',
                        n_iter=50,
                        n_jobs=-1,
                        verbose=0,
                        random_state=42
                    )

                    random_search.fit(X_single_feature, y)

                    best_score = random_search.best_score_
                    best_pipeline = random_search.best_estimator_

                    # inserir/atualizar apenas se for o melhor por model+feature
                    upsert_best_result_classification(best_score, best_pipeline, feature_idx, model_name, overs_name, scaler_name)

                    print(f"acc={best_score:.6f}")

                except Exception as e:
                    print(f"Erro: {e}")
                    continue

print("\nProcesso concluído. Arquivo de resultados:", RESULTS_FILE)

# Sumário final
if os.path.exists(RESULTS_FILE):
    results_df = pd.read_csv(RESULTS_FILE)
    print("\nResumo dos resultados:")
    print(results_df.groupby('model')['metric_score'].agg(['mean','std','min','max']).round(6))
    best_by_feature = results_df.loc[results_df.groupby('feature_index')['metric_score'].idxmax()].reset_index(drop=True)
    print("\nMelhor por feature:")
    print(best_by_feature[['feature_index','model','metric_score','scaler','oversampling']].head(20))
