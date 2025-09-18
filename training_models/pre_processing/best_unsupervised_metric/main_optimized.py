import os
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv("./../../../meta_dados/input_data/experiment_0/unsupervised_metric_sample_0.csv")
X = df.drop(columns=["target"]).to_numpy()
y = df["target"].to_numpy()

kf = KFold(n_splits=60, shuffle=True, random_state=42)

# Parâmetros dos modelos (igual ao seu código)
models_config = {
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)),
        'params': {
            'regressor__n_estimators': [50, 100],
            'regressor__learning_rate': [0.1, 0.5, 1.0],
            'regressor__estimator__max_depth': [1, 2, 3],
            'regressor__estimator__min_samples_split': [2, 5],
            'regressor__estimator__min_samples_leaf': [1, 2, 4],
            'regressor__estimator__criterion': ['gini', 'entropy']
        }
    },
    'BaggingClassifier': {
        'model': BaggingClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42)),
        'params': {
            'regressor__n_estimators': [30, 50, 70],
            'regressor__estimator__max_depth': [1, 2, 3, None],
            'regressor__estimator__min_samples_split': [2, 5, 10],
            'regressor__estimator__min_samples_leaf': [1, 2, 4],
            'regressor__estimator__criterion': ['gini', 'entropy']
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'regressor__max_depth': [1, 2, 3, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__criterion': ['gini', 'entropy'],
            'regressor__max_features': ['sqrt', 'log2', None]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'regressor__loss': ['log_loss', 'exponential'],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__n_estimators': [50, 100, 150],
            'regressor__criterion': ['friedman_mse', 'squared_error'],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_depth': [3, 5, 7]
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'regressor__n_neighbors': [3, 5, 7, 10],
            'regressor__weights': ['uniform', 'distance'],
            'regressor__algorithm': ['auto'],
            'regressor__p': [1, 2]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'regressor__solver': ['lbfgs', 'saga'],
            'regressor__penalty': ['l2', None],
            'regressor__C': [0.1, 1, 10],
            'regressor__max_iter': [100, 200]
        }
    },
    'MLPClassifier': {
        'model': MLPClassifier(random_state=42),
        'params': {
            'regressor__activation': ['relu', 'tanh'],
            'regressor__solver': ['adam'],
            'regressor__learning_rate': ['constant'],
            'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'regressor__alpha': [0.0001, 0.001],
            'regressor__max_iter': [100, 200],
            'regressor__early_stopping': [True]
        }
    },
    'NearestCentroid': {
        'model': NearestCentroid(),
        'params': {
            'regressor__metric': ['euclidean', 'manhattan']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'regressor__n_estimators': [50, 100, 150],
            'regressor__criterion': ['gini', 'entropy'],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', None]
        }
    },
    'SVC': {
        'model': SVC(random_state=42),
        'params': {
            'regressor__C': [0.1, 1, 10],
            'regressor__kernel': ['linear', 'rbf'],
            'regressor__gamma': ['scale', 'auto']
        }
    }
}

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler()
}

selectors = {
    'VarianceThreshold': VarianceThreshold(threshold=0.01),
    'RFE': RFE(estimator=LogisticRegression()),
    'SelectFromModel': SelectFromModel(estimator=DecisionTreeClassifier(random_state=42))
}

def save_results(random_search, pipeline_name, filename="./resultados_individual_features_classification.csv"):
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
                    scoring='balanced_accuracy',
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
                print(f"Balanced Accuracy: {random_search.best_score_:.6f}")
            except Exception as e:
                print(f"Erro: {str(e)}")
                continue

print("\nTreinamento concluído! Resultados salvos em 'resultados_individual_features_classification.csv'")

if os.path.exists("./resultados_individual_features_classification.csv"):
    results_df = pd.read_csv("./resultados_individual_features_classification.csv")
    print("\nResumo dos resultados:")
    print(results_df.groupby('model')['metric_score'].agg(['mean', 'std', 'min', 'max']).round(6))
    print("\nMelhor pipeline por modelo:")
    best_by_model = results_df.loc[results_df.groupby('model')['metric_score'].idxmax()]
    print(best_by_model[['pipeline', 'model', 'metric_score']].round(6))
    print("\nTop 10 pipelines com melhor performance:")
    top_pipelines = best_by_model.nlargest(10, 'metric_score')
    print(top_pipelines[['pipeline', 'model', 'metric_score']].round(6))
