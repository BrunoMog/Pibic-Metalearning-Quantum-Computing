import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from sklearn.utils.validation import indexable, check_X_y
# Handle different sklearn versions for safe_indexing import
try:
    from sklearn.utils import safe_indexing
except ImportError:
    try:
        from sklearn.utils._indexing import safe_indexing
    except ImportError:
        try:
            from sklearn.utils._indexing import _safe_indexing as safe_indexing
        except ImportError:
            # Fallback function if all imports fail
            def safe_indexing(X, indices):
                if hasattr(X, 'iloc'):
                    return X.iloc[indices]
                else:
                    return X[indices]
                
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
from collections.abc import Iterable
import time

def custom_accuracy_score(y_true, y_pred):
    """Custom accuracy score function."""
    if len(y_true) == 0:
        return 0.0
    
    correct = 0
    for true, pred in zip(y_true, y_pred):
        # Se y_true é uma lista de listas, verificar se pred está na lista
        if isinstance(true, (list, tuple, np.ndarray)) and hasattr(true, '__iter__'):
            if pred in true:
                correct += 1
        else:
            # Comparação normal
            if pred == true:
                correct += 1
    return correct / len(y_true)

def custom_cross_val_score(estimator, X, y=None, y_custom=None, groups=None, scoring=None, cv=None,
                          n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                          error_score=np.nan, use_y_custom_for_scoring=False,
                          X_original_index=None, scaler=None, oversampler=None, mapping_y_custom=None):
    """
    Custom implementation of cross_val_score with the same interface as sklearn's version.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
        
    X : array-like of shape (n_samples, n_features)
        The data to fit.
        
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.

    y_custom : array-like with different shape than y, default=None
        Custom target variable to use for scoring.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.
        
    scoring : str, callable or None, default=None
        A str (see model evaluation documentation) or a scorer callable object.
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        
    verbose : int, default=0
        The verbosity level.
        
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
        
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
        
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.

    use_y_custom_for_scoring : bool, default=False
        Whether to use the custom target variable y_custom for scoring.

    mapping_y_custom : callable, default=None
        A function to map y_custom to the appropriate format for scoring.

    X_original_index : array-like of shape (n_samples,), default=None
        The original indices of the samples in X.

    Returns
    -------
    scores : array of float of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
    """
    
    # Input validation
    X, y, groups = indexable(X, y, groups)
    
    if fit_params is None:
        fit_params = {}
    
    # Get cross-validation strategy
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # Get scorer - usar custom_accuracy apenas quando y_custom está realmente disponível
    if (scoring == 'custom_accuracy' or 
        (use_y_custom_for_scoring and y_custom is not None and 
         X_original_index is not None and is_classifier(estimator))):
        # Usar função personalizada apenas para classificação quando y_custom está completo
        def scorer_func(estimator, X_test, y_test):
            y_pred = estimator.predict(X_test)
            return custom_accuracy_score(y_test, y_pred)
        scorer = scorer_func
    else:
        # Usar scoring padrão do sklearn (funciona como validação cruzada normal)
        scorer = check_scoring(estimator, scoring=scoring)
    
    scores = []
    
    if verbose > 0:
        print(f"[CV] Starting cross-validation with {cv.get_n_splits(X, y, groups)} splits")
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        start_time = time.time()
        
        if verbose > 1:
            print(f"[CV] Training fold {fold_idx + 1}/{cv.get_n_splits(X, y, groups)}")
        
        # Split the data PRIMEIRO
        X_train = safe_indexing(X, train_idx)
        X_test = safe_indexing(X, test_idx)
        y_train = safe_indexing(y, train_idx) if y is not None else None
        
        # Apply scaling if provided (DEPOIS do split)
        if scaler is not None:
            scaler_fold = clone(scaler)
            X_train = scaler_fold.fit_transform(X_train)
            X_test = scaler_fold.transform(X_test)
        
        # Apply oversampling if provided (only on training data)
        if oversampler is not None and y_train is not None:
            oversampler_fold = clone(oversampler)
            X_train, y_train = oversampler_fold.fit_resample(X_train, y_train)
        
        # Handle y_test based on whether to use y_custom or not
        if use_y_custom_for_scoring and y_custom is not None and X_original_index is not None:
            y_test = []
            for idx in test_idx:
                original_idx = X_original_index[idx]
                y_test.append(y_custom[original_idx])
        else:
            y_test = safe_indexing(y, test_idx) if y is not None else None

        # Clone and fit the estimator
        estimator_fold = clone(estimator)
        
        try:
            # Fit the estimator
            if y_train is not None:
                estimator_fold.fit(X_train, y_train, **fit_params)
            else:
                estimator_fold.fit(X_train, **fit_params)
            
            # Score the estimator
            if y_test is not None:
                score = scorer(estimator_fold, X_test, y_test)
            else:
                score = scorer(estimator_fold, X_test)
            
            scores.append(score)
            
            if verbose > 1:
                end_time = time.time()
                print(f"[CV] Fold {fold_idx + 1} score: {score:.6f} (time: {end_time - start_time:.2f}s)")
                
        except Exception as e:
            if verbose > 0:
                print(f"[CV] Error in fold {fold_idx + 1}: {e}")
            
            if error_score == 'raise':
                raise
            else:
                scores.append(error_score)
    
    if verbose > 0:
        scores_array = np.array(scores)
        print(f"[CV] Cross-validation completed. Mean score: {np.mean(scores_array):.6f} "
              f"(+/- {np.std(scores_array) * 2:.6f})")
    
    return np.array(scores)

