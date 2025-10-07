import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.base import clone
import warnings
from joblib import Parallel, delayed
from CustomCrossValScore import custom_cross_val_score


def _evaluate_parameter_combination(estimator, params, X, y, groups, scoring, cv, n_jobs_inner, 
                                   verbose, fit_params, error_score, use_y_custom_for_scoring,
                                   original_index, scaler, oversampler, mapping_y_custom, i):
    """
    Helper function to evaluate a single parameter combination.
    Used for parallel execution in CustomRandomizedSearchCV.
    """
    if verbose > 1:
        print(f"[CV {i+1}] Testing parameters: {params}")
        
    # Clone estimator and set parameters
    estimator_clone = clone(estimator)
    try:
        estimator_clone.set_params(**params)
    except ValueError as e:
        if verbose > 0:
            print(f"[CV] Skipping invalid parameters: {params}")
            print(f"[CV] Error: {e}")
        return None
    
    try:
        # Perform cross-validation
        scores = custom_cross_val_score(
            estimator_clone, X, y, 
            y_custom=mapping_y_custom,  # Usar mapping_y_custom para y_custom
            groups=groups, 
            scoring=scoring,
            cv=cv, 
            n_jobs=n_jobs_inner,  # Use inner n_jobs for CV
            verbose=0,
            fit_params=fit_params,
            error_score=error_score,
            use_y_custom_for_scoring=use_y_custom_for_scoring,
            X_original_index=original_index,
            scaler=scaler,
            oversampler=oversampler,
            mapping_y_custom=mapping_y_custom
        )
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if verbose > 1:
            print(f"[CV {i+1}] Score: {mean_score:.6f} (+/- {std_score * 2:.6f})")
        elif verbose > 0:
            print(f"[CV {i+1}] {mean_score:.6f}")
    
        return {
            'mean_test_score': mean_score,
            'std_test_score': std_score,
            'params': params,
            'rank_test_score': None,  # Will be filled later
            'index': i,
            'fitted_estimator': estimator_clone if mean_score != error_score else None
        }
        
    except Exception as e:
        if verbose > 0:
            print(f"[CV] Error with parameters {params}: {e}")
            
        if error_score == 'raise':
            raise
        else:
            # Return error result
            return {
                'mean_test_score': error_score,
                'std_test_score': np.nan,
                'params': params,
                'rank_test_score': None,
                'index': i,
                'fitted_estimator': None
            }


class CustomRandomizedSearchCV:
    """
    Custom implementation of RandomizedSearchCV with the same interface as sklearn's version.
    
    Parameters
    ----------
    estimator : estimator object
        A object of that type is instantiated for each grid point.
        
    param_distributions : dict or list of dict
        Dictionary with parameters names as keys and distributions
        or lists of parameters to try.
        
    n_iter : int, default=10
        Number of parameter settings that are sampled.
        
    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on the test set.
        
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        
    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole dataset.
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        
    verbose : int, default=0
        Controls the verbosity.
        
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
        
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling.
        
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        
    return_train_score : bool, default=False
        If False, the cv_results_ attribute will not include training scores.
    """
    
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score=np.nan, return_train_score=False,
                 original_index=None, use_y_custom_for_scoring=False, y_custom=None,
                 scaler=None, oversampler=None, mapping_y_custom=None):
        
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.scaler = scaler
        self.oversampler = oversampler
        self.mapping_y_custom = mapping_y_custom
        
        # Attributes that will be set after fitting
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_index_ = None
        self.scorer_ = None
        self.n_splits_ = None
        self.cv_results_ = None
        self.original_index = original_index
        self.use_y_custom_for_scoring = use_y_custom_for_scoring
        self.y_custom = y_custom
        
    def fit(self, X, y=None, groups=None, **fit_params):
        # Generate parameter combinations
        param_sampler = ParameterSampler(
            self.param_distributions, 
            n_iter=self.n_iter, 
            random_state=self.random_state
        )
        
        param_combinations = list(param_sampler)
        
        # Get number of CV splits correctly
        from sklearn.model_selection import check_cv
        cv_validated = check_cv(self.cv, y, classifier=hasattr(self.estimator, 'predict_proba'))
        n_splits = cv_validated.get_n_splits(X, y, groups)
        
        if self.verbose > 0:
            print(f"Fitting {n_splits} folds for each of {len(param_combinations)} candidates, "
                f"totalling {n_splits * len(param_combinations)} fits")
            
        # Determine n_jobs for inner CV (set to 1 if outer parallelization is used)
        n_jobs_inner = 1 if self.n_jobs != 1 and self.n_jobs is not None else None
            
        parallel_results = Parallel(
            n_jobs=self.n_jobs, 
            verbose=max(0, self.verbose - 1),
            pre_dispatch=self.pre_dispatch
        )(
            delayed(_evaluate_parameter_combination)(
                self.estimator, params, X, y, groups, self.scoring, self.cv, n_jobs_inner,
                self.verbose, fit_params, self.error_score, self.use_y_custom_for_scoring,
                self.original_index, self.scaler, self.oversampler, self.mapping_y_custom, i
            )
            for i, params in enumerate(param_combinations)
        )

        if self.verbose > 0:
            valid_results = len([r for r in parallel_results if r is not None])
            failed_results = len(parallel_results) - valid_results
            print(f"✅ Completed: {valid_results} successful, {failed_results} failed parameter combinations")

        # Filter out None results (failed parameter combinations)
        results = [result for result in parallel_results if result is not None]
        all_params = [result['params'] for result in results]
        
        # Find best result and refit if needed
        best_score = -np.inf
        best_params = None
        best_estimator = None
        best_result = None
        
        for result in results:
            if result['mean_test_score'] > best_score:
                best_score = result['mean_test_score']
                best_params = result['params']
                best_result = result
                
                # Refit with best parameters if requested
                if self.refit:
                    best_estimator = clone(self.estimator)
                    best_estimator.set_params(**best_params)
                    best_estimator.fit(X, y, **fit_params)
        
        if not results:
            raise ValueError("All parameter combinations failed")
        
        # Sort results by score and assign ranks
        results_sorted = sorted(results, key=lambda x: x['mean_test_score'], reverse=True)
        for i, result in enumerate(results_sorted):
            result['rank_test_score'] = i + 1
        
        # Find the best result
        best_result = results_sorted[0]
        
        # Store attributes
        self.best_score_ = best_result['mean_test_score']
        self.best_params_ = best_result['params']
        self.best_index_ = results.index(best_result)
        self.best_estimator_ = best_estimator
        self.cv_results_ = {
            'mean_test_score': [r['mean_test_score'] for r in results],
            'std_test_score': [r['std_test_score'] for r in results],
            'rank_test_score': [r['rank_test_score'] for r in results],
            'params': all_params
        }
        
        if self.verbose > 0:
            print(f"Best parameters: {self.best_params_}")
            print(f"Best cross-validation score: {self.best_score_:.6f}")
        
        return self
    
    def _prepare_X(self, X):
        """
        Apply scaling to input data if scaler was provided.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed.
            
        Returns
        -------
        X_transformed : array-like
            Transformed input data.
        """
        if self.scaler is not None and hasattr(self.scaler, 'transform'):
            return self.scaler.transform(X)
        return X
    
    def score(self, X, y=None):
        """
        Return the score on the given data, if the estimator has been refit.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
            
        y : array-like of shape (n_samples, n_output) or (n_samples,), default=None
            Target relative to X for classification or regression.
            
        Returns
        -------
        score : float
            The score of self.best_estimator_ on the given data.
        """
        if self.best_estimator_ is None:
            raise ValueError("This CustomRandomizedSearchCV instance was not fitted yet.")
        
        X_transformed = self._prepare_X(X)
        return self.best_estimator_.score(X_transformed, y)
    
    def predict(self, X):
        """
        Call predict on the estimator with the best found parameters.
        
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the underlying estimator.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        if self.best_estimator_ is None:
            raise ValueError("This CustomRandomizedSearchCV instance was not fitted yet.")
        
        X_transformed = self._prepare_X(X)
        return self.best_estimator_.predict(X_transformed)
    
    def predict_proba(self, X):
        """
        Call predict_proba on the estimator with the best found parameters.
        
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the underlying estimator.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        if self.best_estimator_ is None:
            raise ValueError("This CustomRandomizedSearchCV instance was not fitted yet.")
        
        X_transformed = self._prepare_X(X)
        return self.best_estimator_.predict_proba(X_transformed)
    
    def decision_function(self, X):
        """
        Call decision_function on the estimator with the best found parameters.
        
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the underlying estimator.
            
        Returns
        -------
        y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The decision function values.
        """
        if self.best_estimator_ is None:
            raise ValueError("This CustomRandomizedSearchCV instance was not fitted yet.")
        
        X_transformed = self._prepare_X(X)
        return self.best_estimator_.decision_function(X_transformed)
    
    def transform(self, X):
        """
        Call transform on the estimator with the best found parameters.
        
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the underlying estimator.
            
        Returns
        -------
        X_transformed : ndarray
            The transformed data.
        """
        if self.best_estimator_ is None:
            raise ValueError("This CustomRandomizedSearchCV instance was not fitted yet.")
        
        return self.best_estimator_.transform(X)
    
    def inverse_transform(self, X):
        """
        Call inverse_transform on the estimator with the best found parameters.
        
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the underlying estimator.
            
        Returns
        -------
        X_transformed : ndarray
            The inverse transformed data.
        """
        if self.best_estimator_ is None:
            raise ValueError("This CustomRandomizedSearchCV instance was not fitted yet.")
        
        return self.best_estimator_.inverse_transform(X)
