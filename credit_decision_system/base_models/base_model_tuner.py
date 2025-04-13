from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import pandas as pd
from copy import deepcopy

def base_model_tuner(models, X_train, y_train, scoring="roc_auc", cv=5, n_iter=15, n_jobs=-1):
    """
    Two-stage model tuning (RandomizedSearchCV → GridSearchCV).

    Args:
        models (dict): Dictionary of sklearn pipelines.
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        scoring (str): Scoring metric (e.g., "roc_auc").
        cv (int): Cross-validation folds.
        n_iter (int): Number of random search iterations.
        n_jobs (int): Number of jobs for parallel processing.

    Returns:
        tuned_models (dict): Model name → best tuned pipeline.
        best_params (dict): Model name → best param dict.
    """
    tuned_models = {}
    best_params = {}

    # Flatten y_train if it's a DataFrame with 1 column
    if isinstance(y_train, pd.DataFrame):
        if y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0].values
        else:
            y_train = y_train.values
    elif isinstance(y_train, pd.Series):
        y_train = y_train.values

    # Parameters that must be integers
    int_only_params = {
        "classifier__n_estimators",
        "classifier__max_depth",
        "classifier__min_samples_split"
    }

    # Search space
    param_distributions = {
        "logistic_regression": {
            "classifier__C": np.logspace(-3, 3, 100),
            "classifier__solver": ["liblinear", "lbfgs"]
        },
        "calibrated_logistic_regression": {
            "classifier__estimator__C": np.logspace(-3, 3, 100),
            "classifier__estimator__solver": ["liblinear", "lbfgs"]
        },
        "xgboost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": np.linspace(0.01, 0.3, 20),
            "classifier__subsample": np.linspace(0.6, 1.0, 5)
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [5, 10, 15],
            "classifier__min_samples_split": [2, 5, 10]
        }
    }

    for name, model in models.items():
        print(f"\n🔍 Randomized Search for {name}")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42
        )
        random_search.fit(X_train, y_train)
        best_random_params = random_search.best_params_
        print(f"➡️ Best random params: {best_random_params}")

        # Refine around best values
        refined_grid = {}
        for param, val in best_random_params.items():
            if isinstance(val, (float, int)):
                delta = val * 0.5 if val != 0 else 1
                candidates = [val - delta, val, val + delta]

                if param in int_only_params:
                    refined = sorted(set(
                        max(2, int(round(v)))  # Clamp minimum int value to 2
                        for v in candidates
                        if int(round(v)) > 0
                    ))
                    refined_grid[param] = refined
                else:
                    refined_grid[param] = sorted(set(candidates))
            else:
                refined_grid[param] = [val]

        print(f"\n🎯 Grid Search for {name}")
        grid_search = GridSearchCV(
            estimator=deepcopy(model),
            param_grid=refined_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        tuned_models[name] = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
        print(f"✅ Best grid params: {grid_search.best_params_}")

    return tuned_models, best_params
