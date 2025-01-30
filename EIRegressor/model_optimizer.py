import random

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from collections import Counter


class ModelOptimizer:
    def __init__(self, search_method='grid', n_iter=10):
        """
        Initialize the ModelOptimizer with a specified search method and number of iterations for random search.
        :param search_method: Method for hyperparameter tuning ('grid' or 'random').
        :param n_iter: Number of iterations for random search.
        """
        self.search_method = search_method
        self.n_iter = n_iter

    def optimize(self, pipeline, hp_grid, X_train, y_train, scoring='accuracy', cv=5, lower_search_bound=5):
        """
        Perform hyperparameter tuning on the given pipeline.
        :param pipeline: Pipeline including preprocessing and the model.
        :param hp_grid: Hyperparameter grid for the model.
        :param X_train: Training data features.
        :param y_train: Training data target.
        :param scoring: Scoring metric for optimization.
        :param cv: Number of folds for cross-validation.
        :return: The pipeline with the best found parameters.
        """
        # Adjust the number of splits based on the number of samples
        n_splits = min(cv, len(y_train))

        if len(y_train) <= lower_search_bound:
            random_params = {k: random.choice(v) for k, v in hp_grid.items()}
            pipeline.set_params(**random_params)
            pipeline.fit(X_train, y_train)
            return pipeline

        # Check for sufficient samples per class for StratifiedKFold
        if "classifier" in pipeline.named_steps:
            class_counts = Counter(y_train)
            if any(count < n_splits for count in class_counts.values()):
                raise ValueError(f"One or more classes have fewer members ({min(class_counts.values())}) than n_splits={n_splits}.")

            cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True)
        else:
            cv_strategy = KFold(n_splits=n_splits, shuffle=True)

        if self.search_method == 'grid':
            search = GridSearchCV(pipeline, hp_grid, cv=cv_strategy, scoring=scoring, error_score='raise')
        elif self.search_method == 'random':
            search = RandomizedSearchCV(pipeline, hp_grid, n_iter=self.n_iter, cv=cv_strategy, scoring=scoring,
                                        random_state=42)
        else:
            raise ValueError("Invalid search method. Choose 'grid', 'random' or 'stratified'.")

        search.fit(X_train, y_train)
        return search.best_estimator_
