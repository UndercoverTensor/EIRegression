# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from .dsgd.DSClassifierMultiQ import DSClassifierMultiQ
from .dsgd.DSRule import DSRule
from .model_optimizer import ModelOptimizer
from .nanReplace import replace_nan_median
from .bucketing import bucketing


class EmbeddedInterpreterEnsemble():
    """
    Implementation of Embedded interpreter regression based on DS model
    """

    def __init__(self, regressor=None, model_optimizer=None, model_preprocessor=None, n_buckets=3, bucketing_method="quantile",
                 reg_default_args={}, reg_hp_args={}, hp_grids=None, statistic=None, **cla_kwargs):
        """
        Initialize the EmbeddedInterpreter with new parameters for fine-tuning.
        :param hp_grids: List of hyperparameter grids for each bucket's regressor.
        :param optimizer_settings: Settings for the ModelOptimizer.
        """
        self.n_buckets = n_buckets
        self.bins = []
        self.bucketing_method = bucketing_method
        self.y_dtype = None
        self.training_medians = None
        self.classifier = DSClassifierMultiQ(num_classes=n_buckets, **cla_kwargs)
        if not statistic:
            self.regressors = [regressor(**reg_default_args) for _ in range(n_buckets)]
            self.hp_grids = hp_grids or [reg_hp_args for _ in
                                         range(n_buckets)]  # Default to empty grids if none provided
            self.optimizer = model_optimizer
            self.preprocessor = model_preprocessor
        self.statistic = statistic


    def fit(self, X_train, y_train, **cla_kwargs):
        """
        Fits the model using the training data
        :param X: Features for training
        :param y: Labels of features
        :param cla_kwargs: Arguments for the DS classifier fitting
        """

        self.y_dtype = y_train.dtype
        if self.bins == []:
            (buckets, bins) = bucketing(
                labels=y_train, features=X_train, bins=self.n_buckets, type=self.bucketing_method)
            self.bins = bins  # To test classifier later
        else:
            buckets = pd.cut(y_train, self.bins)
        self.classifier.fit(X_train, buckets, **cla_kwargs)
        pred_bucket = self.classifier.predict(X_train)
        self.training_medians = replace_nan_median(X_train)
        if not self.statistic:
            for i in range(self.n_buckets):
                bucket_X = X_train[pred_bucket == i]
                bucket_y = y_train[pred_bucket == i]
                if len(bucket_X) == 0:
                    bucket_X = X_train[buckets == i]
                    bucket_y = y_train[buckets == i]
                if self.hp_grids[i]:  # Check if there is a grid for the current bucket
                    regressor_pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('regressor', self.regressors[i])
                    ])

                    optimized_regressor = self.optimizer.optimize(
                        regressor_pipeline, self.hp_grids[i], bucket_X, bucket_y,
                        scoring='r2',  # Set scoring to a regression metric
                        cv=5  # Or another value, possibly passed through optimizer_settings
                    )
                    self.regressors[i] = optimized_regressor
                else:
                    self.regressors[i].fit(bucket_X, bucket_y)
        else:
            self.bucket_statistics = []
            for i in range(self.n_buckets):
                bucket_X = X_train[pred_bucket == i]
                bucket_y = y_train[pred_bucket == i]
                if len(bucket_X) == 0:
                    bucket_y = y_train[buckets == i]

                # Calculate and store statistics
                if self.statistic == 'median':
                    stat = np.median(bucket_y)
                elif self.statistic == 'mean':
                    stat = np.mean(bucket_y)
                self.bucket_statistics.append(stat)


    def predict(self, X_test, return_buckets=False):
        """
        Predict the classes for the feature vectors
        :param X: Feature vectors
        :param return_buckets: If true, it return buckets assigned to data
        :return: Value predicted for each feature vector. If return_buckets is true, it returns the buckets assigned to data
        """
        buck_pred = self.classifier.predict(X_test, is_ensemble=True)
        y_pred = np.zeros(buck_pred.shape, dtype=self.y_dtype)

        replace_nan_median(X_test, self.training_medians)
        if not self.statistic:
            for i in range(self.n_buckets):
                if not (buck_pred == i).any():
                    continue
                y_pred[buck_pred == i] = self.regressors[i].predict(X_test[buck_pred == i])
        else:
            for i in range(self.n_buckets):
                if not (buck_pred == i).any():
                    continue
                # Use pre-calculated statistics instead of recalculating
                y_pred[buck_pred == i] = self.bucket_statistics[i]

        if return_buckets:
            return buck_pred, y_pred
        return y_pred

    def calculate_bucket_statistics(self, bucket_y, stat_type='median'):
        """
        Calculates median or mean of the targets within a bucket based on the specified statistic type.
        :param bucket_y: Target values for the bucket
        :param stat_type: Type of statistic to compute ('median' or 'mean')
        :return: Computed statistic of bucket_y
        """
        if stat_type == 'median':
            return np.median(bucket_y)
        elif stat_type == 'mean':
            return np.mean(bucket_y)
        else:
            raise ValueError("Invalid stat_type. Use 'median' or 'mean'.")

    def get_bins(self):
        """
        Returns the bins used for bucketing the data
        """
        return self.bins

    def set_bins(self, bins):
        """
        Sets the bins used for bucketing the data
        :param bins: Array of bins
        """
        self.bins = bins

    def predict_proba(self, X):
        """
        Predict the score of belogning to all classes
        :param X: Feature vector
        :return: Class scores for each feature vector
        """
        return self.classifier.predict_proba(X)

    def predict_explain(self, X):
        """
        Predict the score of belogning to each class and give an explanation of that decision
        :param x: A single Feature vectors
        :return: Class scores for each feature vector and a explanation of the decision
        """
        return self.classifier.predict_explain(X)

    def add_rule(self, rule, caption="", m_sing=None, m_uncert=None):
        """
        Adds a rule to the model. If no masses are provided, random masses will be used.
        :param rule: lambda or callable, used as the predicate of the rule
        :param caption: Description of the rule
        :param m_sing: [optional] masses for singletons
        :param m_uncert: [optional] mass for uncertainty
        """
        self.classifier.model.add_rule(DSRule(rule, caption), m_sing, m_uncert)

    def find_most_important_rules(self, classes=None, threshold=0.2):
        """
        Shows the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :return: A list containing the information about most important rules
        """
        return self.classifier.model.find_most_important_rules(classes=classes, threshold=threshold)

    def print_most_important_rules(self, classes=None, threshold=0.2):
        """
        Prints the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :return:
        """
        self.classifier.model.print_most_important_rules(
            classes=classes, threshold=threshold)

    def evaluate_classifier(self, X_test, y_test):
        """
            Evaluates the classifier using the test data
            :param X_test: Features for test
            :param y_test: Labels of features
            :return: F1 score macro and accuracy score
        """
        bins = []
        if len(self.bins) == 3:
            bins = np.append(int(min(y_test) - 1), np.append(self.bins[1], int(max(y_test) + 1)))
        else:
            bins = np.append(
                min(y_test) - 1, np.append(self.bins[1:-1], max(y_test) + 1))
        y_test = pd.cut(y_test, bins, labels=False)
        y_pred = self.classifier.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return acc, f1_macro, cm

    def rules_to_txt(self, filename, classes=None, threshold=0.2, results={}):
        """
        Write the most contributive rules for the classes specified in an output file
        :param filename: Output file name
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :param results: Dictionary with the results to print in txt 
        :return:
        """
        rules = self.classifier.model.find_most_important_rules(
            classes=classes, threshold=threshold)
        with open(filename, 'w') as file:
            for r in results:
                file.write(r + ": " + str(results[r]) + "\n\n")
            file.write(f"Most important rules\n-----------------------------\n")
            for key, rules_list in rules.items():
                file.write(f"\n---{key}---\n")
                for rule in rules_list:
                    file.write(
                        f"rule{rule[1]}: {rule[2]}\nprobabilities_array:{rule[4]}\n\n")

    def get_top_uncertainties(self, classes=None, threshold=0.2, top=5):
        rules = self.classifier.model.find_most_important_rules(
            classes=classes, threshold=threshold)
        results = {}

        for key, rules_list in rules.items():
            all_scores = []

            for rule in rules_list:
                uncertainty_score = rule[4][-1]
                all_scores.append(float(uncertainty_score))

            if len(all_scores) < top:
                top_scores = sorted(all_scores, reverse=False)
            else:
                top_scores = sorted(all_scores, reverse=False)[:top]

            results[f"class{key}"] = top_scores

        return results