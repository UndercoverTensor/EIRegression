# EmbeddedInterpreter.py

# coding=utf-8
import numpy as np
import pandas as pd
import os
import random  # Import random for selecting nearest regressors
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import torch

from .dsgd.DSClassifierMultiQ import DSClassifierMultiQ
from .dsgd.DSRule import DSRule
from .model_optimizer import ModelOptimizer
from .nanReplace import replace_nan_median
from .bucketing import bucketing


class EmbeddedInterpreter():
    """
    Implementation of Embedded interpreter regression based on DS model
    """

    def __init__(self, regressor=None, model_optimizer=None, model_preprocessor=None, n_buckets=3, bucketing_method="quantile",
                 reg_default_args={}, reg_hp_args={}, hp_grids=None, statistic=None, verbose=False, **cla_kwargs):
        """
        Initialize the EmbeddedInterpreter with new parameters for fine-tuning.
        :param hp_grids: List of hyperparameter grids for each bucket's regressor.
        :param optimizer_settings: Settings for the ModelOptimizer.
        :param verbose: If True, print bucket distributions during training and prediction.
        """
        self.n_buckets = n_buckets
        self.bins = []
        self.bucketing_method = bucketing_method
        self.y_dtype = None
        self.training_medians = None
        self.classifier = DSClassifierMultiQ(num_classes=n_buckets, **cla_kwargs)
        self.verbose = verbose  # Initialize the verbose flag
        if not statistic:
            self.regressors = [regressor(**reg_default_args) for _ in range(n_buckets)]
            self.hp_grids = hp_grids or [reg_hp_args for _ in range(n_buckets)]  # Default to empty grids if none provided
            self.optimizer = model_optimizer
            self.preprocessor = model_preprocessor
        self.statistic = statistic
        self.global_mean = None  # Initialize global_mean

    def get_nearest_fitted_regressors(self, i):
        """
        Finds the nearest fitted regressors to bucket i.

        :param i: Bucket index
        :return: List of nearest fitted regressors
        """
        fitted_indices = [j for j, reg in enumerate(self.regressors) if reg is not None]
        if not fitted_indices:
            return None  # No fitted regressors available

        # Calculate distances from bucket i to all fitted buckets
        distances = {j: abs(j - i) for j in fitted_indices}
        min_distance = min(distances.values())

        # Find all regressors at the minimum distance
        nearest_indices = [j for j, d in distances.items() if d == min_distance]

        # Randomly select one of the nearest regressors
        chosen_index = random.choice(nearest_indices)
        return self.regressors[chosen_index]

    def fit(self, X_train, y_train, **cla_kwargs):
        """
        Fits the model using the training data
        :param X: Features for training
        :param y: Labels of features
        :param cla_kwargs: Arguments for the DS classifier fitting
        """

        self.y_dtype = y_train.dtype
        if not self.bins:
            (buckets, bins) = bucketing(
                labels=y_train, features=X_train, bins=self.n_buckets, type=self.bucketing_method)
            self.bins = bins  # To test classifier later
        else:
            buckets = pd.cut(y_train, self.bins)

        # Print bucket distribution for training data if verbose is True
        if self.verbose:
            print(f"\n[Training] Bucket Distribution for n_buckets={self.n_buckets}:")
            if isinstance(buckets, pd.Series):
                bucket_counts = buckets.value_counts().sort_index()
                for interval, count in bucket_counts.items():
                    print(f"  {interval}: {count} samples")
            elif isinstance(buckets, np.ndarray):
                unique, counts = np.unique(buckets, return_counts=True)
                for bucket, count in zip(unique, counts):
                    print(f"  Bucket {bucket}: {count} samples")
            else:
                print("  [Warning] Unknown bucket type. Unable to print distribution.")

        self.classifier.fit(X_train, buckets, **cla_kwargs)
        pred_bucket = self.classifier.predict(X_train)

        # Print predicted bucket distribution after classification if verbose is True
        if self.verbose:
            print(f"\n[Training] Predicted Bucket Distribution after Classification for n_buckets={self.n_buckets}:")
            if isinstance(pred_bucket, pd.Series):
                bucket_counts = pred_bucket.value_counts().sort_index()
                for bucket, count in bucket_counts.items():
                    print(f"  Bucket {bucket}: {count} samples")
            elif isinstance(pred_bucket, np.ndarray):
                unique, counts = np.unique(pred_bucket, return_counts=True)
                for bucket, count in zip(unique, counts):
                    print(f"  Bucket {bucket}: {count} samples")
            else:
                print("  [Warning] Unknown predicted bucket type. Unable to print distribution.")

        self.training_medians = replace_nan_median(X_train)
        self.global_mean = np.mean(y_train)  # Compute global mean for fallback

        if not self.statistic:
            for i in range(self.n_buckets):
                bucket_X = X_train[pred_bucket == i]
                bucket_y = y_train[pred_bucket == i]
                if len(bucket_X) == 0:
                    if self.verbose:
                        print(f"Warning: Bucket {i} has no samples.")
                    self.regressors[i] = None  # Mark regressor as unfitted
                    continue  # Skip fitting for this bucket
                if self.hp_grids[i]:  # Check if there is a grid for the current bucket
                    regressor_pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('regressor', self.regressors[i])
                    ])

                    optimized_regressor = self.optimizer.optimize(
                        regressor_pipeline, self.hp_grids[i], bucket_X, bucket_y,
                        scoring='r2',  # Set scoring to a regression metric
                        cv=3  # Or another value, possibly passed through optimizer_settings
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
                    if self.verbose:
                        print(f"Warning: Bucket {i} has no samples.")
                    self.bucket_statistics.append(np.nan)
                    continue
                # Calculate and store statistics
                if self.statistic == 'median':
                    stat = np.median(bucket_y)
                elif self.statistic == 'mean':
                    stat = np.mean(bucket_y)
                self.bucket_statistics.append(stat)

        # Compute the coverage for each rule
        self.compute_rule_coverage(X_train)

    def predict(self, X_test, return_buckets=False):
        """
        Predict the classes for the feature vectors
        :param X: Feature vectors
        :param return_buckets: If true, it return buckets assigned to data
        :return: Value predicted for each feature vector. If return_buckets is true, it returns the buckets assigned to data
        """
        buck_pred = self.classifier.predict(X_test)
        y_pred = np.zeros(buck_pred.shape, dtype=self.y_dtype)

        replace_nan_median(X_test, self.training_medians)
        if not self.statistic:
            for i in range(self.n_buckets):
                if not (buck_pred == i).any():
                    continue
                if self.regressors[i] is not None:
                    y_pred[buck_pred == i] = self.regressors[i].predict(X_test[buck_pred == i])
                else:
                    # Handle empty regressor by finding the nearest fitted regressor
                    nearest_regressor = self.get_nearest_fitted_regressors(i)
                    if nearest_regressor is not None:
                        if self.verbose:
                            print(f"Bucket {i} has no fitted regressor. Using nearest fitted regressor.")
                        y_pred[buck_pred == i] = nearest_regressor.predict(X_test[buck_pred == i])
                    else:
                        # Assign a default value if no regressors are fitted
                        if self.verbose:
                            print(f"No fitted regressors available. Assigning global mean to bucket {i}.")
                        y_pred[buck_pred == i] = self.global_mean  # Assign global mean
        else:
            for i in range(self.n_buckets):
                if not (buck_pred == i).any():
                    continue
                # Use pre-calculated statistics instead of recalculating
                y_pred[buck_pred == i] = self.bucket_statistics[i]

        # Print bucket distribution for test data if verbose is True
        if self.verbose:
            print(f"\n[Test] Predicted Bucket Distribution for n_buckets={self.n_buckets}:")
            if isinstance(buck_pred, pd.Series):
                bucket_counts = buck_pred.value_counts().sort_index()
                for bucket, count in bucket_counts.items():
                    print(f"  Bucket {bucket}: {count} samples")
            elif isinstance(buck_pred, np.ndarray):
                unique, counts = np.unique(buck_pred, return_counts=True)
                for bucket, count in zip(unique, counts):
                    print(f"  Bucket {bucket}: {count} samples")
            else:
                print("  [Warning] Unknown predicted bucket type. Unable to print distribution.")

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
        Predict the score of belonging to all classes
        :param X: Feature vector
        :return: Class scores for each feature vector
        """
        return self.classifier.predict_proba(X)

    def predict_explain(self, X):
        """
        Predict the score of belonging to each class and give an explanation of that decision
        :param x: A single Feature vectors
        :return: Class scores for each feature vector and an explanation of the decision
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
        :param classes: Array of classes, by default shows all classes
        :param threshold: score minimum value considered to be contributive
        :return: A list containing the information about most important rules
        """
        return self.classifier.model.find_most_important_rules(classes=classes, threshold=threshold)

    def print_most_important_rules(self, classes=None, threshold=0.2):
        """
        Prints the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all classes
        :param threshold: score minimum value considered to be contributive
        :return:
        """
        self.classifier.model.print_most_important_rules(
            classes=classes, threshold=threshold)

    def assign_buckets(self, y_values):
        """
        Assigns buckets to the provided y_values based on the bins determined during training.
        :param y_values: Target values to assign buckets to
        :return: Assigned bucket indices
        """
        min_value, max_value = y_values.min(), y_values.max()
        extended_bins = [min(min_value, self.bins[0])] + list(self.bins[1:-1]) + [max(max_value, self.bins[-1])]
        buckets = pd.cut(y_values, bins=extended_bins, labels=False, include_lowest=True)

        # Handle NaN assignments if any
        if np.isnan(buckets).any():
            print("Warning: Some samples could not be assigned to a bucket. Assigning them to the nearest bucket.")
            buckets = buckets.fillna(method='ffill').fillna(method='bfill')

        return buckets

    def evaluate_classifier(self, X_test, y_test):
        """
        Evaluates the classifier using the test data
        :param X_test: Features for test
        :param y_test: Labels of features
        :return: Accuracy score, F1 macro score, and confusion matrix
        """
        y_test_buckets = self.assign_buckets(y_test)
        y_pred = self.classifier.predict(X_test)
        f1_macro = f1_score(y_test_buckets, y_pred, average='macro')
        acc = accuracy_score(y_test_buckets, y_pred)
        cm = confusion_matrix(y_test_buckets, y_pred)
        return acc, f1_macro, cm

    def rules_to_txt(self, filename, classes=None, threshold=0.2, results={}):
        """
        Write the most contributive rules for the classes specified in an output file
        :param filename: Output file name
        :param classes: Array of classes, by default shows all classes
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

    def compute_similarity(self, x, y_true, threshold=0.2, include_rule_coverage=False):
        """
        Compute the similarity between the activated rules for input x and the most important rules for the actual class y_true.
        Similarity is defined as the proportion of the intersection between these two rule sets over the union.
        When include_rule_coverage is True, computes Weighted Jaccard Similarity using rule coverage.
        :param x: Input sample (feature vector)
        :param y_true: Actual class label (integer index)
        :param threshold: Threshold for important rules
        :param include_rule_coverage: Flag to include rule coverage in similarity computation
        :return: Similarity score
        """
        # Get the activated rules for x
        rules, preds = self.classifier.model.get_rules_by_instance(x)
        # 'preds' are the activated rules (DSRule instances)
        activated_rules_set = set(preds)

        # Get the most important rules for y_true
        important_rules_dict = self.classifier.model.find_most_important_rules(classes=[y_true], threshold=threshold)
        important_rules_list = important_rules_dict.get(y_true, [])
        # Build a mapping from rule captions to DSRule objects
        rule_caption_to_rule = {str(rule): rule for rule in self.classifier.model.preds}
        # Get the DSRule objects for the important rules
        important_rules_set = set()
        for rule_info in important_rules_list:
            rule_caption = rule_info[2]  # Rule caption
            rule_obj = rule_caption_to_rule.get(rule_caption)
            if rule_obj:
                important_rules_set.add(rule_obj)

        # Compute the intersection and union of the rule sets
        intersection = activated_rules_set & important_rules_set
        union = activated_rules_set  # Only activated_rules_set is considered for union

        if include_rule_coverage:
            # Compute the Weighted Jaccard Similarity
            # Sum the coverage of overlapping rules (intersection)
            overlap_coverage = sum(rule.coverage for rule in intersection)
            # Sum the coverage of all rules in the union
            total_coverage = sum(rule.coverage for rule in union)
            # Compute similarity, handling division by zero
            if total_coverage > 0:
                similarity = overlap_coverage / total_coverage
            else:
                similarity = 0.0
        else:
            # Original similarity calculation (unweighted)
            if len(important_rules_set) == 0:
                similarity = 0.0
            else:
                similarity = len(intersection) / len(important_rules_set)

        return similarity

    def compute_rule_coverage(self, X_data):
        """
        Computes and updates the coverage for each rule in the classifier model.
        :param X_data: Training data as a NumPy array.
        """
        # Convert X_data to DataFrame
        X_df = pd.DataFrame(X_data)

        # Iterate over all rules and compute coverage
        for rule in self.classifier.model.preds:
            # Apply rule to X_df to get a boolean array indicating where the rule applies
            # Compute coverage as the sum of True values in the boolean array
            rule.coverage = X_df.apply(rule, axis=1).sum()