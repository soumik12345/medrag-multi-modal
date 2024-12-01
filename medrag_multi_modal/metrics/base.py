from typing import Optional

import numpy as np
import weave


class BaseAccuracyMetric(weave.Scorer):
    """
    BaseAccuracyMetric is a class that extends the
    [`weave.Scorer`](https://weave-docs.wandb.ai/guides/evaluation/scorers#class-based-scorers)
    to provide a comprehensive evaluation of accuracy metrics for a given set of score rows.

    This class is designed to process a list of score rows, each containing a
    'correct' key that indicates whether a particular prediction was correct.
    The `summarize` method calculates various statistical measures and metrics
    based on this data, including:

    - True and false counts: The number of true and false predictions.
    - True and false fractions: The proportion of true and false predictions.
    - Standard error: The standard error of the mean for the true predictions.
    - Precision: The ratio of true positive predictions to the total number of
      positive predictions.
    - Recall: The ratio of true positive predictions to the total number of
      actual positives.
    - F1 Score: The harmonic mean of precision and recall, providing a balance
      between the two metrics.

    The `summarize` method returns a dictionary containing these metrics,
    allowing for a detailed analysis of the model's performance.

    Methods:
        summarize(score_rows: list) -> Optional[dict]:
            Processes the input score rows to compute and return a dictionary
            of accuracy metrics.
    """

    @weave.op()
    def summarize(self, score_rows: list) -> Optional[dict]:
        """
        Summarizes the accuracy metrics from a list of score rows.

        This method processes a list of score rows, each containing a 'correct' key
        that indicates whether a particular prediction was correct. It calculates
        various statistical measures and metrics based on this data, including:

        - True and false counts: The number of true and false predictions.
        - True and false fractions: The proportion of true and false predictions.
        - Standard error: The standard error of the mean for the true predictions.
        - Precision: The ratio of true positive predictions to the total number of
          positive predictions.
        - Recall: The ratio of true positive predictions to the total number of
          actual positives.
        - F1 Score: The harmonic mean of precision and recall, providing a balance
          between the two metrics.

        The method returns a dictionary containing these metrics, allowing for a
        detailed analysis of the model's performance.

        Args:
            score_rows (list): A list of dictionaries, each containing a 'correct'
                key with a boolean value indicating the correctness of a prediction.

        Returns:
            Optional[dict]: A dictionary containing the calculated accuracy metrics,
                or None if the input list is empty.
        """
        valid_data = [
            x.get("correct") for x in score_rows if x.get("correct") is not None
        ]
        count_true = list(valid_data).count(True)
        int_data = [int(x) for x in valid_data]

        sample_mean = np.mean(int_data) if int_data else 0
        sample_variance = np.var(int_data) if int_data else 0
        sample_error = np.sqrt(sample_variance / len(int_data)) if int_data else 0

        # Calculate precision, recall, and F1 score
        true_positives = count_true
        false_positives = len(valid_data) - count_true
        false_negatives = len(score_rows) - len(valid_data)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "correct": {
                "true_count": count_true,
                "false_count": len(score_rows) - count_true,
                "true_fraction": float(sample_mean),
                "false_fraction": 1.0 - float(sample_mean),
                "stderr": float(sample_error),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        }
