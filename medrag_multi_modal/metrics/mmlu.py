import weave

from medrag_multi_modal.assistant.schema import MedQAResponse
from medrag_multi_modal.metrics.base import BaseAccuracyMetric


class MMLUOptionAccuracy(BaseAccuracyMetric):
    """
    MMLUOptionAccuracy is a metric class that inherits from `BaseAccuracyMetric`.

    This class is designed to evaluate the accuracy of a multiple-choice question
    response by comparing the provided answer with the correct answer from the
    given options. It uses the MedQAResponse schema to extract the response
    and checks if it matches the correct answer.

    Methods:
    --------
    score(output: MedQAResponse, options: list[str], answer: str) -> dict:
        Compares the provided answer with the correct answer and returns a
        dictionary indicating whether the answer is correct.
    """

    @weave.op()
    def score(self, output: MedQAResponse, options: list[str], answer: str):
        return {"correct": options[answer] == output.response.answer}
