import weave

from medrag_multi_modal.assistant.schema import MedQAResponse
from medrag_multi_modal.metrics.base import BaseAccuracyMetric


class MMLUOptionAccuracy(BaseAccuracyMetric):
    @weave.op()
    def score(self, output: MedQAResponse, options: list[str], answer: str):
        return {"correct": options[answer] == output.response.answer}
