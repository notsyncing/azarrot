from abc import ABC, abstractmethod

from transformers import TextIteratorStreamer

from azarrot.common_data import GenerationRequest, GenerationStatistics, Model


class BaseBackend(ABC):
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def load_model(self, model: Model) -> None:
        pass

    @abstractmethod
    def unload_model(self, model_id: str) -> None:
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest) -> tuple[TextIteratorStreamer, GenerationStatistics]:
        pass
