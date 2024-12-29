from abc import ABC, abstractmethod


class Frontend(ABC):
    @abstractmethod
    def id(self) -> str:
        pass
