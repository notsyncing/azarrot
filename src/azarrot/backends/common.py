from dataclasses import asdict

from torch import Tensor
from transformers import AutoTokenizer, TextIteratorStreamer

from azarrot.common_data import GenerationMessage, GenerationStatistics


class CountedTextIteratorStreamer(TextIteratorStreamer):
    _generation_statistics: GenerationStatistics

    def __init__(  # type: ignore[no-untyped-def]
        self,
        tokenizer: "AutoTokenizer",
        generation_statistics: GenerationStatistics,
        skip_prompt: bool = False,
        timeout: float | None = None,
        **decode_kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)

        self._generation_statistics = generation_statistics

    def put(self, value: Tensor) -> None:
        if len(value.shape) > 1:
            value = value[0]

        self._generation_statistics.total_tokens += len(value)

        super().put(value)


def to_transformers_chat_messages( messages: list[GenerationMessage]) -> list[dict[str, str]]:
    return [asdict(m) for m in messages]
