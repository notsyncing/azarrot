from typing import Any

from torch import Tensor
from transformers import AutoTokenizer, TextIteratorStreamer

from azarrot.common_data import GenerationMessage, GenerationStatistics, TextGenerationMessageContent


class CountedTextIteratorStreamer(TextIteratorStreamer):
    _generation_statistics: GenerationStatistics
    _failed = False

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

        self._generation_statistics.completion_tokens += len(value)

        super().put(value)

    def set_failed(self) -> None:
        self._failed = True
        self.text_queue.put(self.stop_signal)

    def __next__(self) -> Any:      # noqa: D105
        if self._failed:
            raise ValueError("TextStreamer is forced to fail")

        return super().__next__()


def to_transformers_chat_messages(messages: list[GenerationMessage]) -> list[dict[str, str]]:
    c = []

    for m in messages:
        for mc in m.content:
            content: str

            if isinstance(mc, TextGenerationMessageContent):
                content = mc.text
            else:
                raise ValueError("Invalid generation message for chat: %s", str(mc))

            c.append({"role": m.role, "content": content})

    return c
