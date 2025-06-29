from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, override

import torch
from transformers.generation.streamers import TextIteratorStreamer

from azarrot.common_data import (
    EmptyMessageChunk,
    GeneratedMessageChunk,
    GenerationStatistics,
    ModelQuirks,
    ModelToolCallConfig,
    TextGeneratedMessageChunk,
    ToolCallGeneratedMessageChunk,
)

if TYPE_CHECKING:
    from transformers import AutoTokenizer


EmbeddingsGenerationResult = list[list[float]]


class StopGenerationError(Exception):
    pass


class CustomTextIteratorStreamer(TextIteratorStreamer):
    _generation_statistics: GenerationStatistics
    _failed = False
    _first_token = True
    _model_quirks: ModelQuirks | None = None
    _output_buffer: str = ""
    _full_text = False
    _object_queue: Queue[Any]
    _current_ended = False
    _batch_mode = False
    _cut_text = False

    def __init__(  # type: ignore[no-untyped-def]
        self,
        tokenizer: "AutoTokenizer",
        generation_statistics: GenerationStatistics,
        skip_prompt: bool = False,
        timeout: float | None = None,
        model_quirks: ModelQuirks | None = None,
        **decode_kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)

        self._object_queue = Queue()
        self._generation_statistics = generation_statistics
        self._model_quirks = model_quirks

    def get_generation_statistics(self) -> GenerationStatistics:
        return self._generation_statistics

    def get_model_quirks(self) -> ModelQuirks | None:
        return self._model_quirks

    def set_batch_mode(self, batch_mode: bool = True) -> None:
        self._batch_mode = batch_mode

    def get_completion_tokens(self) -> int:
        return self._generation_statistics.completion_tokens

    def put(self, value: torch.Tensor) -> None:
        if self._cut_text:
            return

        if len(value.shape) > 1:
            value = value[0]

        if not self.next_tokens_are_prompt:
            self._generation_statistics.completion_tokens += len(value)

            if self._first_token:
                self._generation_statistics.first_token_time = datetime.now()
                self._first_token = False

        super().put(value)

    def fetch_object(self) -> Any:
        return self._object_queue.get()

    def put_object(self, value: Any) -> None:
        self._object_queue.put(value)

    def __check_and_strip_stop_before_strings(self) -> tuple[bool, int]:
        if self._model_quirks is None:
            raise ValueError("Called with None model_quirks!")

        if self._model_quirks.additional_stop_before_strings is None:
            raise ValueError("Called with None additional_stop_before_strings!")

        for stop_str in self._model_quirks.additional_stop_before_strings:
            start_index = self._output_buffer.find(stop_str)

            if start_index >= 0:
                return True, start_index

        return False, -1

    def __on_stream_end(self) -> None:
        pass

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        if self._cut_text:
            if stream_end:
                super().on_finalized_text("", stream_end)

            return

        if self._model_quirks is not None:
            if self._model_quirks.output_buffering_length > 0:
                self._output_buffer += text

                if self._model_quirks.full_text_indicators is not None and not self._full_text:
                    for full_text_indicator in self._model_quirks.full_text_indicators:
                        if self._output_buffer.find(full_text_indicator) >= 0:
                            self._full_text = True

                if self._model_quirks.additional_stop_before_strings is not None:
                    should_stop, cut_index = self.__check_and_strip_stop_before_strings()

                    if should_stop:
                        self._cut_text = True
                        self.__on_stream_end()
                        new_text = self._output_buffer[:cut_index]
                        self._output_buffer = ""

                        if not self._batch_mode:
                            super().on_finalized_text(new_text, stream_end=True)
                            raise StopGenerationError

                        super().on_finalized_text(new_text, stream_end=False)
                        return

                if not self._full_text and len(self._output_buffer) > self._model_quirks.output_buffering_length * 2:
                    output_text = self._output_buffer[: self._model_quirks.output_buffering_length]
                    self._output_buffer = self._output_buffer[self._model_quirks.output_buffering_length :]
                    super().on_finalized_text(output_text, stream_end=False)

                if stream_end:
                    self.__on_stream_end()
                    super().on_finalized_text(self._output_buffer, stream_end)
                    self._output_buffer = ""

                return

        super().on_finalized_text(text, stream_end)

    def set_failed(self) -> None:
        if not self._failed:
            self._failed = True
            self.text_queue.put(self.stop_signal)

    def __next__(self) -> Any:
        if self._failed:
            raise ValueError("TextStreamer is forced to fail")

        if not self._current_ended:
            try:
                return super().__next__()
            except (StopIteration, Empty):
                self._current_ended = True

        if self._current_ended:
            raise StopIteration

        raise ValueError("TextStreamer has invalid end state")

    def update_start_generation_time(self, time: datetime) -> None:
        self._generation_statistics.start_time = time


class BatchedCustomTextIteratorStreamer(CustomTextIteratorStreamer):
    _inner_streamers: list[CustomTextIteratorStreamer]

    def __init__(self, inner_streamers: list[CustomTextIteratorStreamer]) -> None:
        self._inner_streamers = inner_streamers
        first_streamer = inner_streamers[0]

        for s in self._inner_streamers:
            s.set_batch_mode()

        super().__init__(
            first_streamer.tokenizer,
            first_streamer.get_generation_statistics(),
            first_streamer.skip_prompt,
            first_streamer.timeout,
            first_streamer.get_model_quirks(),
            **first_streamer.decode_kwargs,
        )

    def get_batched_streamers(self) -> list[CustomTextIteratorStreamer]:
        return self._inner_streamers

    def put(self, value: torch.Tensor) -> None:
        size = value.shape[0]

        for i in range(size):
            inner_value = value.index_select(0, torch.IntTensor([i]))
            self._inner_streamers[i].put(inner_value)

    def end(self) -> None:
        for s in self._inner_streamers:
            s.end()

    def set_failed(self) -> None:
        for streamer in self._inner_streamers:
            streamer.set_failed()

    def update_start_generation_time(self, time: datetime) -> None:
        for s in self._inner_streamers:
            s.update_start_generation_time(time)


GM = TypeVar("GM", bound="GenerationMethods")
R = TypeVar("R")


class GenerationMethods[GM: "GenerationMethods", R](ABC):
    def is_batching_supported(self) -> bool:
        return True

    @abstractmethod
    def merge_into_batch(self, others: list[GM]) -> None:
        pass

    @abstractmethod
    def generate(self) -> tuple[bool, list[R]]:
        pass

    def update_start_generation_time(self, time: datetime) -> None:  # noqa: B027
        pass

    def on_execution_failed(self) -> None:  # noqa: B027
        pass


class CompletionChunkStreamer(Iterator[GeneratedMessageChunk]):
    _text_streamer: TextIteratorStreamer
    _tc_config: ModelToolCallConfig | None = None
    _state: Literal["text", "tool_call"]
    _tool_call_counter: int = 0
    _tool_call_extracting_state: dict[str, Any]
    _queue: Queue

    def __init__(
        self,
        text_streamer: TextIteratorStreamer,
        model_tool_call_config: ModelToolCallConfig | None = None,
    ) -> None:
        self._text_streamer = text_streamer
        self._tc_config = model_tool_call_config
        self._state = "text"
        self._tool_call_extracting_state = {}
        self._queue = Queue()

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> GeneratedMessageChunk:  # noqa: PLR0915
        if not self._queue.empty():
            return self._queue.get()

        try:
            text: str = self._text_streamer.__next__()
        except StopIteration as e:
            raise StopIteration from e

        tcsi: str | None
        tcei: str | None

        if self._tc_config is not None:
            tcsi = self._tc_config.tool_call_start_indicator
            tcei = self._tc_config.tool_call_stop_indicator
        else:
            tcsi = None
            tcei = None

        while len(text) > 0:
            if self._state == "text":
                if tcsi is not None:
                    tcsi_start = text.find(tcsi)

                    if tcsi_start >= 0:
                        remaining_text = text[:tcsi_start]

                        if len(remaining_text) > 0:
                            self._queue.put(TextGeneratedMessageChunk(content=remaining_text))

                        text = text[tcsi_start:]
                        self._state = "tool_call"

                if self._state == "text" and len(text) > 0:
                    self._queue.put(TextGeneratedMessageChunk(content=text))
                    text = ""

            if self._state == "tool_call":
                assert tcsi is not None
                assert tcei is not None
                assert self._tc_config is not None

                tool_call_start_pos = text.find(tcsi)

                if tool_call_start_pos == 0:
                    tool_call_content = text[tool_call_start_pos + len(tcsi) :]
                elif tool_call_start_pos < 0:
                    tool_call_content = text
                else:
                    raise ValueError(f"Invalid tool call start pos {tool_call_start_pos} in text {text}")

                if self._tc_config.tool_call_info_extracting_method is None:
                    raise ValueError("No tool_call_info_extracting_method is configured for current model!")

                tool_call_end_pos = tool_call_content.find(tcei)

                if tool_call_end_pos >= 0:
                    text = tool_call_content[tool_call_end_pos + len(tcei) :]
                    tool_call_content = tool_call_content[:tool_call_end_pos]
                else:
                    text = ""

                tool_call_extracted_info = self._tc_config.tool_call_info_extracting_method(
                    self._tc_config, self._tool_call_extracting_state, tool_call_content
                )

                if tool_call_extracted_info.name_completed:
                    self._queue.put(
                        ToolCallGeneratedMessageChunk(
                            index=self._tool_call_counter,
                            name=tool_call_extracted_info.name,
                            arguments=tool_call_extracted_info.arguments or "",
                        )
                    )

                if tool_call_end_pos >= 0:
                    self._tool_call_counter += 1
                    self._tool_call_extracting_state = {}
                    self._state = "text"

        if not self._queue.empty():
            return self._queue.get()
        else:
            return EmptyMessageChunk()
