import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from typing import Any, Generic, TypeVar, cast

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedModel, TextIteratorStreamer, set_seed

from azarrot.common_data import GenerationMessage, GenerationStatistics, ModelQuirks, TextGenerationMessageContent

CTIS_HAS_OBJECT = "__ctis_has_object__"
CTIS_DELEGATE_TO_NEXT = "__ctis_delegate_to_next__"


@dataclass
class GenerationHandlers:
    full_text_handler: Callable[["CustomTextIteratorStreamer", str], tuple[bool, str | None]] | None = None


class StopGenerationError(Exception):
    pass


class CustomTextIteratorStreamer(TextIteratorStreamer):
    _generation_statistics: GenerationStatistics
    _failed = False
    _first_token = True
    _model_quirks: ModelQuirks | None = None
    _output_buffer: str = ""
    _full_text = False
    _generation_handlers: GenerationHandlers | None = None
    _object_queue: Queue[Any]
    _current_ended = False
    _next_streamer: "CustomTextIteratorStreamer | None" = None
    _batch_mode = False
    _cut_text = False

    def __init__(  # type: ignore[no-untyped-def]
        self,
        tokenizer: "AutoTokenizer",
        generation_statistics: GenerationStatistics,
        skip_prompt: bool = False,
        timeout: float | None = None,
        model_quirks: ModelQuirks | None = None,
        generation_handlers: GenerationHandlers | None = None,
        **decode_kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)

        self._object_queue = Queue()
        self._generation_statistics = generation_statistics
        self._model_quirks = model_quirks
        self._generation_handlers = generation_handlers

    def get_generation_statistics(self) -> GenerationStatistics:
        return self._generation_statistics

    def get_model_quirks(self) -> ModelQuirks | None:
        return self._model_quirks

    def get_generation_handlers(self) -> GenerationHandlers | None:
        return self._generation_handlers

    def set_next_streamer(self, next_streamer: "CustomTextIteratorStreamer") -> None:
        self._next_streamer = next_streamer

    def set_batch_mode(self, batch_mode: bool = True) -> None:
        self._batch_mode = batch_mode

    def get_completion_tokens(self) -> int:
        return self._generation_statistics.completion_tokens

    def put(self, value: Tensor) -> None:
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
        if self._full_text and self._generation_handlers is not None:
            if self._generation_handlers.full_text_handler is not None:
                should_override, new_text = self._generation_handlers.full_text_handler(self, self._output_buffer)

                if should_override:
                    if new_text == CTIS_DELEGATE_TO_NEXT:
                        if self._next_streamer is None:
                            raise ValueError(
                                "You have specified delegate to next TextStreamer, but no TextStreamer was given!"
                            )

                        new_text = ""

                    self._output_buffer = new_text if new_text is not None else ""

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

    def __next__(self) -> Any:  # noqa: D105
        if self._failed:
            raise ValueError("TextStreamer is forced to fail")

        if not self._current_ended:
            try:
                return super().__next__()
            except StopIteration:
                self._current_ended = True

        if self._current_ended:
            if self._next_streamer is not None:
                try:
                    return self._next_streamer.__next__()
                except StopIteration as e:
                    self._generation_statistics.completion_tokens += self._next_streamer.get_completion_tokens()
                    raise StopIteration from e
            else:
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
            first_streamer.get_generation_handlers(),
            **first_streamer.decode_kwargs,
        )

    def put(self, value: Tensor) -> None:
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


GM_co = TypeVar("GM_co", bound="GenerationMethods", covariant=True)


class GenerationMethods(ABC, Generic[GM_co]):
    streamer: CustomTextIteratorStreamer
    seed: int | None
    generation_kwargs: dict[str, Any]

    def __init__(
        self, streamer: CustomTextIteratorStreamer, seed: int | None, generation_kwargs: dict[str, Any]
    ) -> None:
        self.streamer = streamer
        self.seed = seed
        self.generation_kwargs = generation_kwargs

    @abstractmethod
    def merge_into_batch(self, others: list[GM_co]) -> None:
        pass

    @abstractmethod
    def generate(self) -> bool:
        pass


class TransformersGenerationMethods(GenerationMethods):
    _log = logging.getLogger(__name__)
    model: PreTrainedModel

    def __init__(
        self,
        model: PreTrainedModel,
        streamer: CustomTextIteratorStreamer,
        seed: int | None,
        generation_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(streamer, seed, generation_kwargs)

        self.model = model

    def _merge_kwargs_tensors(self, key: str, others: list["TransformersGenerationMethods"]) -> None:
        if key not in self.generation_kwargs:
            return

        self_tensor = cast(torch.Tensor, self.generation_kwargs[key])
        other_tensor_list = [cast(torch.Tensor, gm.generation_kwargs[key]) for gm in others]
        nested_tensor = torch.nested.as_nested_tensor([self_tensor, *other_tensor_list])
        self.generation_kwargs[key] = torch.squeeze(nested_tensor.to_padded_tensor(0))

    def _stack_kwargs_tensors(self, key: str, others: list["TransformersGenerationMethods"]) -> None:
        if key not in self.generation_kwargs:
            return

        self_tensor = cast(torch.Tensor, self.generation_kwargs[key])
        other_tensor_list = [cast(torch.Tensor, gm.generation_kwargs[key]) for gm in others]
        self.generation_kwargs[key] = torch.cat((self_tensor, *other_tensor_list), dim=0)

    def merge_into_batch(self, others: list["TransformersGenerationMethods"]) -> None:
        self._merge_kwargs_tensors("inputs", others)

        other_streamers = [gm.streamer for gm in others]
        self.generation_kwargs["streamer"] = BatchedCustomTextIteratorStreamer([self.streamer, *other_streamers])

    def generate(self) -> bool:
        if self.seed is not None:
            set_seed(self.seed)

        failed = False

        try:
            self.model.generate(**self.generation_kwargs)
        except StopGenerationError:
            pass
        except:
            self._log.exception("An error occurred when generating text")
            self.streamer.set_failed()
            failed = True

        if self.seed is not None:
            set_seed(int(datetime.now().timestamp()))

        return not failed


def to_transformers_chat_messages(messages: list[GenerationMessage]) -> list[dict[str, str]]:
    c = []

    for m in messages:
        for mc in m.contents:
            content: str

            if isinstance(mc, TextGenerationMessageContent):
                content = mc.text
            else:
                raise ValueError("Invalid generation message for chat: %s", str(mc))

            c.append({"role": m.role, "content": content})

    return c
