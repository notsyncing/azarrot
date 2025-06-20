import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast, override

import torch
from transformers.trainer_utils import set_seed

from azarrot.backends.common import (
    BatchedCustomTextIteratorStreamer,
    CustomTextIteratorStreamer,
    GenerationMethods,
    StopGenerationError,
)
from azarrot.common_data import GenerationMessage, TextGenerationMessageContent, ToolCallRequestMessageContent

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


class TransformersGenerationMethods(GenerationMethods["TransformersGenerationMethods", CustomTextIteratorStreamer]):
    _log = logging.getLogger(__name__)

    streamer: CustomTextIteratorStreamer
    _seed: int | None
    generation_kwargs: dict[str, Any]
    _model: "PreTrainedModel"

    def __init__(
        self,
        model: "PreTrainedModel",
        streamer: CustomTextIteratorStreamer,
        seed: int | None,
        generation_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        self.streamer = streamer
        self._seed = seed
        self.generation_kwargs = generation_kwargs
        self._model = model

    def _merge_kwargs_tensors(self, key: str, others: list["TransformersGenerationMethods"]) -> None:
        if key not in self.generation_kwargs:
            return

        self_tensor = cast("torch.Tensor", self.generation_kwargs[key])
        other_tensor_list = [cast("torch.Tensor", gm.generation_kwargs[key]) for gm in others]
        nested_tensor = torch.nested.as_nested_tensor([self_tensor, *other_tensor_list])
        self.generation_kwargs[key] = torch.squeeze(nested_tensor.to_padded_tensor(0))

    def _stack_kwargs_tensors(self, key: str, others: list["TransformersGenerationMethods"]) -> None:
        if key not in self.generation_kwargs:
            return

        self_tensor = cast("torch.Tensor", self.generation_kwargs[key])
        other_tensor_list = [cast("torch.Tensor", gm.generation_kwargs[key]) for gm in others]
        self.generation_kwargs[key] = torch.cat((self_tensor, *other_tensor_list), dim=0)

    @override
    def merge_into_batch(self, others: list["TransformersGenerationMethods"]) -> None:
        self._merge_kwargs_tensors("inputs", others)

        other_streamers = [gm.streamer for gm in others]
        self.generation_kwargs["streamer"] = BatchedCustomTextIteratorStreamer([self.streamer, *other_streamers])

    @override
    def generate(self) -> tuple[bool, list[CustomTextIteratorStreamer]]:
        if self._seed is not None:
            set_seed(self._seed)

        failed = False

        try:
            with torch.inference_mode():
                self._model.generate(**self.generation_kwargs)
        except StopGenerationError:
            pass
        except:
            self._log.exception("An error occurred when generating text")
            self.streamer.set_failed()
            failed = True

        if self._seed is not None:
            set_seed(int(datetime.now().timestamp()))

        result_streamer = self.generation_kwargs["streamer"]

        if isinstance(result_streamer, BatchedCustomTextIteratorStreamer):
            results = result_streamer.get_batched_streamers()
        else:
            results = [self.streamer]

        return not failed, results

    @override
    def update_start_generation_time(self, time: datetime) -> None:
        self.streamer.update_start_generation_time(time)

    @override
    def on_execution_failed(self) -> None:
        self.streamer.set_failed()


def to_transformers_chat_messages(messages: list[GenerationMessage]) -> list[dict[str, str]]:
    c = []

    for m in messages:
        transformers_msg: dict[str, Any]

        if isinstance(m.contents[0], TextGenerationMessageContent):
            transformers_msg = {"content": m.contents[0].text}
        elif isinstance(m.contents[0], ToolCallRequestMessageContent):
            transformers_msg = {
                "tool_calls": [
                    {
                        "type": "function",
                        "id": t.id,
                        "function": {"name": t.function_name, "arguments": t.function_arguments},
                    }
                    for t in m.contents
                    if isinstance(t, ToolCallRequestMessageContent)
                ]
            }
        else:
            raise ValueError(f"Invalid generation message for chat: {m}")

        transformers_msg["role"] = m.role
        c.append(transformers_msg)

    return c
