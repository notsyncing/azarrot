import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast, override

import torch
from transformers.cache_utils import DynamicCache
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed

from azarrot.backends.caching import ModelPrefixCache
from azarrot.backends.common import (
    BatchedCustomTextIteratorStreamer,
    CustomTextIteratorStreamer,
    GenerationMethods,
    StopGenerationError,
)
from azarrot.common_data import (
    GenerationMessage,
    ImageGenerationMessageContent,
    TextGenerationMessageContent,
    ToolCallRequestMessageContent,
    ToolCallResponseMessageContent,
)

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin


class TransformersModelPrefixCache(ModelPrefixCache[DynamicCache]):
    _cache: DynamicCache | None = None
    _cached_tokens: torch.Tensor | None = None

    def __init__(self, max_size: int) -> None:
        super().__init__(max_size)

    @override
    def _cache_splitter(self, cache: DynamicCache, next_part_start_pos: int) -> tuple[DynamicCache, DynamicCache]:
        prev_cache = DynamicCache()
        next_cache = DynamicCache()

        for layer in range(len(cache)):
            prev_cache.update(
                key_states=cache.key_cache[layer][..., :next_part_start_pos, :],
                value_states=cache.value_cache[layer][..., :next_part_start_pos, :],
                layer_idx=layer,
            )

            next_cache.update(
                key_states=cache.key_cache[layer][..., next_part_start_pos:, :],
                value_states=cache.value_cache[layer][..., next_part_start_pos:, :],
                layer_idx=layer,
            )

        return prev_cache, next_cache

    @override
    def _cache_merger(self, caches: list[DynamicCache]) -> DynamicCache:
        if len(caches) <= 0:
            raise ValueError("Empty list!")

        if len(caches) == 1:
            return caches[0]

        new_cache = DynamicCache()

        for layer in range(len(caches[0])):
            new_cache.update(caches[0].key_cache[layer].clone(), caches[0].value_cache[layer].clone(), layer)

        for cache in caches[1:]:
            for layer in range(len(cache)):
                new_cache.update(cache.key_cache[layer], cache.value_cache[layer], layer)

        return new_cache

    @override
    def _cache_sizing(self, cache: DynamicCache) -> int:
        total_size = 0

        for layer in range(len(cache)):
            total_size += cache.key_cache[layer].element_size() * cache.key_cache[layer].nelement()
            total_size += cache.value_cache[layer].element_size() * cache.value_cache[layer].nelement()

        return total_size + 64

    @override
    def _cache_shaping(self, cache: DynamicCache) -> torch.Size:
        if len(cache) <= 0:
            return torch.Size([])

        return cache.value_cache[0].size()

    @override
    def _create_empty_cache(self, to_device: str) -> DynamicCache:
        return DynamicCache()

    @override
    def _copy_cache(self, original_cache: DynamicCache) -> DynamicCache:
        new_cache = DynamicCache()
        new_cache.key_cache = []
        new_cache.value_cache = []

        for c in original_cache.key_cache:
            t = c.clone() if c.is_contiguous() else c.contiguous()
            new_cache.key_cache.append(t)

        for c in original_cache.value_cache:
            t = c.clone() if c.is_contiguous() else c.contiguous()
            new_cache.value_cache.append(t)

        return new_cache


class TransformersGenerationMethods(GenerationMethods["TransformersGenerationMethods", CustomTextIteratorStreamer]):
    _log = logging.getLogger(__name__)

    streamer: CustomTextIteratorStreamer
    seed: int | None
    generation_kwargs: dict[str, Any]
    model: "PreTrainedModel"
    prefix_cache: TransformersModelPrefixCache | None

    def __init__(
        self,
        model: "PreTrainedModel",
        streamer: CustomTextIteratorStreamer,
        prefix_cache: TransformersModelPrefixCache | None,
        seed: int | None,
        generation_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        self.streamer = streamer
        self.seed = seed
        self.generation_kwargs = generation_kwargs
        self.model = model
        self.prefix_cache = prefix_cache

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
        self._merge_kwargs_tensors("input_ids", others)

        if "past_key_values" in self.generation_kwargs:
            cache = cast("DynamicCache", self.generation_kwargs["past_key_values"])
            other_caches = [cast("DynamicCache", gm.generation_kwargs["past_key_values"]) for gm in others]
            self.generation_kwargs["past_key_values"] = DynamicCache.from_batch_splits([cache, *other_caches])

            cache_position = cast("torch.Tensor", self.generation_kwargs.get("cache_position", torch.Tensor([0])))
            other_cache_positions = [gm.generation_kwargs.get("cache_position", torch.Tensor([0])) for gm in others]
            self.generation_kwargs["cache_position"] = torch.cat([cache_position, *other_cache_positions])

        other_streamers = [gm.streamer for gm in others]
        self.generation_kwargs["streamer"] = BatchedCustomTextIteratorStreamer([self.streamer, *other_streamers])

    @override
    def split_from_batch(self, others: list["TransformersGenerationMethods"]) -> None:
        if "past_key_values" in self.generation_kwargs:
            cache = cast("DynamicCache", self.generation_kwargs["past_key_values"])
            all_caches = cache.batch_split(cache.key_cache[0].size(dim=0), 1)
            self.generation_kwargs["past_key_values"] = all_caches[0]

            for i, other in enumerate(others):
                other.generation_kwargs["past_key_values"] = all_caches[i + 1]

    @override
    def generate(self) -> tuple[bool, list[CustomTextIteratorStreamer]]:
        if self.seed is not None:
            set_seed(self.seed)

        failed = False

        try:
            with torch.inference_mode():
                self.model.generate(**self.generation_kwargs)  # type: ignore[reportCallIssue, operator, unused-ignore]
        except StopGenerationError:
            pass
        except:
            self._log.exception("An error occurred when generating text")
            self.streamer.set_failed()
            failed = True

        if self.seed is not None:
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
    def on_execution_successful(self, index_in_batch: int) -> None:
        if self.prefix_cache is not None and "past_key_values" in self.generation_kwargs:
            cache = cast("DynamicCache", self.generation_kwargs["past_key_values"])
            input_tokens: torch.Tensor = self.generation_kwargs["input_ids"]
            output_tokens = self.streamer.get_all_output_tokens()

            if output_tokens is not None:
                self.prefix_cache.update_cache(cache, torch.cat([input_tokens[0], output_tokens[:-1]]))

    @override
    def on_execution_failed(self) -> None:
        self.streamer.set_failed()


class ProcessorToTokenizerAdapter:
    _processor: "ProcessorMixin"

    def __init__(self, processor: "ProcessorMixin") -> None:
        self._processor = processor

    def decode(self, tokens: list, **kwargs: Any) -> str:
        return (cast("Any", self._processor)).decode(tokens, **kwargs)

    def encode(self, input_text: Any, **kwargs: Any) -> list[int]:
        if isinstance(self._processor, PreTrainedTokenizerBase):
            return self._processor.encode(input_text, **kwargs)
        else:
            tokenizer: PreTrainedTokenizerBase = cast("Any", self._processor).tokenizer
            return tokenizer.encode(input_text, **kwargs)


def to_transformers_chat_messages(messages: list[GenerationMessage]) -> list[dict[str, str]]:
    c = []

    for m in messages:
        transformers_msg: dict[str, Any]

        if isinstance(m.contents[0], ToolCallRequestMessageContent):
            transformers_msg = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": t.id,
                        "function": {"name": t.function_name, "arguments": t.function_arguments},
                    }
                    for t in m.contents
                    if isinstance(t, ToolCallRequestMessageContent)
                ],
            }

            c.append(transformers_msg)
        elif all(isinstance(mc, ToolCallResponseMessageContent) for mc in m.contents):
            for mc in m.contents:
                assert isinstance(mc, ToolCallResponseMessageContent)

                transformers_msg = {"role": "tool", "content": mc.result}

                c.append(transformers_msg)
        else:
            contents: list | str

            if len(m.contents) == 1 and isinstance(m.contents[0], TextGenerationMessageContent):
                # For many models only expecting non-list content field in their chat templates
                contents = m.contents[0].text
            elif all(isinstance(mc, TextGenerationMessageContent) for mc in m.contents):
                # For many models only expecting non-list content field in their chat templates
                contents = "".join([(cast("TextGenerationMessageContent", mc)).text for mc in m.contents])
            else:
                contents = []

                for mc in m.contents:
                    if isinstance(mc, TextGenerationMessageContent):
                        contents.append({"type": "text", "text": mc.text})
                    elif isinstance(mc, ImageGenerationMessageContent):
                        contents.append({"type": "image", "path": mc.image_file_path})
                    else:
                        raise ValueError(f"Generation message for chat contains unsupported content {mc}: {m}")

            transformers_msg = {"role": m.role, "content": contents}
            c.append(transformers_msg)

    return c
