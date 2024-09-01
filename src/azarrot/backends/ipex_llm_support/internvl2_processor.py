from typing import Any, cast

import torch
from transformers import PreTrainedTokenizer

from azarrot.backends.common import TransformersGenerationMethods
from azarrot.backends.ipex_llm_support.internvl2_tools import load_image
from azarrot.common_data import GenerationMessage, ImageGenerationMessageContent, TextGenerationMessageContent

INTERNVL2_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"  # noqa: S105


class InternVL2TransformersGenerationMethods(TransformersGenerationMethods):
    def merge_into_batch(self, others: list["TransformersGenerationMethods"]) -> None:
        self._merge_kwargs_tensors("input_ids", others)
        self._merge_kwargs_tensors("attention_mask", others)
        self._stack_kwargs_tensors("pixel_values", others)

        return super().merge_into_batch(others)


def internvl2_patch_model(model: Any, tokenizer: PreTrainedTokenizer) -> None:
    img_context_token_id = tokenizer.convert_tokens_to_ids(INTERNVL2_IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id


def internvl2_apply_chat_template(
    model: Any, tokenizer: PreTrainedTokenizer, messages: list[GenerationMessage]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    image_list: list[torch.Tensor] = []

    c = []

    for m in messages:
        final_content = ""

        for mc in m.contents:
            if isinstance(mc, TextGenerationMessageContent):
                final_content += mc.text
            elif isinstance(mc, ImageGenerationMessageContent):
                image = load_image(mc.image_file_path)
                image_list.append(image)
                final_content += f"<img>{INTERNVL2_IMG_CONTEXT_TOKEN * model.num_image_token * image.size(0)}</img>"
            else:
                raise ValueError("Invalid generation message for chat: %s", str(mc))

        c.append({"role": m.role, "content": final_content})

    inputs = tokenizer.apply_chat_template(c, add_generation_prompt=True, return_tensors="pt")

    pixel_values = torch.cat(image_list) if len(image_list) > 0 else None
    return cast(torch.Tensor, inputs), pixel_values
