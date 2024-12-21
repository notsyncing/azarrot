import logging
from typing import Any

import torch
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoConfig
from typing_extensions import override

from azarrot.backends.transformers_based_backend import TransformersBasedBackend
from azarrot.common_data import (
    Model,
)

IPEX_LLM_TASK_MODEL_MAP = {
    "text-generation": AutoModelForCausalLM,
}

BACKEND_ID_IPEX_LLM = "ipex-llm"


class IPEXLLMBackend(TransformersBasedBackend):
    _log = logging.getLogger(__name__)

    @override
    def id(self) -> str:
        return BACKEND_ID_IPEX_LLM

    @override
    def _get_model_class(self, task: str) -> Any:
        return IPEX_LLM_TASK_MODEL_MAP.get(task)

    @override
    def _customize_model_kwargs(self, model: Model, model_kwargs: dict[str, Any]) -> None:
        model_config = AutoConfig.from_pretrained(model.path.absolute(), trust_remote_code=True)

        if "quantization_config" in model_config:
            quantization_config = model_config.quantization_config

            if "quant_method" in quantization_config:
                quantization_method = quantization_config["quant_method"]

                if quantization_method == "gptq":
                    self._log.info("GPTQ model detected. Will use torch_dtype=torch.float to load this model.")
                    model_kwargs["torch_dtype"] = torch.float

        if model.ipex_llm is not None:
            if model.ipex_llm.use_cache:
                model_kwargs["use_cache"] = True

        load_in_4bit = False
        load_in_low_bit = None

        if not model.use_original_precision:
            if model.ipex_llm is not None:
                if model.ipex_llm.quantization_mode == "default":
                    load_in_4bit = True
                else:
                    load_in_low_bit = model.ipex_llm.quantization_mode
            else:
                load_in_4bit = True

        model_kwargs["load_in_4bit"] = load_in_4bit
        model_kwargs["load_in_low_bit"] = load_in_low_bit

        model_kwargs["optimize_model"] = True
