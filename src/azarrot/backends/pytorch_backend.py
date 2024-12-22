import logging
from typing import Any, cast

import torch
from transformers import PreTrainedModel
from typing_extensions import override

from azarrot.backends.transformers_based_backend import TransformersBasedBackend
from azarrot.common_data import Model

BACKEND_ID_PYTORCH = "pytorch"


class PyTorchBackend(TransformersBasedBackend):
    _log = logging.getLogger(__name__)

    @override
    def id(self) -> str:
        return BACKEND_ID_PYTORCH

    @override
    def _customize_model_kwargs(self, model: Model, model_kwargs: dict[str, Any]) -> None:
        model_kwargs["low_cpu_mem_usage"] = True

        # TODO: Enable this when bitsandbytes is usable
        # if not model.use_original_precision:
        #     model_kwargs["quantization_config"] = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=torch.bfloat16
        #     )

    @override
    def _customize_loaded_model(self, model: Model, loaded_model: PreTrainedModel) -> PreTrainedModel:
        if model.pytorch is not None:
            if model.pytorch.compile:
                self._log.info("Compiling model %s with backend %s", model.id, model.pytorch.compile_backend)
                compiled_model = torch.compile(loaded_model, backend=model.pytorch.compile_backend)
                loaded_model = cast(PreTrainedModel, compiled_model)

        return loaded_model
