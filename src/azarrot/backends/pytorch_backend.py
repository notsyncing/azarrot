import logging
from typing import TYPE_CHECKING, Any, override

import torch

from azarrot.backends.transformers_based_backend import TransformersBasedBackend
from azarrot.common_data import Model
from azarrot.config import ServerConfig

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils import PreTrainedTokenizer

BACKEND_ID_PYTORCH = "pytorch"


class PyTorchBackend(TransformersBasedBackend):
    _log = logging.getLogger(__name__)
    _force_use_device: str | None = None

    def __init__(self, config: ServerConfig, force_use_device: str | None = None) -> None:
        self._force_use_device = force_use_device

        super().__init__(config)

    @override
    def _determine_default_device(self, accel_device_count: int) -> str:
        if self._force_use_device is not None:
            self._log.info("Forced to use device %s", self._force_use_device)
            return self._force_use_device

        return super()._determine_default_device(accel_device_count)

    @override
    def id(self) -> str:
        return BACKEND_ID_PYTORCH

    @override
    def _customize_model_and_kwargs(self, model: Model, model_config: Any, model_kwargs: dict[str, Any]) -> None:
        model_kwargs["low_cpu_mem_usage"] = True

        # TODO: Enable this when bitsandbytes is usable
        # if not model.use_original_precision and "quantization_config" not in model_config:
        #     model_kwargs["quantization_config"] = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=torch.bfloat16
        #     )

    @override
    def _customize_loaded_model(
        self,
        model: Model,
        loaded_model: "PreTrainedModel",
        loaded_tokenizer: "PreTrainedTokenizer",
        model_kwargs: dict[str, Any],
    ) -> "PreTrainedModel":
        if model.pytorch is not None:
            if model.pytorch.compile:
                self._log.info("Compiling model %s with backend %s", model.id, model.pytorch.compile_backend)
                loaded_model.forward = torch.compile(loaded_model.forward, backend=model.pytorch.compile_backend)

        return loaded_model
