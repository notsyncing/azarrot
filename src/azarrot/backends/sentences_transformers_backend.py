import functools
import gc
import logging
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import cast, override

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from azarrot.backends.backend_base import BackendGenerationTask, BaseBackend
from azarrot.backends.common import (
    CompletionChunkStreamer,
    EmbeddingsGenerationResult,
    GenerationMethods,
)
from azarrot.backends.pytorch_common import (
    determine_pytorch_default_device,
    parse_pytorch_device_str,
    print_pytorch_device_list,
)
from azarrot.common_data import (
    EmbeddingModelInfo,
    EmbeddingsGenerationRequest,
    GenerationStatistics,
    Model,
    ModelInfo,
    RerankResultItem,
    ReranksGenerationRequest,
    TextGenerationRequest,
)
from azarrot.config import ServerConfig

BACKEND_ID_SENTENCE_TRANSFORMERS = "sentence-transformers"

SentenceTransformerModel = SentenceTransformer | CrossEncoder


@dataclass
class LoadedSentenceTransformersModel:
    data: Model
    model: SentenceTransformerModel
    device: str


class SentenceTransformerGenerationMethods(
    GenerationMethods["SentenceTransformerGenerationMethods", EmbeddingsGenerationResult]
):
    _log = logging.getLogger(__name__)
    _model: SentenceTransformer
    _inputs: list[list[str]]
    _gen_stats: GenerationStatistics

    def __init__(
        self, model: SentenceTransformer, inputs: list[str], generation_statistics: GenerationStatistics
    ) -> None:
        super().__init__()

        self._model = model
        self._inputs = [inputs]
        self._gen_stats = generation_statistics

    def get_inputs(self) -> list[list[str]]:
        return self._inputs

    @override
    def merge_into_batch(self, others: list["SentenceTransformerGenerationMethods"]) -> None:
        for other in others:
            self._inputs.append(other.get_inputs()[0])

    @override
    def generate(self) -> tuple[bool, list[EmbeddingsGenerationResult]]:
        input_counts = [len(input_list) for input_list in self._inputs]
        flatten_inputs: list[str] = functools.reduce(operator.iadd, self._inputs, [])

        try:
            embeddings = self._model.encode(flatten_inputs, convert_to_tensor=True)
        except:
            self._log.exception("An error occurred when generating embeddings")
            return False, []

        embeddings_list = embeddings.tolist()
        embeddings_count = len(embeddings_list)

        self._gen_stats.end_time = datetime.now()
        self._gen_stats.first_token_time = datetime.now()

        if embeddings_count > 0:
            self._gen_stats.completion_tokens = len(embeddings_list) * len(embeddings_list[0])
        else:
            self._gen_stats.completion_tokens = 0

        results: list[EmbeddingsGenerationResult] = []

        for count in input_counts:
            current_list = []
            counter = count

            while counter > 0:
                current_list.append(embeddings_list.pop(0))
                counter = counter - 1

            results.append(current_list)

        return True, results


class CrossEncoderGenerationMethods(GenerationMethods["CrossEncoderGenerationMethods", list[RerankResultItem]]):
    _log = logging.getLogger(__name__)
    _model: CrossEncoder
    _inputs: list[ReranksGenerationRequest]
    _gen_stats: GenerationStatistics

    def __init__(
        self, model: CrossEncoder, inputs: ReranksGenerationRequest, generation_statistics: GenerationStatistics
    ) -> None:
        super().__init__()

        self._model = model
        self._inputs = [inputs]
        self._gen_stats = generation_statistics

    def get_inputs(self) -> list[ReranksGenerationRequest]:
        return self._inputs

    @override
    def is_batching_supported(self) -> bool:
        return False

    @override
    def merge_into_batch(self, others: list["CrossEncoderGenerationMethods"]) -> None:
        raise NotImplementedError

    @override
    def generate(self) -> tuple[bool, list[list[RerankResultItem]]]:
        assert len(self._inputs) == 1

        gen_input = self._inputs[0]

        if gen_input.max_tokens_per_document is not None:
            for i, text in enumerate(gen_input.documents):
                text_length = len(text)

                if text_length > gen_input.max_tokens_per_document:
                    self._log.warning(
                        "Rerank input document index %d is overlength (%d vs %d), will be truncated!",
                        i,
                        text_length,
                        gen_input.max_tokens_per_document,
                    )

                    gen_input.documents[i] = gen_input.documents[i][0 : gen_input.max_tokens_per_document]

        try:
            ranks = self._model.rank(query=gen_input.query, documents=gen_input.documents, top_k=gen_input.max_count)
        except:
            self._log.exception("An error occurred when generating reranks")
            return False, []

        results = [
            RerankResultItem(input_index=cast("int", rank["corpus_id"]), score=float(rank["score"])) for rank in ranks
        ]

        self._gen_stats.end_time = datetime.now()
        self._gen_stats.first_token_time = datetime.now()
        self._gen_stats.prompt_tokens = sum([len(t) for t in gen_input.documents])
        self._gen_stats.completion_tokens = len(results)

        return True, [results]


class SentenceTransformersBackend(BaseBackend):
    _log = logging.getLogger(__name__)
    _default_device: str = "xpu"
    _models: dict[str, LoadedSentenceTransformersModel]

    def __init__(self, server_config: ServerConfig, force_use_device: str | None = None) -> None:
        super().__init__(server_config)

        self._models = {}

        if force_use_device is None:
            accel_device_count = print_pytorch_device_list(self._log, self.id())
            self._default_device = self._determine_default_device(accel_device_count)
        else:
            self._default_device = force_use_device

        self._log.info("Using default device: %s", self._default_device)

    def _determine_default_device(self, accel_device_count: int) -> str:
        return determine_pytorch_default_device(accel_device_count)

    @override
    def id(self) -> str:
        return BACKEND_ID_SENTENCE_TRANSFORMERS

    @override
    def _parse_device_str(self, device_str: str) -> list[str]:
        return parse_pytorch_device_str(device_str)

    @override
    def generate(self, request: TextGenerationRequest) -> tuple[CompletionChunkStreamer, GenerationStatistics]:
        raise NotImplementedError

    @override
    def _generate(
        self, request: TextGenerationRequest
    ) -> tuple[BackendGenerationTask, CompletionChunkStreamer, GenerationStatistics]:
        raise NotImplementedError

    @override
    def load_model(self, model: Model) -> ModelInfo:
        model_path = str(model.path.absolute())
        device = self._determine_device_for_model(model.id)
        model.device = device

        model_info: ModelInfo

        if model.task == "feature-extraction":
            st_model = SentenceTransformer(model_path, local_files_only=True, trust_remote_code=True, device=device)

            model_info = EmbeddingModelInfo(dimension=st_model.get_sentence_embedding_dimension() or -1)
        elif model.task == "text-classification":
            st_model = CrossEncoder(model_path, local_files_only=True, trust_remote_code=True, device=device)

            model_info = ModelInfo()
        else:
            raise ValueError(f"Unsupported task {model.task} for model {model.id}")

        self._models[model.id] = LoadedSentenceTransformersModel(model, st_model, device)

        self._log.info("Loaded model %s for task %s", model.id, model.task)

        return model_info

    @override
    def unload_model(self, model_id: str) -> None:
        if model_id not in self._models:
            self._log.warning("Model %s is not loaded.", model_id)
            return

        del self._models[model_id]
        torch.xpu.empty_cache()
        gc.collect()

        self._log.info("Model %s unloaded.", model_id)

    def __get_model(self, model_id: str) -> LoadedSentenceTransformersModel:
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} is not loaded!")

        return self._models[model_id]

    @override
    def _generate_embeddings(
        self, request: EmbeddingsGenerationRequest
    ) -> tuple[BackendGenerationTask, GenerationStatistics]:
        loaded_model = self.__get_model(request.model_id)

        if not isinstance(loaded_model.data.info, EmbeddingModelInfo):
            raise ValueError(f"Model {loaded_model.data.id} is not an embeddings model!")

        assert isinstance(loaded_model.model, SentenceTransformer)

        gen_stats = GenerationStatistics(
            start_time=datetime.now(),
            first_token_time=datetime.now(),
            end_time=datetime.max,
            prompt_tokens=0,
            completion_tokens=0,
        )

        m = SentenceTransformerGenerationMethods(
            model=loaded_model.model,
            inputs=request.text if isinstance(request.text, list) else [request.text],
            generation_statistics=gen_stats,
        )

        task = BackendGenerationTask(
            model_id=loaded_model.data.id,
            model_quirks=None,
            backend_id=self.id(),
            methods=m,
            device=loaded_model.device,
            seed=None,
        )

        return task, gen_stats

    @override
    def _generate_reranks(
        self, request: ReranksGenerationRequest
    ) -> tuple[BackendGenerationTask, GenerationStatistics]:
        loaded_model = self.__get_model(request.model_id)

        if not isinstance(loaded_model.model, CrossEncoder):
            raise ValueError(f"Model {loaded_model.data.id} is not a reranking model!")

        gen_stats = GenerationStatistics(
            start_time=datetime.now(),
            first_token_time=datetime.now(),
            end_time=datetime.max,
            prompt_tokens=0,
            completion_tokens=0,
        )

        m = CrossEncoderGenerationMethods(model=loaded_model.model, inputs=request, generation_statistics=gen_stats)

        task = BackendGenerationTask(
            model_id=loaded_model.data.id,
            model_quirks=None,
            backend_id=self.id(),
            methods=m,
            device=loaded_model.device,
            seed=None,
        )

        return task, gen_stats
