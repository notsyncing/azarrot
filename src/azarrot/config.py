from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_TOKENS = 512
DEFAULT_REASONING_MAX_TOKENS = 32768

ENV_AZARROT_TEST_MODE = "AZARROT_TEST_MODE"
ENV_AZARROT_TEST_RESOURCES_ROOT = "AZARROT_TEST_RESOURCES_ROOT"


@dataclass
class OpenAIFrontendConfig:
    vector_store_default_embedding_model_id: str | None = None
    assistant_file_search_reranker_default_model_id: str | None = None


@dataclass
class VectorStoreConfig:
    worker_file_process_scan_interval: int = 60  # Seconds
    worker_max_file_count_per_scan: int = 10
    worker_store_cleanup_scan_interval: int = 60  # Seconds


@dataclass
class ModelPrefixCacheConfig:
    max_cache_size: int = 8 * 1024 * 1024 * 1024


@dataclass
class ServerConfig:
    models_dir: Path = Path("./models")
    working_dir: Path = Path("./working")
    host: str = "127.0.0.1"
    port: int = 8080

    huggingface_download_to_home: bool = False

    model_device_map: dict[str, str] = field(default_factory=dict)
    single_token_generation_timeout: int = 60000
    auto_batch_threshold: int = 100
    auto_batch_max_size: int = 8
    default_seed: int | None = None
    log_generation_details: bool = False

    partial_file_expire_time: int = 3600 * 1000

    vector_store_configs: VectorStoreConfig = field(default_factory=lambda: VectorStoreConfig())

    openai_configs: OpenAIFrontendConfig = field(default_factory=lambda: OpenAIFrontendConfig())

    model_prefix_cache_configs: dict[str, ModelPrefixCacheConfig] = field(default_factory=dict)
