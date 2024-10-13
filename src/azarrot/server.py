import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import alembic
import alembic.command
import alembic.config
import uvicorn
import yaml
from fastapi import FastAPI
from sqlalchemy import Engine, create_engine

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.ipex_llm_backend import IPEXLLMBackend
from azarrot.backends.openvino_backend import OpenVINOBackend
from azarrot.common_data import WorkingDirectories
from azarrot.config import ServerConfig
from azarrot.file_store import FileStore
from azarrot.frontends.backend_pipe import BackendPipe
from azarrot.frontends.openai_frontend import OpenAIFrontend
from azarrot.models.chat_templates import ChatTemplateManager
from azarrot.models.model_manager import ModelManager
from azarrot.tools import GLOBAL_TOOL_MANAGER

log = logging.getLogger(__name__)


def __parse_arguments_and_load_config() -> ServerConfig:
    parser = argparse.ArgumentParser(prog="azarrot", description="An OpenAI compatible server")

    parser.add_argument(
        "--config",
        dest="config_file",
        metavar="CONFIG_FILE",
        default="./server.yml",
        help="Specify the path to server config file",
    )

    parser.add_argument(
        "--models-dir", dest="models_dir", metavar="MODELS_DIR", help="Specify the path where models are stored"
    )

    parser.add_argument(
        "--working-dir",
        dest="working_dir",
        metavar="WORKING_DIR",
        help="Specify the path where caches and other mutable data are stored",
    )

    parser.add_argument("--host", dest="host", help="Specify the host of the API server")

    parser.add_argument("--port", dest="port", type=int, help="Specify the port of the API server")

    args = parser.parse_args()

    config = ServerConfig()

    config_file = Path(args.config_file).resolve()

    if config_file.exists():
        log.info("Using server config from %s", config_file)

        with config_file.open() as f:
            config_yaml = yaml.safe_load(f)

            if "models_dir" in config_yaml:
                config.models_dir = Path(config_yaml["models_dir"]).resolve()

            if "working_dir" in config_yaml:
                config.working_dir = Path(config_yaml["working_dir"]).resolve()

            if "host" in config_yaml:
                config.host = config_yaml["host"]

            if "port" in config_yaml:
                config.port = config_yaml["port"]

            if "model_device_map" in config_yaml:
                for k, v in config_yaml["model_device_map"].items():
                    config.model_device_map[k] = v

            if "single_token_generation_timeout" in config_yaml:
                config.single_token_generation_timeout = config_yaml["single_token_generation_timeout"]

            if "auto_batch_threshold" in config_yaml:
                config.auto_batch_threshold = config_yaml["auto_batch_threshold"]

            if "auto_batch_max_size" in config_yaml:
                config.auto_batch_max_size = config_yaml["auto_batch_max_size"]

            if "partial_file_expire_time" in config_yaml:
                config.partial_file_expire_time = config_yaml["partial_file_expire_time"]

    if args.models_dir is not None:
        config.models_dir = Path(args.models_dir).resolve()

    if args.working_dir is not None:
        config.working_dir = Path(args.working_dir).resolve()

    if args.host is not None:
        config.host = args.host

    if args.port is not None:
        config.port = args.port

    return config


def __create_directory(base_dir: Path, target_dir_name: str) -> Path:
    path = base_dir / target_dir_name

    if not path.exists():
        path.mkdir(parents=True)

    return path


def __create_working_directories(config: ServerConfig) -> WorkingDirectories:
    if not config.working_dir.exists():
        config.working_dir.mkdir(parents=True)

    image_temp_path = __create_directory(config.working_dir, "upload_images")
    file_store_path = __create_directory(config.working_dir, "uploaded_files")

    return WorkingDirectories(root=config.working_dir, uploaded_images=image_temp_path, file_store=file_store_path)


def __init_database(database_path: Path) -> Engine:
    url = f"sqlite:///{database_path.absolute()}"
    migration_script_path = str(Path(__file__).absolute().parent / Path("alembic"))
    log.info("Database url: %s, migration scripts at: %s", url, migration_script_path)

    alembic_cfg = alembic.config.Config()
    alembic_cfg.set_main_option("script_location", migration_script_path)
    alembic_cfg.set_main_option("sqlalchemy.url", url)
    alembic.command.upgrade(alembic_cfg, "head")

    return create_engine(url)


@dataclass
class Server:
    config: ServerConfig
    model_manager: ModelManager
    backend_pipe: BackendPipe
    backends: list[BaseBackend]
    frontends: list[OpenAIFrontend]
    api: FastAPI

    _uvicorn_server: uvicorn.Server | None = None

    def start(self) -> None:
        log.info("Starting API server...")
        uvicorn_config = uvicorn.Config(self.api, host=self.config.host, port=self.config.port)
        self._uvicorn_server = uvicorn.Server(uvicorn_config)
        self._uvicorn_server.run()

    def stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            self._uvicorn_server = None


def create_server(config: ServerConfig | None = None, enable_backends: list[type[BaseBackend]] | None = None) -> Server:
    log.info("Azarrot is initializing...")

    if config is None:
        config = __parse_arguments_and_load_config()

    log.info("Current config:")

    for attr in dir(config):
        if not attr.startswith("__"):
            log.info("%s = %s", attr, getattr(config, attr))

    working_dirs = __create_working_directories(config)

    db = __init_database(working_dirs.root / "azarrot.db")
    file_store = FileStore(config, working_dirs.file_store, db)

    chat_template_manager = ChatTemplateManager(GLOBAL_TOOL_MANAGER)

    backends: list[BaseBackend]

    if enable_backends is not None:
        backends = [b(config) for b in enable_backends]
    else:
        backends = [
            IPEXLLMBackend(config),
            OpenVINOBackend(config),
        ]

    log.info("Enabled backends: %s", backends)

    model_manager = ModelManager(config, backends)

    backend_pipe = BackendPipe(backends, chat_template_manager, GLOBAL_TOOL_MANAGER)

    api = FastAPI()

    frontends = [OpenAIFrontend(model_manager, backend_pipe, file_store, api, working_dirs)]

    return Server(
        config=config,
        model_manager=model_manager,
        backend_pipe=backend_pipe,
        backends=backends,
        frontends=frontends,
        api=api,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    server = create_server()
    server.start()


if __name__ == "__main__":
    main()
