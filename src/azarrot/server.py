import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
import yaml
from fastapi import FastAPI

from azarrot.backends.ipex_llm_backend import IPEXLLMBackend
from azarrot.backends.openvino_backend import OpenVINOBackend
from azarrot.common_data import WorkingDirectories
from azarrot.config import ServerConfig
from azarrot.frontends.backend_pipe import BackendPipe
from azarrot.frontends.openai_frontend import OpenAIFrontend
from azarrot.models.chat_templates import ChatTemplateManager
from azarrot.models.model_manager import ModelManager
from azarrot.tools import GLOBAL_TOOL_MANAGER

if TYPE_CHECKING:
    from azarrot.backends.backend_base import BaseBackend

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

    if args.models_dir is not None:
        config.models_dir = Path(args.models_dir).resolve()

    if args.working_dir is not None:
        config.working_dir = Path(args.working_dir).resolve()

    if args.host is not None:
        config.host = args.host

    if args.port is not None:
        config.port = args.port

    return config


def __create_working_directories(config: ServerConfig) -> WorkingDirectories:
    if not config.working_dir.exists():
        config.working_dir.mkdir(parents=True)

    image_temp_path = config.working_dir / "uploaded_images"

    if not image_temp_path.exists():
        image_temp_path.mkdir(parents=True)

    return WorkingDirectories(root=config.working_dir, uploaded_images=image_temp_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    log.info("Azarrot is initializing...")

    config = __parse_arguments_and_load_config()
    log.info("Current config:")

    for attr in dir(config):
        if not attr.startswith("__"):
            log.info("%s = %s", attr, getattr(config, attr))

    working_dirs = __create_working_directories(config)

    chat_template_manager = ChatTemplateManager(GLOBAL_TOOL_MANAGER)

    backends: list[BaseBackend] = [
        IPEXLLMBackend(config),
        OpenVINOBackend(config),
    ]

    model_manager = ModelManager(config, backends)

    backend_pipe = BackendPipe(backends, chat_template_manager, GLOBAL_TOOL_MANAGER)

    log.info("Starting API server...")
    api = FastAPI()
    OpenAIFrontend(model_manager, backend_pipe, api, working_dirs)
    uvicorn.run(api, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
