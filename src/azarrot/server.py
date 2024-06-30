import argparse
import logging
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI

from azarrot.backends.openvino_backend import OpenVINOBackend
from azarrot.config import ServerConfig
from azarrot.frontends.openai_frontend import OpenAIFrontend
from azarrot.models import ModelManager

log = logging.getLogger(__name__)

openvino_backend: OpenVINOBackend


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

    config_file = Path(args.config_file).absolute()

    if config_file.exists():
        log.info("Using server config from %s", config_file)

        with config_file.open() as f:
            config_yaml = yaml.safe_load(f)

            if config_yaml["models_dir"] is not None:
                config.models_dir = Path(config_yaml["models_dir"]).absolute()

            if config_yaml["working_dir"] is not None:
                config.working_dir = Path(config_yaml["working_dir"]).absolute()

            if config_yaml["host"] is not None:
                config.host = config_yaml["host"]

            if config_yaml["port"] is not None:
                config.port = config_yaml["port"]

            if config_yaml["model_device_map"] is not None:
                for k, v in config_yaml["model_device_map"].items():
                    config.model_device_map[k] = v

    if args.models_dir is not None:
        config.models_dir = Path(args.models_dir).absolute()

    if args.working_dir is not None:
        config.working_dir = Path(args.working_dir).absolute()

    if args.host is not None:
        config.host = args.host

    if args.port is not None:
        config.port = args.port

    return config


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    log.info("Azarrot is initializing...")

    config = __parse_arguments_and_load_config()
    log.info("Current config:")

    for attr in dir(config):
        if not attr.startswith("__"):
            log.info("%s = %s", attr, getattr(config, attr))

    if not config.working_dir.exists():
        config.working_dir.mkdir(parents=True)

    backend = OpenVINOBackend(config)

    model_manager = ModelManager(config, backend)

    log.info("Starting API server...")
    api = FastAPI()
    OpenAIFrontend(model_manager, backend, api)
    uvicorn.run(api, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
