import argparse
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from azarrot.backends.openvino_backend import OpenVINOBackend
from azarrot.config import ServerConfig
from azarrot.frontends.openai_frontend import OpenAIFrontend
from azarrot.models import ModelManager

log = logging.getLogger(__name__)

openvino_backend: OpenVINOBackend


def __parse_arguments() -> ServerConfig:
    parser = argparse.ArgumentParser(prog="azarrot", description="An OpenAI compatible server")

    parser.add_argument("--models-dir", dest="models_dir", metavar="MODELS_DIR", default="./models",
        help="Specify the path where models are stored")

    parser.add_argument("--working-dir", dest="working_dir", metavar="WORKING_DIR", default="./working",
        help="Specify the path where caches and other mutable data are stored")

    parser.add_argument("--host", dest="host", default="127.0.0.1", help="Specify the host of the API server")

    parser.add_argument("--port", dest="port", type=int, default=8080, help="Specify the port of the API server")

    args = parser.parse_args()

    return ServerConfig(
        models_dir=Path(args.models_dir).absolute(),
        working_dir=Path(args.working_dir).absolute(),
        host=args.host,
        port=args.port
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    log.info("Azarrot is initializing...")

    config = __parse_arguments()
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
