import logging
from backends.openvino import OpenVINOBackend


log = logging.getLogger(__name__)

openvino_backend: OpenVINOBackend


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    log.info("Azarrot is initializing...")

    openvino_backend = OpenVINOBackend()


if __name__ == "__main__":
    main()