import uuid
from time import sleep

from pymilvus import MilvusClient
from pymilvus.client.types import LoadState

METADATA_KEY_FILE_ID = "file_id"


def make_collection_name(store_id: str | uuid.UUID) -> str:
    return f"vs_{str(store_id).replace('-', '_')}"


def load_collection(client: MilvusClient, collection_name: str) -> None:
    loaded = False

    while not loaded:
        collection_stat = client.get_load_state(collection_name)
        state = collection_stat["state"]

        if state == LoadState.NotLoad:
            client.load_collection(collection_name)
            sleep(1)
        elif state == LoadState.Loading:
            sleep(2)
        elif state == LoadState.NotExist:
            raise ValueError(f"Vector store collection {collection_name} does not exist!")
        else:
            loaded = True
