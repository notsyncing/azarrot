from langchain_core.embeddings import Embeddings

from azarrot.common_data import EmbeddingsGenerationRequest, Model
from azarrot.frontends.backend_pipe import BackendPipe


class LocalEmbeddings(Embeddings):
    _model: Model
    _backend_pipe: BackendPipe

    def __init__(self, model: Model, backend_pipe: BackendPipe) -> None:
        super().__init__()

        self._model = model
        self._backend_pipe = backend_pipe

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embed_data, _ = self._backend_pipe.generate_embeddings(
            self._model, EmbeddingsGenerationRequest(model_id=self._model.id, text=texts)
        )

        return embed_data

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
