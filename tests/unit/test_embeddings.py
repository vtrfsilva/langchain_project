
class DummyResponse:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data


def test_embed_query(monkeypatch):
    from src.embeddings import LLMStudioEmbeddings
    import requests
    def mock_post(url, json):
        assert json == {"input": "text"}
        return DummyResponse({"data": [{"embedding": [1, 2, 3]}]})
    monkeypatch.setattr(requests, "post", mock_post)
    emb = LLMStudioEmbeddings(api_url="http://mock")
    result = emb.embed_query("text")
    assert result == [1, 2, 3]


def test_embed_documents(monkeypatch):
    from src.embeddings import LLMStudioEmbeddings
    monkeypatch.setattr(LLMStudioEmbeddings, "embed_query", lambda self, t: [len(t)])
    emb = LLMStudioEmbeddings(api_url="http://mock")
    result = emb.embed_documents(["a", "bb"])
    assert result == [[1], [2]]
