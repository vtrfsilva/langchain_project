
class DummyResponse:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data


def test_call_updates_history(monkeypatch):
    import requests
    from src.llm import CustomLLM
    def mock_post(url, json):
        assert json == {"messages": [{"role": "user", "content": "hi"}]}
        return DummyResponse({"choices": [{"message": {"content": "hey"}}]})
    monkeypatch.setattr(requests, "post", mock_post)
    llm = CustomLLM(api_url="http://mock")
    response = llm._call("hi")
    assert response == "hey"
    assert llm.message_history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]
