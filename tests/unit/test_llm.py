
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


def test_instances_have_separate_histories(monkeypatch):
    import requests
    from src.llm import CustomLLM

    def mock_post(url, json):
        # Always return a simple assistant response regardless of input
        return DummyResponse({"choices": [{"message": {"content": "resp"}}]})

    # Patch requests.post for the test and ensure the module under test uses it
    monkeypatch.setattr(requests, "post", mock_post)
    import src.llm as llm_module
    monkeypatch.setattr(llm_module, "requests", requests, raising=False)

    llm1 = CustomLLM(api_url="http://mock")
    llm2 = CustomLLM(api_url="http://mock")

    llm1._call("hello1")
    # Only llm1 should have history at this point
    assert llm2.message_history == []

    llm2._call("hello2")

    assert llm1.message_history == [
        {"role": "user", "content": "hello1"},
        {"role": "assistant", "content": "resp"},
    ]
    assert llm2.message_history == [
        {"role": "user", "content": "hello2"},
        {"role": "assistant", "content": "resp"},
    ]
    assert llm1.message_history is not llm2.message_history
