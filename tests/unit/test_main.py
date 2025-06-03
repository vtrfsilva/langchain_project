from unittest.mock import MagicMock


def test_main_invokes_console(monkeypatch):
    import src.main as main_module
    from langchain.schema import Document
    monkeypatch.setattr(main_module, "load_documents", lambda path: [Document("d")])
    monkeypatch.setattr(main_module, "LLMStudioEmbeddings", lambda api_url: MagicMock())
    monkeypatch.setattr(main_module, "create_faiss_index", lambda docs, emb: MagicMock())
    monkeypatch.setattr(main_module, "CustomLLM", lambda api_url: MagicMock())
    monkeypatch.setattr(main_module, "load_qa_chain", lambda llm, chain_type="stuff": MagicMock())
    called = {}
    def fake_console(chain, emb, vs):
        called['yes'] = True
    monkeypatch.setattr(main_module, "interactive_console", fake_console)

    main_module.main()

    assert called.get('yes')
