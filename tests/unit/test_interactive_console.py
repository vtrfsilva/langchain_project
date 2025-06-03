from types import SimpleNamespace
from unittest.mock import MagicMock



def test_interactive_console(monkeypatch, capsys):
    from src.interactive_console import interactive_console
    from langchain.schema import Document
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0, 1]

    index = MagicMock()
    index.search.return_value = ([[0.0]], [[0]])
    docstore = MagicMock()
    docstore.search.return_value = Document(page_content="text")
    vectorstore = SimpleNamespace(index=index, docstore=docstore)

    qa_chain = MagicMock()
    qa_chain.run.return_value = "resp"

    inputs = iter(["pergunta", "sair"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    interactive_console(qa_chain, embeddings, vectorstore)

    qa_chain.run.assert_called_once()
    captured = capsys.readouterr().out
    assert "Encerrando a sessão." in captured
