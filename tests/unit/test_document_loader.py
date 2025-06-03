import tempfile


def test_load_documents(tmp_path):
    from src.document_loader import load_documents
    from langchain.schema import Document

    data = "line1\nline2\n"
    file_path = tmp_path / "docs.txt"
    file_path.write_text(data)

    docs = load_documents(str(file_path))

    assert [d.page_content for d in docs] == ["line1", "line2"]
