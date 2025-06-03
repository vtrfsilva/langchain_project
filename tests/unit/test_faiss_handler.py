class DummyEmbeddings:
    def __init__(self, embeds):
        self._embeds = embeds
    def embed_documents(self, texts):
        return self._embeds


def test_create_faiss_index():
    from src.faiss_handler import create_faiss_index
    from langchain.schema import Document

    docs = [Document(page_content="a"), Document(page_content="b")]
    embeds = DummyEmbeddings([[1, 0], [0, 1]])
    vectorstore = create_faiss_index(docs, embeds)

    assert vectorstore.index.vectors == [[1, 0], [0, 1]]
    assert vectorstore.docstore.search(0).page_content == "a"
