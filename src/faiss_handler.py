import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

def create_faiss_index(documents, embeddings):
    texts = [doc.page_content for doc in documents]
    document_embeddings = embeddings.embed_documents(texts)
    index = faiss.IndexFlatL2(len(document_embeddings[0]))
    index.add(np.array(document_embeddings))
    docstore = InMemoryDocstore(dict(enumerate(documents)))
    return FAISS(
        embedding_function=embeddings.embed_documents,
        index=index,
        docstore=docstore,
        index_to_docstore_id={i: i for i in range(len(texts))}
    )
