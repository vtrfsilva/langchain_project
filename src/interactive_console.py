import numpy as np
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.vectorstores import FAISS

from src.embeddings import LLMStudioEmbeddings

def interactive_console(qa_chain : BaseCombineDocumentsChain, embeddings : LLMStudioEmbeddings, vectorstore : FAISS):
    while True:
        query = input("Digite sua pergunta (ou 'sair' para encerrar): ")
        if query.lower() == 'sair':
            print("Encerrando a sessão.")
            break
        query_embedding = embeddings.embed_query(query)
        D, I = vectorstore.index.search(np.array([query_embedding]), k=2)
        retrieved_docs = [vectorstore.docstore.search(i) for i in I[0] if vectorstore.docstore.search(i)]
        
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"[Documento {i}]: {doc.page_content}")
            response = qa_chain.run(input_documents=retrieved_docs, question=query)
            print(f"Resposta: {response}")
        else:
            print("Nenhum documento relevante foi encontrado.")
