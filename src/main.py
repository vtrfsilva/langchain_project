from src.embeddings import LLMStudioEmbeddings
from src.llm import CustomLLM
from src.faiss_handler import create_faiss_index
from src.document_loader import load_documents
from src.interactive_console import interactive_console
from langchain.chains.question_answering import load_qa_chain

def main():
    # Configurações iniciais
    api_url = "http://localhost:1234"
    documents_path = "./data/documents.txt"

    # Carregar documentos
    documents = load_documents(documents_path)

    # Criar embeddings e FAISS
    embeddings = LLMStudioEmbeddings(api_url=f"{api_url}")
    vectorstore = create_faiss_index(documents, embeddings)

    # Configurar LLM e QA Chain
    llm = CustomLLM(api_url=f"{api_url}/v1/chat/completions")
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    # Iniciar console interativo
    interactive_console(qa_chain, embeddings, vectorstore)

if __name__ == "__main__":
    main()
