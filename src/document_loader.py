from langchain.schema import Document

def load_documents(file_path):
    with open(file_path, 'r') as f:
        raw_documents = f.readlines()
    return [Document(page_content=text.strip()) for text in raw_documents]
