import types
import sys
import pytest
from pathlib import Path

# Ensure project root is on sys.path so 'src' can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class FakeResponse:
    def __init__(self, data=None):
        self._data = data or {}
    def json(self):
        return self._data

@pytest.fixture(autouse=True)
def fake_external_modules(monkeypatch):
    # requests
    requests = types.ModuleType('requests')
    requests.post = lambda *a, **k: FakeResponse()
    monkeypatch.setitem(sys.modules, 'requests', requests)

    # numpy
    numpy = types.ModuleType('numpy')
    numpy.array = lambda data: list(data)
    monkeypatch.setitem(sys.modules, 'numpy', numpy)

    # langchain base structure
    langchain = types.ModuleType('langchain')

    # embeddings
    embeddings_pkg = types.ModuleType('langchain.embeddings')
    embeddings_base = types.ModuleType('langchain.embeddings.base')
    class Embeddings:  # minimal base class
        pass
    embeddings_base.Embeddings = Embeddings
    embeddings_pkg.base = embeddings_base
    langchain.embeddings = embeddings_pkg

    # llms
    llms_pkg = types.ModuleType('langchain.llms')
    llms_base = types.ModuleType('langchain.llms.base')
    class LLM:
        pass
    llms_base.LLM = LLM
    llms_pkg.base = llms_base
    langchain.llms = llms_pkg

    # vectorstores
    vectorstores_pkg = types.ModuleType('langchain.vectorstores')
    class FAISS:
        def __init__(self, embedding_function, index, docstore, index_to_docstore_id):
            self.embedding_function = embedding_function
            self.index = index
            self.docstore = docstore
            self.index_to_docstore_id = index_to_docstore_id
    vectorstores_pkg.FAISS = FAISS
    langchain.vectorstores = vectorstores_pkg

    # docstore
    docstore_pkg = types.ModuleType('langchain.docstore')
    class InMemoryDocstore:
        def __init__(self, store):
            self.store = store
        def search(self, key):
            return self.store.get(key)
    docstore_pkg.InMemoryDocstore = InMemoryDocstore
    langchain.docstore = docstore_pkg

    # schema
    schema_pkg = types.ModuleType('langchain.schema')
    class Document:
        def __init__(self, page_content):
            self.page_content = page_content
    schema_pkg.Document = Document
    langchain.schema = schema_pkg

    # chains
    chains_pkg = types.ModuleType('langchain.chains')
    qa_pkg = types.ModuleType('langchain.chains.question_answering')
    def load_qa_chain(llm, chain_type="stuff"):
        class QAChain:
            def __init__(self, llm):
                self.llm = llm
            def run(self, input_documents, question):
                return "answer"
        return QAChain(llm)
    qa_pkg.load_qa_chain = load_qa_chain
    chains_pkg.question_answering = qa_pkg

    combine_pkg = types.ModuleType('langchain.chains.combine_documents')
    combine_base = types.ModuleType('langchain.chains.combine_documents.base')
    class BaseCombineDocumentsChain:
        def run(self, input_documents, question):
            return "answer"
    combine_base.BaseCombineDocumentsChain = BaseCombineDocumentsChain
    combine_pkg.base = combine_base
    chains_pkg.combine_documents = combine_pkg
    langchain.chains = chains_pkg

    # register langchain modules
    for mod in [
        ('langchain', langchain),
        ('langchain.embeddings', embeddings_pkg),
        ('langchain.embeddings.base', embeddings_base),
        ('langchain.llms', llms_pkg),
        ('langchain.llms.base', llms_base),
        ('langchain.vectorstores', vectorstores_pkg),
        ('langchain.docstore', docstore_pkg),
        ('langchain.schema', schema_pkg),
        ('langchain.chains', chains_pkg),
        ('langchain.chains.question_answering', qa_pkg),
        ('langchain.chains.combine_documents', combine_pkg),
        ('langchain.chains.combine_documents.base', combine_base),
    ]:
        monkeypatch.setitem(sys.modules, mod[0], mod[1])

    # faiss
    faiss_pkg = types.ModuleType('faiss')
    class IndexFlatL2:
        def __init__(self, dim):
            self.dim = dim
            self.vectors = []
        def add(self, arr):
            self.vectors.extend(arr)
        def search(self, queries, k):
            n = min(k, len(self.vectors))
            I = [list(range(n)) for _ in queries]
            D = [[0.0] * n for _ in queries]
            return D, I
    faiss_pkg.IndexFlatL2 = IndexFlatL2
    monkeypatch.setitem(sys.modules, 'faiss', faiss_pkg)

    yield
