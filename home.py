from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langsmith import Client
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.vectorstores import Chroma  # LangChain's Chroma wrapper
from main import embedding_model  # Your SentenceTransformer instance

# Embedder wrapper class to match LangChain's expected interface
class EmbeddingWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # texts: list of strings
        return self.model.encode(texts)

    def embed_query(self, text):
        # text: single string
        return self.model.encode([text])[0]

# Wrap your SentenceTransformer embedding model
embedding_function = EmbeddingWrapper(embedding_model)

llm = ChatOllama(
    model="llama3",
    temperature=0.8,
    num_predict=256,
    verbose=True
)

query = "BUmble Bee media"

# Initialize Chroma vectorstore with wrapper
vectorstore = Chroma(
    collection_name="bumblebee_docs",
    embedding_function=embedding_function,
    persist_directory="./chroma_store"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

client = Client()
qa_prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat")

combine_doc_chain = create_stuff_documents_chain(llm, qa_prompt)

retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_doc_chain
)

# Invoke the retrieval chain
result = retrieval_chain.invoke({"input": query})

print(result['answer'])
