from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from chromadb import PersistentClient

# Load file
loader = TextLoader("a.txt")
documents = loader.load()

# Split the document
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(texts)

# Initialize Chroma
client = PersistentClient(path="./chroma_ratan")
collection = client.get_or_create_collection(name="ratan_docs")

# Clear existing data (optional for clean testing)
collection.delete(ids=[f"chunk-{i}" for i in range(len(texts))])

# Store chunks
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    collection.add(documents=[text], embeddings=[embedding], ids=[f"chunk-{i}"])

# print("✅ Stored Ratan Tata content in Chroma.")
# query = "Who is Ratan Tata?"
# query_embedding = embedding_model.encode([query])[0]



# results = collection.query(
#     query_embeddings=[query_embedding],
#     n_results=3,
#     include=["documents", "distances"]
# )

# # Debug output
# print("Distances:", results["distances"])
# print("Results:", results["documents"])


# for doc, dist in zip(results["documents"][0], results["distances"][0]):
#     if dist < 0.7:
#         print(f"✅ Similarity: {1 - dist:.2f} — Result: {doc}")
#     else:
#         print(f"❌ Too dissimilar (distance: {dist:.2f}) — Skipped")
