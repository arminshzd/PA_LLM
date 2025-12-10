from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)

results = vectorstore.similarity_search("machine learning", k=3)
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content[:500])  # Print first 200 chars