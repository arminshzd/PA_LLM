import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(data_dir:str = "./data") -> list:
    """
    Load all documents from the database (txt or pdf format only).

    Args:
        data_dir (str, optional): Database path. Defaults to "./data".
    """
    docs = []
    n_docs = 0

    # read the docs in
    for file in os.listdir(data_dir):
        f_path = os.path.join(data_dir, file)

        if f_path.endswith(".pdf"):
            loader = PyPDFLoader(f_path)
            docs.extend(loader.load())
            print(f"Loaded {file}.")
            n_docs += 1

        elif f_path.endswith(".txt"):
            loader = TextLoader(f_path, encoding="utf-8")
            docs.extend(loader.load())
            print(f"Loaded {file}.")
            n_docs += 1

    print(f"Total documents loaded: {n_docs}")
    return docs

def chunk_documents(documents:list) -> list:
    """
    Split the documents for retrieval.

    Args:
        documents (list): List containing all documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_vecstore(chunks:list, persist_dir: str="./vectorstore") -> Chroma:
    """
    Create embeddings and a vectorstore representation database for retrieval.

    Args:
        chunks (list): Chunked dataset
        persist_dir (str, optional): Directory to store the database. Defaults to "./vectorstore".

    Returns:
        Chroma: Chroma vector store database created from the dataset
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("Creating embeddings... (this make take a minute)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"Vector store created with {len(chunks)} chunks.")
    print(f"Vector store saved at: {persist_dir}")

    return vectorstore


def main():
    """
    main wrapper for data ingestion
    """

    print("=" * 50)
    print("Document Ingestion Pipeline")
    print("=" * 50)

    # load the docs
    print("\n[1/3] Loading documents...")
    documents = load_documents("./data")
    
    if len(documents) == 0:
        print("No documents found! Please add files to ./data directory")
        return
    
    # chunk the docs
    print("\n[2/3] Splitting documents...")
    chunks = chunk_documents(documents)
    
    # create vector store from chunks
    print("\n[3/3] Creating vector store...")
    vectorstore = create_vecstore(chunks, "./vectorstore")
    
    print("\n" + "=" * 50)
    print("Ingestion complete!")
    print("=" * 50)
    print("\nRun query.py to ask questions.")


if __name__=="__main__":
    main()