from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from utils import format_source_excerpt

def load_vecstore(persist_dir: str="./vectorstore") -> Chroma:
    """
    load the vector store created by `ingest.py`

    Args:
        persist_dir (str, optional): Directory of the Chromadb vector store. Defaults to "./vectorstore".

    Returns:
        Chroma: loaded Chromadb vector store
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    print(f"Loaded vector store with {vectorstore._collection.count()} chunks.")
    return vectorstore

def init_llm() -> OllamaLLM:
    """
    Initialize an Ollama instance (Llama 3.2)

    Returns:
        OllamaLLM: Initiated LLM instance
    """
    llm = OllamaLLM(
        model = "llama3.2",
        temperature=0.1
    )

    print("Initialized Llama 3.2 model")
    return llm

def format_docs(docs: list) -> str:
    """
    Format retrieved documents into a string for passing to the LLM.

    Args:
        docs (list): Retrieved chunks

    Returns:
        str: Formatted string for LLM
    """

    formatted = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        formatted.append(f"[Document {i}] (Source: {source}, Page: {page})\n{doc.page_content}\n")
    
    return "\n\n".join(formatted)

def create_prompt(context: str, question: str, history: list) -> str:
    """
    Create a prompt to pass to the LLM containing the instructions, context from
    the retrieved documents, and the user's question.

    Args:
        context (str): Context from vector store
        question (str): User's question

    Returns:
        str: Prepared prompt for the LLM
    """

    history_prompt = "\n".join([
            f"Previous Q: {q}\nPrevious A: {a}" 
            for q, a in history[-3:]  # Last 2 exchanges
        ])

    prompt = f"""You are a personal assistant answering questions only using the information in the provided context.

    {history_prompt}

    Context from relavant documents:
    {context}

    Question: {question}

    Instructions:
    - Answer the question based ONLY on the context provided above
    - If the answer is in the context, provide a clear and concise answer
    - Always cite which document number(s) your answer comes from (e.g., "According to Document 1...")
    - If the answer is NOT in the context, say "I don't have enough information to answer that question based on the available documents"
    - Do not make up information; "I don't know" is always preffered to incorrect information

    Answer:"""

    return prompt

def query(vectorstore: Chroma, llm: OllamaLLM, question: str, history: list, k: int=3):
    """
    RAG pipeline implementation:
    1. retrieve relavant chunks
    2. format context
    3. create prompt
    4. get LLM's response
    5. Display results with sources

    Args:
        vectorstore (Chroma): Database's vector store created with `ingest.py`
        llm (OllamaLLM): Instance of the LLM model
        question (str): User's prompt
        history (list): History of previous prompts
        k (int, optional): Number of sources to retrieve from the vector restore. Defaults to 3.
    """

    print(f"\nQuestion: {question}")
    print("-" * 50)

    # retrieving the top `k` relavant chunks
    retrieved = vectorstore.similarity_search(question, k=3)

    if not retrieved:
        print("No relavant information found in the database.")
        return None
    
    # format the context
    context = format_docs(retrieved)

    # create the LLM prompt
    prompt = create_prompt(context, question, history)

    # get LLM response
    print(f"\nAnswer:")
    print("-" * 50)
    answer = llm.invoke(prompt)
    print(answer)

    print("\n" + "=" * 50)
    print("Sources:")
    print("=" * 50)
    
    for i, doc in enumerate(retrieved, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        
        print(f"\n[Document {i}] Source: {source} (Page: {page})")
        print(f"Relevant excerpt: {format_source_excerpt(doc.page_content[:200])}...")
    
    return {
        'answer': answer,
        'source_documents': retrieved
    }

def main():
    """
    Main wrapper for `query.py`
    """
    print("=" * 50)
    print("Personal Assistant Q&A System")
    print("=" * 50)

    vecstore = load_vecstore()
    llm = init_llm()

    # history tracker
    history = []

    # interactive loop
    while True:
        user_q = input("\nAsk a question (or 'quit' to exit): ").strip()

        if user_q.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not user_q:
            print("No question entered. Please enter a question.")
            continue

        try:
            query_res = query(vecstore, llm, user_q, history)
            history.append([user_q, query_res['answer']])
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try rephrasing your question.")    


if __name__=="__main__":
    main()