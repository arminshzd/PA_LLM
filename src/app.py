import streamlit as st
from query import load_vecstore, init_llm, format_docs, create_prompt

st.title("Virtual Personal Assistant")
st.markdown("Ask questions about your documents and get AI-powered answers with citations.")

# Initialize (cache so it only runs once)
@st.cache_resource
def load_system():
    vectorstore = load_vecstore()
    llm = init_llm()
    return vectorstore, llm

vectorstore, llm = load_system()

if "history" not in st.session_state:
    st.session_state.history = []

# Input
question = st.text_input("Your question:", placeholder="What did I write about...?")

if st.button("Submit") and question:
    history = st.session_state.history
    with st.spinner("Thinking..."):
        docs = vectorstore.similarity_search(question, k=3)

        if not docs:
            st.warning("No relevant information found in the database.")
        else:
            context = format_docs(docs)
            prompt = create_prompt(context, question, history)
            answer = llm.invoke(prompt)

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Document {i}: {doc.metadata.get('source', 'Unknown')}"):
                    st.write(doc.page_content)

            history.append((question, answer))
