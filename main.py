import streamlit as st
from dotenv import load_dotenv
import pickle
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain 
from langchain_core.prompts import ChatPromptTemplate

# Sidebar contents
with st.sidebar:
    st.title("SSP assistant😊")
    st.markdown('''
        ## About
        This app was designed by ssp students to ease their revision process 
    ''')
    #add_vertical_space()
    st.write('Made by khris calvin')
    
    # Display vector store info
    if os.path.exists("ITB_notes_2025.pkl"):
        file_size = os.path.getsize("ITB_notes_2025.pkl") / (1024 * 1024)
        st.info(f"📊 Vector DB loaded: {file_size:.2f} MB")

# Configuration
VECTOR_STORE_PATH = "ITB_notes_2025.pkl"  # Change this to your .pkl file name

@st.cache_resource
def load_embeddings():
    """
    Load the EXACT same embedding model used during vector store creation.
    IMPORTANT: This must match the model used in Google Colab!
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs
    )
    return embeddings

@st.cache_resource
def load_vector_store(path):
    """Load the pre-created vector store from pickle file"""
    if not os.path.exists(path):
        st.error(f"❌ Vector store file not found: {path}")
        st.error(f"📁 Current directory: {os.getcwd()}")
        st.error(f"📂 Available files: {[f for f in os.listdir('.') if f.endswith('.pkl')]}")
        st.info("💡 Please run the Google Colab notebook to create the vector store first!")
        st.stop()
    
    file_size = os.path.getsize(path) / (1024 * 1024)
    
    try:
        with open(path, "rb") as f:
            vector_store = pickle.load(f)
        st.success(f"✅ Vector store loaded successfully! ({file_size:.2f} MB)")
        return vector_store
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        st.error("💡 Make sure the .pkl file was created with the same langchain version")
        st.stop()

@st.cache_resource
def setup_qa_chain(_vector_store):
    """Set up the QA chain using modern LangChain approach"""
    load_dotenv()
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        temperature=0, 
        model="gemini-2.5-flash"  # or "gemini-1.5-pro" for better quality
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert in Bussiness administration. Answer the following question based only on the provided context.
    Think step by step and provide a clear, detailed answer.
    If the context doesn't contain enough information, say so but go ahead use the data you where trained on to answer the prompt provided.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    
    Answer:""")
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retriever
    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
    )
    
    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

def main():
    st.header("🎓 Romy.ai - Introduction to bussiness administration")
    
    # Load resources
    try:
        with st.spinner("Loading embeddings..."):
            embeddings = load_embeddings()
        
        with st.spinner("Please don't close the tab wait a moment for the resources to load..."):
            vector_store = load_vector_store(VECTOR_STORE_PATH)
        
        with st.spinner("Setting up AI chain..."):
            qa_chain = setup_qa_chain(vector_store)
        
    except Exception as e:
        st.error(f"❌ Failed to load resources: {e}")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask questions about bussiness administration...."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from the chain
                    response = qa_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    if not answer or answer.strip() == "":
                        answer = "I couldn't find relevant information. Please try rephrasing your question."
                    
                    st.markdown(answer)
                    
                    # Optional: Show source documents in expander
                    with st.expander("📚 View source documents"):
                        for i, doc in enumerate(response["context"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(doc.page_content[:300] + "...")
                            st.divider()
                    
                except Exception as e:
                    answer = f"❌ An error occurred: {str(e)}\n\nPlease try again."
                    st.error(f"Error type: {type(e).__name__}")
                    st.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()