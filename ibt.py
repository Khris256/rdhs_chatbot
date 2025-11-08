import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space

# Try importing with error handling
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.embeddings import HuggingFaceEmbeddings

    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Dependencies may be incompatible. Check requirements.txt")
    st.info("Required: pydantic, langchain-google-genai")
    st.stop()

load_dotenv()

# IMPORTANT: API key is stored in Streamlit Cloud Secrets
# Go to: App Settings > Secrets > Add GOOGLE_API_KEY
# This code will NOT expose your API key in the public repo
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("⚠️ GOOGLE_API_KEY not found in Streamlit secrets!")
    st.info("Please add your API key in Streamlit Cloud: Settings > Secrets")
    st.stop()

st.set_page_config(
    page_title="UDHS Chatbot😎",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="auto",
)

# Sidebar contents
with st.sidebar:
    st.title("UDHS assistant😊")
    st.markdown('''
        ## About
        Welcome to the UDHS Chatbot, This AI assistant have been retrained on udhs report to give insights and statistics about about udhs.
    ''')
    add_vertical_space()
    st.write('Made by Khris Calvin')

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    body {
        color: #FAFAFA;
        background-color:#020203;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        text-align: right;
    }
    .stChatMessage.assistant {
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("IBT Chatbot")

VECTOR_STORE_PATH = "UDHS-2022-Report(1).pkl"

@st.cache_resource
def load_embeddings():
    """
    Load the EXACT same embedding model used during vector store creation.
    Model: sentence-transformers/all-MiniLM-L6-v2
    Device: CPU (as specified in main.py)
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
        st.error(f"Current directory: {os.getcwd()}")
        st.error(f"Available files: {os.listdir('.')}")
        st.stop()
    

    
    try:
        with open(path, "rb") as f:
            vector_store = pickle.load(f)
        st.write(f"Hi there ✌️ I an AI designed to help you answer questions about UDHS report. Ask any question below.")
        return vector_store
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        st.stop()

@st.cache_resource
def setup_qa_chain(_vector_store):
    """Set up the QA chain using modern LangChain approach"""
    llm = ChatGoogleGenerativeAI(
        temperature=0, 
        model="gemini-2.5-flash"
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    System Role:
You are a professional data assistant trained on the Uganda Demographic and Health Survey (UDHS) and related demographic datasets.
Your task is to accurately answer user questions using facts retrieved from the UDHS vector database.
All numerical or comparative results should be presented in structured tables for clarity.

⚙️ Core Instructions

Grounding:

Use only verified facts retrieved from the UDHS or related national survey content in the database.

Never invent or estimate values not explicitly found in the data.

Response Format:
Always follow this structure when data is available:

🔹 Summary:
[Brief explanation of what the table shows in simple terms.]

🔹 Statistical Table:
| Indicator | Value | Year | Region/Subgroup | Source (UDHS Section) |
|------------|--------|------|-----------------|-----------------------|
| Example: Under-five mortality rate | 64 deaths per 1,000 | 2016 | National | Chapter 8 |
| Example: Infant mortality rate | 43 deaths per 1,000 | 2016 | Urban areas | Chapter 8 |

🔹 Insights:
[Highlight key observations or comparisons in one or two short paragraphs.]

🔹 Data Source:
Uganda Demographic and Health Survey ([year]), [chapter or section if available].


When Multiple Years Exist:

Present them side-by-side in one table, e.g.:

Indicator	2011	2016	% Change	Source
Total Fertility Rate	6.2	5.4	-13%	UDHS

If Data is Not Found:
Reply with:
“No specific UDHS statistic was found for that indicator. Please refine your query (e.g., specify region, year, or variable).”

Clarity & Tone:

Keep language formal, factual, and clear.

Avoid unnecessary narrative text.

Always include table headers and units (%, rate per 1,000, etc.).

Optional (if your app supports rendering):

You may use Markdown tables or HTML tables depending on your frontend (e.g., for web apps, <table> formatting looks better).

If user requests visualization, suggest:
“Would you like a chart view (bar or line) for these statistics?”

🧩 Example Response (using Markdown table)

User: What are Uganda’s fertility trends from 2000 to 2016?
AI Output:

🔹 Summary:
Fertility rates in Uganda have steadily declined over the years according to UDHS reports.

🔹 Statistical Table:
| Year | Total Fertility Rate (TFR) | Urban | Rural | Source |
|------|-----------------------------|--------|--------|---------|
| 2000–01 | 6.9 | 4.8 | 7.1 | UDHS 2000–01 |
| 2006 | 6.7 | 4.4 | 7.1 | UDHS 2006 |
| 2011 | 6.2 | 4.1 | 6.8 | UDHS 2011 |
| 2016 | 5.4 | 4.0 | 5.9 | UDHS 2016 |

🔹 Insights:
The Total Fertility Rate (TFR) has decreased from 6.9 in 2000 to 5.4 in 2016 — showing a 22% decline. Urban fertility remains lower than rural across all years.

🔹 Data Source:
Uganda Demographic and Health Survey (2000–2016), Fertility Section.
    
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
        search_kwargs={"k": 8}
    )
    
    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# Load resources
try:
    embeddings = load_embeddings()
    vector_store = load_vector_store(VECTOR_STORE_PATH)
    qa_chain = setup_qa_chain(vector_store)
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about UDHS report..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Use the retrieval chain
            response = qa_chain.invoke({"input": prompt})
            full_response = response["answer"]
            
            if not full_response or full_response.strip() == "":
                full_response = "I couldn't find relevant information in the document. Please try rephrasing your question."
            
        except Exception as e:
            full_response = f"❌ An error occurred: {str(e)}\n\nPlease try again or rephrase your question."
            st.error(f"Error type: {type(e).__name__}")
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})