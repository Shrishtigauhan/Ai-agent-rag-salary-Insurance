import os
from typing import List
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Multi-Agent RAG: Salary & Insurance", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent RAG")
st.caption("Two specialist agents share one vector store. A coordinator routes your question to the right expert.")

# -----------------------------
# API Key
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in your `.env` file or environment variables.")
    st.stop()

# -----------------------------
# Sidebar: Upload files
# -----------------------------
with st.sidebar:
    st.header("📄 Upload your knowledge files")
    salary_file = st.file_uploader("salary.txt", type=["txt"], help="Explains salaries (monthly, annual, deductions).")
    insurance_file = st.file_uploader("insurance.txt", type=["txt"], help="Explains insurance (coverage, premium, claims).")
    use_samples = st.checkbox("Use sample content if files are not provided", value=True)
    build_btn = st.button("🔧 Build / Rebuild Vector Store")

# -----------------------------
# Sample content
# -----------------------------
SAMPLE_SALARY = """Your sample salary text here..."""
SAMPLE_INSURANCE = """Your sample insurance text here..."""

def read_txt_file(file, fallback_text: str, name: str) -> Document:
    if file is not None:
        content = file.read().decode("utf-8", errors="ignore")
        return Document(page_content=content, metadata={"source": name})
    else:
        return Document(page_content=fallback_text, metadata={"source": name})

# -----------------------------
# Build vectorstore
# -----------------------------
@st.cache_resource(show_spinner=True)
def build_vectorstore(_api_key: str, salary_text: str, insurance_text: str):
    docs = [
        Document(page_content=salary_text, metadata={"source": "salary"}),
        Document(page_content=insurance_text, metadata={"source": "insurance"})
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_api_key)
        vs = FAISS.from_documents(split_docs, embeddings)
    except Exception:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vs = FAISS.from_documents(split_docs, embeddings)
    return vs.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# Load Documents
# -----------------------------
if salary_file is None and not use_samples:
    st.error("Please upload salary.txt or enable 'Use sample content'.")
    st.stop()
if insurance_file is None and not use_samples:
    st.error("Please upload insurance.txt or enable 'Use sample content'.")
    st.stop()

salary_doc = read_txt_file(salary_file, SAMPLE_SALARY, "salary.txt")
insurance_doc = read_txt_file(insurance_file, SAMPLE_INSURANCE, "insurance.txt")

if "retriever" not in st.session_state or build_btn:
    with st.spinner("Building vector store..."):
        st.session_state.retriever = build_vectorstore(
            api_key,
            salary_doc.page_content,
            insurance_doc.page_content
        )
        st.success("Vector store ready ✅")

retriever = st.session_state.get("retriever", None)
if retriever is None:
    st.error("Retriever not initialized. Click 'Build / Rebuild Vector Store' in the sidebar.")
    st.stop()

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

salary_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a Salary Expert. Only answer salary-related questions. "
        "If the question is not about salary, respond exactly: 'I cannot answer this.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)

insurance_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an Insurance Expert. Only answer insurance-related questions. "
        "If the question is not about insurance, respond exactly: 'I cannot answer this.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)

salary_agent = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": salary_prompt},
    return_source_documents=True,
)

insurance_agent = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": insurance_prompt},
    return_source_documents=True,
)

# -----------------------------
# Coordinator
# -----------------------------
def classify_query(query: str) -> str:
    system = (
        "You are a router. Read the user's question and answer with one word: "
        "'salary' if it is about salary/pay/deductions/CTC; "
        "'insurance' if it is about insurance/policy/premium/coverage/claims. "
        "If unclear, guess the most likely."
    )
    resp = llm.generate([{"role": "system", "content": system}, {"role": "user", "content": query}])
    label = resp.generations[0][0].text.strip().lower()
    return "salary" if "salary" in label else "insurance"

def coordinator(query: str):
    label = classify_query(query)
    if label == "salary":
        out = salary_agent.invoke({"query": query})
        return "Salary Agent", out
    else:
        out = insurance_agent.invoke({"query": query})
        return "Insurance Agent", out

# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about **salary** or **insurance**. "
                                         "Try: 'How do I calculate annual salary?' or 'What is included in my insurance policy?'."}
    ]

# Quick sidebar prompts
with st.sidebar:
    st.subheader("🧪 Quick prompts")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Annual salary?"):
            st.session_state.messages.append({"role": "user", "content": "How do I calculate annual salary?"})
    with col2:
        if st.button("Policy coverage?"):
            st.session_state.messages.append({"role": "user", "content": "What is included in my insurance policy?"})

# Display chat messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Type your question...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent_name, result = coordinator(query)
            answer = result["result"] if isinstance(result, dict) and "result" in result else str(result)
            st.markdown(f"**Handled by:** `{agent_name}`\n\n{answer}")

            # Show sources
            srcs = result.get("source_documents", []) if isinstance(result, dict) else []
            if srcs:
                with st.expander("Show retrieved sources"):
                    for i, d in enumerate(srcs, 1):
                        meta = d.metadata or {}
                        src = meta.get("source", "unknown")
                        st.markdown(f"**{i}.** *{src}*")
                        st.code(d.page_content.strip()[:1000], language="text")

    st.session_state.messages.append({"role": "assistant", "content": f"**Handled by:** `{agent_name}`\n\n{answer}"})
