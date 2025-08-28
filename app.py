
import os
from io import StringIO
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import asyncio
import nest_asyncio

# Patch nested event loops (important for Streamlit + async libs)
nest_asyncio.apply()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())



# ----------------------------
# App-wide constants & templates
# ----------------------------

DEFAULT_SALARY_TEXT = """
Salaries are typically structured with a monthly pay that sums to an annual total.
Annual salary = monthly salary × 12. Deductions may include income tax, provident fund,
professional tax, and other withholdings. Gross salary is the total before deductions.
Net (take-home) salary = Gross salary − total deductions. Some companies pay bonuses
or allowances (HRA, travel) that can be taxable or partially exempt.
"""

DEFAULT_INSURANCE_TEXT = """
An insurance policy usually defines coverage, premiums, and the claim process.
Coverage may include hospitalization, outpatient care, and prescription medicines.
The premium is the amount you pay periodically (monthly or yearly). Deductibles or
co-pays can apply. To file a claim: notify the insurer, submit required documents
(bills, prescriptions, discharge summary), and follow the approval process. Exclusions
(e.g., cosmetic procedures) are not covered.
"""

SALARY_AGENT_SYSTEM = (
    "You are the Salary Agent. ONLY answer questions about salary structure, monthly vs annual, deductions, gross vs net, allowances, bonuses, taxes, and payroll calculations. If the user asks about insurance or anything else, say briefly that you only handle salary topics, and suggest they ask the Insurance Agent."
)

INSURANCE_AGENT_SYSTEM = (
    "You are the Insurance Agent. ONLY answer questions about insurance coverage, premiums, claims, deductibles/co-pays, and exclusions. If the user asks about salary or anything else, say briefly that you only handle insurance topics, and suggest they ask the Salary Agent."
)

COORDINATOR_CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        """
You are a routing classifier. Read the user question and output one token exactly: SALARY or INSURANCE.
Choose SALARY for salary/payroll/deductions/compensation math. Choose INSURANCE for policy/coverage/premium/claims.
Question: {question}
Answer with exactly one word: SALARY or INSURANCE.
        """.strip()
    ),
)

# Answer formatting template for RAG agents
RAG_ANSWER_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "system"],
    template=(
        """
{system}
Use the provided context to answer concisely and helpfully. If the context lacks details, say so and provide a best-effort general answer within your domain.

Context:
{context}

Question: {question}
Helpful answer:
        """.strip()
    ),
)

# ----------------------------
# Utility: Build / update the vector store
# ----------------------------

def build_vector_store(texts: List[str], embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs: List[Document] = []
    for t in texts:
        for chunk in splitter.split_text(t):
            docs.append(Document(page_content=chunk))
    vs = FAISS.from_documents(docs, embedding=embeddings)
    return vs

# ----------------------------
# Coordinator (router)
# ----------------------------

def classify_query(llm: ChatGoogleGenerativeAI, question: str) -> str:
    """Return 'SALARY' or 'INSURANCE' using a minimal LLM call. Falls back to keywords."""
    try:
        msg = COORDINATOR_CLASSIFIER_PROMPT.format(question=question)
        out = llm.invoke(msg)
        label = (out.content or "").strip().upper()
        if label.startswith("SALARY"):
            return "SALARY"
        if label.startswith("INSURANCE"):
            return "INSURANCE"
    except Exception:
        pass
    # Fallback keyword heuristic
    q = question.lower()
    salary_kw = ["salary", "pay", "annual", "monthly", "deduction", "pf", "allowance", "bonus", "ctc", "take-home", "net", "gross"]
    insurance_kw = ["insurance", "policy", "premium", "coverage", "claim", "cashless", "deductible", "co-pay", "exclusion"]
    if any(k in q for k in salary_kw) and not any(k in q for k in insurance_kw):
        return "SALARY"
    return "INSURANCE"

# ----------------------------
# Agent builders
# ----------------------------

def build_rag_chain(llm: ChatGoogleGenerativeAI, retriever, system_prompt: str) -> RetrievalQA:
    # LangChain RetrievalQA with a custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": RAG_ANSWER_TEMPLATE.partial(system=system_prompt),
        },
        return_source_documents=True,
    )
    return qa_chain

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Multi-Agent RAG (Gemini): Salary + Insurance", page_icon="🤖", layout="wide")
load_dotenv()

with st.sidebar:
    st.header("⚙️ Setup")
    

    st.divider()
    st.subheader("📄 Knowledge Files")
    salary_file = st.file_uploader("Upload salary.txt", type=["txt"], key="salary_upl")
    insurance_file = st.file_uploader("Upload insurance.txt", type=["txt"], key="ins_upl")

    use_defaults = st.checkbox("Use built-in sample texts if files are missing", value=True)

    if st.button("📚 Build / Rebuild Knowledge Base", type="primary"):
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Please provide GOOGLE_API_KEY to build the vector store.")
        else:
            sal_text = DEFAULT_SALARY_TEXT
            ins_text = DEFAULT_INSURANCE_TEXT
            if salary_file is not None:
                sal_text = salary_file.read().decode("utf-8", errors="ignore")
            elif not use_defaults:
                st.warning("salary.txt not provided; using empty content.")
                sal_text = ""

            if insurance_file is not None:
                ins_text = insurance_file.read().decode("utf-8", errors="ignore")
            elif not use_defaults:
                st.warning("insurance.txt not provided; using empty content.")
                ins_text = ""

            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vs = build_vector_store([sal_text, ins_text], embeddings)
                st.session_state["vectorstore"] = vs
                st.session_state["retriever"] = vs.as_retriever(search_kwargs={"k": 4})
                st.success("Vector store built successfully ✅")

                # Confirmation of data source
                with st.expander("Show active knowledge base content"):
                    st.markdown("### Salary.txt Content")
                    st.text(sal_text[:1000] + ("..." if len(sal_text) > 1000 else ""))
                    st.markdown("### Insurance.txt Content")
                    st.text(ins_text[:1000] + ("..." if len(ins_text) > 1000 else ""))

            except Exception as e:
                st.exception(e)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {role, content}
if "vectorstore" not in st.session_state:
    # lazy-build a default VS so the app is usable immediately
    try:
        if os.getenv("GOOGLE_API_KEY"):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vs = build_vector_store([DEFAULT_SALARY_TEXT, DEFAULT_INSURANCE_TEXT], embeddings)
            st.session_state["vectorstore"] = vs
            st.session_state["retriever"] = vs.as_retriever(search_kwargs={"k": 4})
    except Exception:
        pass

st.title("🤖 Multi-Agent : Salary + Insurance ")
st.caption("Upload your TXT files to get better results and Information ")

col1, col2 = st.columns([3, 2])

with col2:
    st.subheader("🧪 Quick Test Prompts")
    if st.button("Salary: What is payroll?"):
        st.session_state["prefill"] = "What is payroll?"
    if st.button("Insurance: What is included in my insurance policy?"):
        st.session_state["prefill"] = "What is included in my insurance policy?"

with col1:
    st.subheader("💬 Chat")

    # Show chat history
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prefill = st.session_state.pop("prefill", "") if "prefill" in st.session_state else ""
    user_input = st.chat_input( placeholder=prefill or "e.g., What's my net salary after deductions?", key="chat_input")
    if prefill and not user_input:
        # allow user to just press Enter
        user_input = prefill

    if user_input:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Please provide GOOGLE_API_KEY in the sidebar to ask questions.")
        else:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Build LLM and agents lazily when needed
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

            retriever = st.session_state.get("retriever")
            if retriever is None:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vs = build_vector_store([DEFAULT_SALARY_TEXT, DEFAULT_INSURANCE_TEXT], embeddings)
                retriever = vs.as_retriever(search_kwargs={"k": 4})
                st.session_state["vectorstore"] = vs
                st.session_state["retriever"] = retriever

            salary_agent = build_rag_chain(llm, retriever, SALARY_AGENT_SYSTEM)
            insurance_agent = build_rag_chain(llm, retriever, INSURANCE_AGENT_SYSTEM)

            # Route
            route = classify_query(llm, user_input)

            with st.chat_message("assistant"):
                with st.spinner(f"Coordinator routed to {route.title()} Agent…"):
                    if route == "SALARY":
                        res = salary_agent.invoke({"query": user_input})
                    else:
                        res = insurance_agent.invoke({"query": user_input})

                    answer = res.get("result", "Sorry, I couldn't generate an answer.")
                    st.markdown(answer)

                    # Confirmation: show retrieved chunks from vector store
                    with st.expander("Show retrieved context (from uploaded files)"):
                        srcs = res.get("source_documents", []) or []
                        if not srcs:
                            st.info("No retrieved context found.")
                        for i, d in enumerate(srcs, 1):
                            st.markdown(f"**Chunk {i}:**\n\n{d.page_content}")

            st.session_state["messages"].append({"role": "assistant", "content": answer})

st.divider()


