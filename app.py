
import os
from io import StringIO
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


from langchain_community.embeddings import HuggingFaceEmbeddings


st.set_page_config(page_title="Multi-Agent RAG: Salary & Insurance", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent RAG • ")
st.caption("Two specialist agents share one vector store. A coordinator routes your question to the right expert.")

# -----------------------------
# 1) API Key management
# -----------------------------
# -----------------------------
# 1) API Key management
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in your `.env` file or environment variables.")
    st.stop()

with st.sidebar:
    st.header("📄 Upload your knowledge files")
    salary_file = st.file_uploader("salary.txt", type=["txt"], help="Explains salaries (monthly, annual, deductions).")
    insurance_file = st.file_uploader("insurance.txt", type=["txt"], help="Explains insurance (coverage, premium, claims).")

    use_samples = st.checkbox("Use sample content if files are not provided", value=True)
    build_btn = st.button("🔧 Build / Rebuild Vector Store")


def read_txt_file(file, fallback_text: str, name: str) -> Document:
    if file is not None:
        content = file.read().decode("utf-8", errors="ignore")
        return Document(page_content=content, metadata={"source": name})
    else:
        return Document(page_content=fallback_text, metadata={"source": name})

SAMPLE_SALARY = """
Salary Structure (India Example):

1. **Basic Pay**
   - The fixed component of salary; usually 35–50% of CTC.
   - Forms the basis for calculating allowances and deductions.

2. **Allowances**
   - **HRA (House Rent Allowance):** Paid if employee lives in rented accommodation. Tax exemption depends on salary, rent paid, and city.
   - **DA (Dearness Allowance):** Adjustment for inflation, more common in govt/public sector jobs.
   - **Conveyance Allowance:** For travel between home and office.
   - **Medical Allowance:** For medical expenses, sometimes requires bills.
   - **Special Allowance / Performance Allowance:** Variable or discretionary.

3. **Gross Salary**
   - Formula: Basic Pay + All Allowances + Bonus + Other Benefits.

4. **Deductions**
   - **Income Tax (TDS):** Based on annual taxable income.
   - **Provident Fund (PF):** 12% of Basic Pay from employee; employer also contributes.
   - **Professional Tax:** Levied by some states.
   - **Employee State Insurance (ESI):** For employees earning below a certain threshold.
   - **Other deductions:** Loan EMI, advances, etc.

5. **Net Salary (In-hand Salary)**
   - Formula: Gross Salary – Deductions.
   - This is what gets credited to bank account monthly.

6. **Annual Salary**
   - Formula: Monthly Salary × 12.
   - Includes fixed and some variable components.

7. **CTC (Cost to Company)**
   - The total annual cost the employer spends on an employee.
   - Includes Gross Salary + Employer Contributions (PF, Gratuity, Insurance premiums, Stock Options).
   - CTC is usually higher than in-hand salary.

8. **Bonus & Incentives**
   - Performance Bonus (annual or quarterly).
   - Joining Bonus, Retention Bonus, Referral Bonus.

9. **Gratuity**
   - A lump-sum benefit paid if the employee works 5+ years in the company.
   - Formula (basic): (15 × last drawn salary × years of service) / 26.

10. **Example Salary Calculation**
    - Monthly Salary (CTC) = ₹80,000
      • Basic Pay = ₹32,000
      • HRA = ₹16,000
      • Allowances = ₹12,000
      • Bonus (monthly avg) = ₹5,000
    - Gross = ₹65,000
    - Deductions:
      • PF (Employee) = ₹3,840
      • Tax (TDS) = ₹5,000
      • Professional Tax = ₹200
    - Net (In-hand) = ~₹56,000
    - Annual CTC = ₹9.6 lakh
    - Annual In-hand = ~₹6.7 lakh

11. **Key Salary FAQs**
    - Difference between Gross & Net? → Deductions.
    - Why is CTC higher than In-hand? → Includes employer contributions, not directly received.
    - What deductions are mandatory? → Income Tax, PF, ESI (if eligible).
"""

SAMPLE_INSURANCE = """
Insurance Policy Basics:

1. **Health Insurance**
   - Covers hospitalization, outpatient, and emergency medical care.
   - Types:
     • Individual Plan – covers one person.
     • Family Floater Plan – covers all family members under single sum insured.
     • Group Health Insurance – usually offered by employer.
   - Coverage:
     • Room rent, ICU charges, surgery, doctor’s fees, medicines.
     • Pre- and post-hospitalization expenses (usually 30 & 60 days).
   - Claims:
     • Cashless (network hospital directly settles).
     • Reimbursement (pay and then claim back).
   - Exclusions:
     • Cosmetic surgery, experimental treatment, self-inflicted injuries.
     • Pre-existing conditions excluded for 2–4 years (varies by policy).
   - Example: Premium ₹15,000/year for ₹5 lakh coverage for family of 4.

2. **Life Insurance**
   - Provides financial support to nominee in case of policyholder’s death.
   - Types:
     • Term Plan – Pure risk cover; cheapest and highest coverage.
     • Endowment Plan – Insurance + savings, lower coverage.
     • ULIP – Insurance + investment linked to market funds.
   - Example: Term plan premium ₹12,000/year for ₹1 crore sum assured.

3. **Motor Insurance**
   - Mandatory by law for all vehicles.
   - Types:
     • Third-Party Liability – Covers damage to other people/property.
     • Comprehensive – Covers own vehicle + third-party liability.
   - Add-ons: Zero depreciation cover, roadside assistance, engine protection.

4. **Travel Insurance**
   - Covers medical emergencies, baggage loss, trip cancellation during travel.
   - Usually short-term; required for visa in many countries.

5. **General Insurance vs Life Insurance**
   - General: Covers assets (health, car, home, travel), usually 1-year renewable.
   - Life: Long-term protection; payout happens on death/maturity.

6. **Premiums**
   - Depends on age, coverage amount, type of plan, health history.
   - Paid monthly, quarterly, or annually.

7. **Claim Settlement Ratio (CSR)**
   - Important metric to choose insurer.
   - CSR = (Claims settled ÷ Total claims received) × 100.
   - Higher ratio → more reliable insurer.

8. **Key Insurance FAQs**
   - What is waiting period? → Initial time (e.g., 30 days or 2–4 years for pre-existing conditions).
   - Can I have multiple policies? → Yes, claims can be made proportionally.
   - What is co-pay? → Percentage of bill that policyholder must bear.
   - Is tax benefit available? → Yes, premiums eligible for tax deduction under Section 80C (life) and 80D (health).
"""


import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
@st.cache_resource(show_spinner=True)


def build_vectorstore(_salary_doc, _insurance_doc):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents([_salary_doc, _insurance_doc])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embeddings)
    return vs.as_retriever(search_kwargs={"k": 4})


if "retriever" not in st.session_state or build_btn:
    
    if salary_file is None and not use_samples:
        st.error("Please upload salary.txt or enable 'Use sample content'.")
        st.stop()
    if insurance_file is None and not use_samples:
        st.error("Please upload insurance.txt or enable 'Use sample content'.")
        st.stop()

    salary_doc = read_txt_file(salary_file, SAMPLE_SALARY, "salary.txt")
    insurance_doc = read_txt_file(insurance_file, SAMPLE_INSURANCE, "insurance.txt")

    with st.spinner("Building vector store..."):
        st.session_state.retriever = build_vectorstore(api_key, salary_doc, insurance_doc)
        st.success("Vector store ready ✅")

retriever = st.session_state.get("retriever", None)
if retriever is None:
    st.error("Retriever not initialized. Click 'Build / Rebuild Vector Store' in the sidebar.")
    st.stop()

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

def classify_query(query: str) -> str:
    system = (
        "You are a router. Read the user's question and answer with one word: "
        "'salary' if it is about salary/pay/deductions/CTC; "
        "'insurance' if it is about insurance/policy/premium/coverage/claims. "
        "If unclear, guess the most likely."
    )
    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": query}])
    label = (resp.content or "").strip().lower()
    return "salary" if "salary" in label else "insurance"

def coordinator(query: str):
    label = classify_query(query)
    if label == "salary":
        out = salary_agent.invoke({"query": query})
        return "Salary Agent", out
    else:
        out = insurance_agent.invoke({"query": query})
        return "Insurance Agent", out


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about **salary** or **insurance**. "
                                         "Try: 'How do I calculate annual salary?' or 'What is included in my insurance policy?'."}
    ]

with st.sidebar:
    st.subheader("🧪 Quick prompts")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Annual salary?"):
            st.session_state.messages.append({"role": "user", "content": "How do I calculate annual salary?"})
    with col2:
        if st.button("Policy coverage?"):
            st.session_state.messages.append({"role": "user", "content": "What is included in my insurance policy?"})


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
