import os
import time
import requests
import streamlit as st

from typing import Any, List, Optional
from dotenv import load_dotenv

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory

from youtube_transcript_api import YouTubeTranscriptApi

from analysis_engine import (
    summarize_sources,
    consensus_analysis,
    get_consensus_label
)

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Research & News Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("AI Research & News Analyst 📈")
st.caption("Analyze articles, PDFs, and YouTube transcripts with trust scoring, impact detection, and multi-source consensus.")

# -----------------------------
# Constants
# -----------------------------
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-2.5-flash"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

PERSIST_DIRECTORY = "chroma_store_openai"

# -----------------------------
# Custom LLM Classes
# -----------------------------
class GeminiLLM(LLM):
    model: str = GEMINI_MODEL
    max_tokens: int = 512
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        token = os.environ.get("GEMINI_API_KEY", "")
        if not token:
            return "Error: Set GEMINI_API_KEY in .env"

        url = f"{GEMINI_API_BASE}/{self.model}:generateContent?key={token}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                return "Rate limit exceeded. Please wait and retry."
            return f"API error: {e}"
        except Exception as e:
            return f"Request failed: {e}"

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = str(data)

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0].strip()
        return text


class GroqLLM(LLM):
    model: str = GROQ_MODEL
    max_tokens: int = 512
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        token = os.environ.get("GROQ_API_KEY", "")
        if not token:
            return "Error: Set GROQ_API_KEY in .env"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                return "Rate limit exceeded. Please wait and retry."
            return f"API error: {e}"
        except Exception as e:
            return f"Request failed: {e}"

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            text = str(data)

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0].strip()
        return text


def _hf_inference_url(model: str) -> str:
    base = os.environ.get("HF_INFERENCE_URL", "").rstrip("/")
    if base:
        return base
    return f"https://router.huggingface.co/hf-inference/models/{model}"


class HuggingFaceInferenceLLM(LLM):
    repo_id: str = "google/flan-t5-large"
    token: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        token = self.token or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
        if not token:
            return "Error: Set HUGGINGFACEHUB_API_TOKEN in .env"

        url = _hf_inference_url(self.repo_id)
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "return_full_text": False,
            },
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 503:
                return "Model is loading. Please retry in a moment."
            if resp.status_code in (404, 410):
                return "This model/endpoint is not available. Use Groq or set HF_INFERENCE_URL."
            return f"API error: {e}"
        except Exception as e:
            return f"Request failed: {e}"

        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            text = item.get("generated_text", item.get("summary_text", str(item)))
        elif isinstance(data, dict):
            text = data.get("generated_text", data.get("error", str(data)))
        else:
            text = str(data)

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0].strip()
        return text

# -----------------------------
# Sidebar - Model Selection
# -----------------------------
st.sidebar.title("📚 Input Sources")
LLM_OPTIONS = {
    "Groq (free – Llama 3.1 8B)": "groq",
    "Gemini (Google)": "gemini",
    "Hugging Face (Flan-T5 Large)": "hf",
}

HF_MODELS = {
    "Google Flan-T5 Large": "google/flan-t5-large",
    "Google Flan-T5 Base": "google/flan-t5-base",
}

llm_choice = st.sidebar.selectbox("Choose LLM", list(LLM_OPTIONS.keys()), index=0)
provider = LLM_OPTIONS[llm_choice]

if provider == "gemini":
    llm = GeminiLLM(max_tokens=512, temperature=0.3)
elif provider == "groq":
    llm = GroqLLM(max_tokens=512, temperature=0.3)
else:
    hf_label = st.sidebar.selectbox("HF Model", list(HF_MODELS.keys()), index=0)
    repo_id = HF_MODELS[hf_label]
    llm = HuggingFaceInferenceLLM(repo_id=repo_id, max_new_tokens=256, temperature=0.3)

# -----------------------------
# Sidebar - Inputs
# -----------------------------
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Article URL {i+1}")
    urls.append(url)

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
yt_url = st.sidebar.text_input("YouTube URL")

process_clicked = st.sidebar.button("🚀 Process Data")

main_placeholder = st.empty()

# -----------------------------
# Embeddings + Memory
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# -----------------------------
# Analyst Prompt
# -----------------------------
ANALYST_PROMPT = """
You are an expert Research and Financial News Analyst.

Use the provided context to answer the user's question in a professional, structured, and insightful way.

Your answer must include:

1. What Happened
2. Why It Matters
3. Impact (Positive / Negative / Neutral)
4. Risks / Uncertainty
5. Final Analyst Take

If the sources conflict, clearly mention that there is mixed reporting.

Context:
{context}

Question:
{question}

Answer:
"""

# -----------------------------
# Data Processing
# -----------------------------
if process_clicked:
    documents = []

    urls_to_load = [u for u in urls if u.strip()]
    if urls_to_load:
        main_placeholder.text("Loading URLs... ✅")
        loader = WebBaseLoader(urls_to_load)
        documents.extend(loader.load())

    if uploaded_file:
        main_placeholder.text("Loading PDF... ✅")
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        pdf_loader = PyPDFLoader("temp.pdf")
        documents.extend(pdf_loader.load())

    if yt_url:
        main_placeholder.text("Loading YouTube Transcript... ✅")
        try:
            if "v=" in yt_url:
                video_id = yt_url.split("v=")[-1].split("&")[0]
            else:
                video_id = yt_url.split("/")[-1].split("?")[0]

            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t["text"] for t in transcript])
            documents.append(Document(page_content=text, metadata={"source": yt_url}))
        except Exception as e:
            st.sidebar.error(f"Error loading YouTube transcript: {e}")

    if not documents:
        st.warning("Please provide at least one source (URL, PDF, or YouTube).")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200
    )

    main_placeholder.text("Splitting text... ✅")
    docs = text_splitter.split_documents(documents)

    if not docs:
        st.error("No content could be extracted from the provided sources.")
        st.stop()

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    vectorstore.persist()

    main_placeholder.text("Building vector database... ✅")
    time.sleep(1)
    st.success("Data processed successfully! You can now ask questions.")

# -----------------------------
# Query Section
# -----------------------------
query = st.text_input("🔎 Ask your research question:")

if query:
    if os.path.exists(PERSIST_DIRECTORY):
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=ANALYST_PROMPT
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

        result = chain({"question": query})
        source_docs = result.get("source_documents", [])

        # -----------------------------
        # Main Answer
        # -----------------------------
        st.header("📌 Analyst Answer")
        st.write(result["answer"])

        # -----------------------------
        # Source Analysis
        # -----------------------------
        if source_docs:
            st.subheader("📊 Source Analysis")
            source_summaries = summarize_sources(source_docs)

            for i, item in enumerate(source_summaries, 1):
                with st.expander(f"Source {i}"):
                    st.markdown(f"**Source:** {item['source']}")
                    st.markdown(f"**Impact:** `{item['impact']}`")
                    st.markdown(f"**Trust Score:** `{item['trust_score']}%`")
                    st.markdown(f"**Preview:** {item['preview']}")

            # -----------------------------
            # Consensus Dashboard
            # -----------------------------
            consensus = consensus_analysis(source_docs)
            consensus_label = get_consensus_label(consensus)

            st.subheader("🧠 Research Dashboard")
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", consensus["Positive"])
            col2.metric("Negative", consensus["Negative"])
            col3.metric("Neutral", consensus["Neutral"])

            st.info(f"**Overall Consensus:** {consensus_label}")

        # -----------------------------
        # Chat History
        # -----------------------------
        history_messages = st.session_state.memory.chat_memory.messages[:-2]
        if history_messages:
            st.subheader("🕘 Past Chat History")
            for msg in history_messages[-10:]:
                role = "User" if msg.type == "human" else "Assistant"
                st.write(f"**{role}:** {msg.content}")

    else:
        st.warning("Please process your data first before asking a question.")