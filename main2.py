import os
import shutil
import requests
import tempfile
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
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="AI Research & News Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("AI Research & News Analyst 📈")
st.caption(
    "Analyze articles, PDFs, and YouTube transcripts with trust scoring, "
    "impact detection, and multi-source consensus."
)

# -----------------------------
# Constants
# -----------------------------
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-2.5-flash"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

PERSIST_DIRECTORY = "chroma_store"

# -----------------------------
# Custom LLM Classes
# -----------------------------
class GeminiLLM(LLM):
    model: str = GEMINI_MODEL
    max_tokens: int = 700
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
            return "Error: GEMINI_API_KEY not found in .env"

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
            return f"Gemini API error: {e}"
        except Exception as e:
            return f"Gemini request failed: {e}"

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
    max_tokens: int = 700
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
            return "Error: GROQ_API_KEY not found in .env"

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
            return f"Groq API error: {e}"
        except Exception as e:
            return f"Groq request failed: {e}"

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            text = str(data)

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0].strip()
        return text


# -----------------------------
# Sidebar - LLM Selection
# -----------------------------
st.sidebar.title("📚 Input Sources")

LLM_OPTIONS = {
    "Gemini (Google)": "gemini",
    "Groq (free – Llama 3.1 8B)": "groq",
}

llm_choice = st.sidebar.selectbox("Choose LLM", list(LLM_OPTIONS.keys()), index=0)
provider = LLM_OPTIONS[llm_choice]

if provider == "gemini":
    llm = GeminiLLM(max_tokens=700, temperature=0.3)
else:
    llm = GroqLLM(max_tokens=700, temperature=0.3)

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
reset_clicked = st.sidebar.button("🗑 Reset Knowledge Base")

# -----------------------------
# Reset Vector DB
# -----------------------------
if reset_clicked:
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    st.session_state.pop("memory", None)
    st.success("Knowledge base reset successfully.")
    st.stop()

# -----------------------------
# Embeddings + Memory
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# -----------------------------
# Prompt Template
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
# Helper Functions
# -----------------------------
def extract_youtube_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url.split("/")[-1].split("?")[0]


def fetch_youtube_transcript(yt_url: str) -> str:
    video_id = extract_youtube_video_id(yt_url)

    try:
        # Newer versions support list_transcripts
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None

            try:
                transcript = transcript_list.find_transcript(["en"])
            except:
                pass

            if transcript is None:
                try:
                    transcript = transcript_list.find_generated_transcript(["en"])
                except:
                    pass

            if transcript is None:
                raise Exception("No English transcript available for this video.")

            transcript_data = transcript.fetch()

        else:
            # Older versions fallback
            transcript_data = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=["en"]
            )

        yt_text = " ".join([item["text"] for item in transcript_data if "text" in item])

        if not yt_text.strip():
            raise Exception("Transcript fetched but empty.")

        return yt_text

    except Exception as e:
        raise Exception(f"Transcript fetch failed: {e}")


def build_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    if not docs:
        raise ValueError("No valid text chunks could be created from the sources.")

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    vectorstore.persist()
    return vectorstore


# -----------------------------
# Data Processing
# -----------------------------
if process_clicked:
    documents = []

    try:
        # -------- URLs --------
        urls_to_load = [u.strip() for u in urls if u.strip()]
        if urls_to_load:
            with st.spinner("Loading URLs..."):
                try:
                    loader = WebBaseLoader(urls_to_load)
                    documents.extend(loader.load())
                    st.success("✅ URLs loaded successfully!")
                except Exception as e:
                    st.warning(f"⚠️ URL loading failed: {e}")

        # -------- PDF --------
        if uploaded_file:
            with st.spinner("Loading PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_pdf_path = tmp_file.name

                    pdf_loader = PyPDFLoader(temp_pdf_path)
                    documents.extend(pdf_loader.load())

                    os.remove(temp_pdf_path)
                    st.success("✅ PDF loaded successfully!")
                except Exception as e:
                    st.warning(f"⚠️ PDF loading failed: {e}")

        # -------- YouTube --------
        if yt_url.strip():
            with st.spinner("Loading YouTube transcript..."):
                try:
                    yt_text = fetch_youtube_transcript(yt_url)
                    documents.append(
                        Document(
                            page_content=yt_text,
                            metadata={"source": yt_url}
                        )
                    )
                    st.success("✅ YouTube transcript loaded successfully!")
                except Exception as e:
                    st.warning(f"⚠️ Could not fetch YouTube transcript: {e}")

        # -------- Validation --------
        if not documents:
            st.warning("Please provide at least one valid source (URL, PDF, or YouTube).")
            st.stop()

        # -------- Build Vectorstore --------
        with st.spinner("Building knowledge base..."):
            build_vectorstore(documents)

        st.success("✅ Data processed successfully! You can now ask questions.")

    except Exception as e:
        st.error(f"Error while processing data: {e}")

# -----------------------------
# Query Section
# -----------------------------
query = st.text_input("🔎 Ask your research question:")

if query:
    if os.path.exists(PERSIST_DIRECTORY):
        try:
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

            with st.spinner("Analyzing sources..."):
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

        except Exception as e:
            st.error(f"Error while answering question: {e}")
    else:
        st.warning("Please process your data first before asking a question.")