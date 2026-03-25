import os
import streamlit as st
import time
from typing import Any, List, Optional

import requests
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from youtube_transcript_api import YouTubeTranscriptApi

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Gemini – default, works with GEMINI_API_KEY from https://aistudio.google.com/apikey
# ---------------------------------------------------------------------------
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-2.5-flash"


class GeminiLLM(LLM):
    """LLM via Google Gemini API. Set GEMINI_API_KEY in .env."""

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
            return "Error: Set GEMINI_API_KEY in .env (get one at https://aistudio.google.com/apikey)"

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
                return "Rate limit. Please wait a moment and retry."
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


# ---------------------------------------------------------------------------
# Groq (free tier) – optional, works with GROQ_API_KEY from https://console.groq.com
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


class GroqLLM(LLM):
    """LLM via Groq API (free tier). Set GROQ_API_KEY in .env."""

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
            return "Error: Set GROQ_API_KEY in .env (get one at https://console.groq.com)"

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
                return "Rate limit. Please wait a moment and retry."
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


# ---------------------------------------------------------------------------
# Hugging Face – use only if you have a custom Inference Endpoint URL
# Set HF_INFERENCE_URL in .env to your endpoint (e.g. from HF Inference Endpoints).
# Public serverless (api-inference / router) is deprecated or returns 404.
# ---------------------------------------------------------------------------
def _hf_inference_url(model: str) -> str:
    base = os.environ.get("HF_INFERENCE_URL", "").rstrip("/")
    if base:
        return base  # custom Inference Endpoint URL (single model)
    return f"https://router.huggingface.co/hf-inference/models/{model}"


class HuggingFaceInferenceLLM(LLM):
    """LLM using Hugging Face (custom HF_INFERENCE_URL or router)."""

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
                return "This model/endpoint is not available. Use Groq (free) or set HF_INFERENCE_URL to your own Inference Endpoint."
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


# ---------------------------------------------------------------------------
# App: choose Gemini (default), Groq, or Hugging Face
# ---------------------------------------------------------------------------
LLM_OPTIONS = {
    "Gemini (Google)": "gemini",
    "Groq (free – Llama 3.1 8B)": "groq",
    "Hugging Face (Flan-T5 Large)": "hf",
}
HF_MODELS = {
    "Google Flan-T5 Large": "google/flan-t5-large",
    "Google Flan-T5 Base": "google/flan-t5-base",
}

st.title("AI Research Assistant 📈")
st.sidebar.title("News Article URLs")

llm_choice = st.sidebar.selectbox(
    "LLM",
    options=list(LLM_OPTIONS.keys()),
    index=0,
)
provider = LLM_OPTIONS[llm_choice]

if provider == "gemini":
    llm = GeminiLLM(max_tokens=512, temperature=0.3)
elif provider == "groq":
    llm = GroqLLM(max_tokens=512, temperature=0.3)
else:
    hf_label = st.sidebar.selectbox("HF model", list(HF_MODELS.keys()), index=0)
    repo_id = HF_MODELS[hf_label]
    llm = HuggingFaceInferenceLLM(repo_id=repo_id, max_new_tokens=256, temperature=0.3)

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
yt_url = st.sidebar.text_input("YouTube URL")

process_clicked = st.sidebar.button("Process Data")
persist_directory = "chroma_store_openai"

main_placeholder = st.empty()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if process_clicked:
    documents = []
    
    urls_to_load = [u for u in urls if u.strip()]
    if urls_to_load:
        main_placeholder.text("Loading URLs...✅")
        loader = WebBaseLoader(urls_to_load)
        documents.extend(loader.load())
        
    if uploaded_file:
        main_placeholder.text("Loading PDF...✅")
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        pdf_loader = PyPDFLoader("temp.pdf")
        documents.extend(pdf_loader.load())
        
    if yt_url:
        main_placeholder.text("Loading YouTube Transcript...✅")
        try:
            if "v=" in yt_url:
                video_id = yt_url.split("v=")[-1].split("&")[0]
            else:
                video_id = yt_url.split("/")[-1].split("?")[0]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t["text"] for t in transcript])
            documents.extend([Document(page_content=text, metadata={"source": yt_url})])
        except Exception as e:
            st.sidebar.error(f"Error loading YouTube transcript: {e}")

    if not documents:
        main_placeholder.text("Please provide at least one source (URL, PDF, or YouTube).")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(documents)
    if not docs:
        main_placeholder.text(
            "No content could be extracted from the provided sources. "
            "Please check the inputs and try again."
        )
        st.stop()

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
            memory=st.session_state.memory
        )
        result = chain({"question": query})
        
        st.header("Answer")
        st.write(result["answer"])

        # Get past history excluding the current turn (last 2 messages: human and ai)
        history_messages = st.session_state.memory.chat_memory.messages[:-2]
        if history_messages:
            st.subheader("Past Chat History")
            for msg in history_messages[-10:]:  # show up to 5 previous turns
                role = "User" if msg.type == "human" else "Assistant"
                st.write(f"**{role}:** {msg.content}")
