# rag-chatbot by Manav & Mustafa

# 🏋️ Fitness RAG Chatbot (Groq + Streamlit)

A simple AI chatbot for answering fitness, nutrition, and wellness questions — powered by Retrieval-Augmented Generation (RAG), Groq LLMs, FAISS, and Streamlit. No documents or files required — all knowledge is built-in.

---

## 🚀 Features

- 🤖 Uses Groq's LLaMA 3 model for generation
- 🧠 Local embeddings via `sentence-transformers`
- 🔍 Fast retrieval using FAISS
- 🧾 No file setup — just run it and chat
- 💻 Simple Streamlit UI

---

## 📦 Installation

1. **Clone the repo**


git clone https://github.com/Mustafaahmed00/rag-chatbot
cd yourrepo

pip install -r requirements.txt

Set your Groq API key with 
export GROQ_API_KEY="your-groq-key"  # for  Mac/Linux

# OR (Windows PowerShell)
$env:GROQ_API_KEY="your-groq-key"

Thats all and hit below command:
streamlit run chatbot.py\


Thank you
