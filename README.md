# 🤖 AI Chat Assistant with Document Q&A (RAG)

A Streamlit chatbot built with LangChain and OpenRouter. Chat with an AI assistant, or upload a PDF and get answers grounded in its content, with page references.

**🔗 Live demo:** https://ai-chatbot-fpr56p8rbanxsskhakmykg.streamlit.app

## ✨ Features

- **Conversational AI:** streaming replies with chat memory, so follow-up questions work
- **Document Assistant (RAG):** upload a PDF and ask questions about it
- **Source references:** every document answer shows the pages and text it was based on
- **Personalities:** switch between Helpful assistant, Friendly tutor, Funny buddy and Short & direct
- **Adjustable settings:** model, creativity (temperature), max tokens and number of chunks retrieved
- **Suggestion buttons:** one-click starter questions for both chat and document mode
- **Custom UI:** dark gradient theme, animated chat bubbles, typing indicator and hover effects

## 🧠 How the Document Assistant works (RAG)

1. **Load:** the PDF is read page by page.
2. **Split:** the text is cut into ~1000-character chunks with a small overlap.
3. **Embed:** each chunk is converted into a vector using a local embedding model (FastEmbed, `BAAI/bge-small-en-v1.5`).
4. **Retrieve:** when you ask a question, the most similar chunks are found with a vector search.
5. **Generate:** the chunks and your question are sent to the model, which is told to answer only from that context and cite page numbers.

## 🛠️ Tech stack

| Area | Tool |
|---|---|
| Interface | Streamlit |
| LLM framework | LangChain |
| Model provider | OpenRouter |
| Embeddings | FastEmbed (runs locally, no API key) |
| Vector store | LangChain `InMemoryVectorStore` |
| PDF parsing | pypdf |
| Hosting | Streamlit Community Cloud |

## 🚀 Run it locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/navya1951-crypto/AI-chatbot.git
   cd AI-chatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your OpenRouter API key.** Create a file named `.env` in the project folder:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```
   Get a key at [openrouter.ai](https://openrouter.ai). Never commit this file.

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment

The app is deployed on Streamlit Community Cloud. To deploy your own copy:

1. Push the project to GitHub (without your `.env` file).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from your repo with `app.py` as the main file.
3. In **Advanced settings → Secrets**, add:
   ```
   OPENROUTER_API_KEY = "your_key_here"
   ```

## 📁 Project structure

```
AI-chatbot
├── .streamlit
│   └── config.toml      # dark theme settings
├── app.py               # the full application
├── requirements.txt     # Python dependencies
└── .gitignore           # keeps .env and venv out of Git
```

## ⚠️ Notes and limitations

- Scanned PDFs (images only) have no text to read, so they need OCR, which isn't included.
- Uploaded documents are held in memory for the current session only and aren't stored.
- The default `openrouter/free` option picks from available free models, so replies may come from different models and vary in quality.
- Follow-up questions in document mode search using only your latest message, so very short follow-ups like "explain more" may retrieve weaker context.

## 🔮 Ideas for future improvements

- Support multiple PDFs and other file types (DOCX, TXT)
- Save the vector index to disk with Chroma or FAISS
- Rewrite follow-up questions before retrieval for better context
- Export conversations as PDF or text
- Voice input and spoken replies
- Side-by-side model comparison

## 👤 Author

Built by **Navya** ([@navya1951-crypto](https://github.com/navya1951-crypto)).
