
# UMT Admission Chatbot

![Failed to load image](/screenshot.png)

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, **OpenAI**, and **Pinecone**. It provides an interactive interface for users to ask questions about admissions at the University of Management and Technology (UMT). The chatbot retrieves relevant information from a vector database (Pinecone) containing FAQs and program details, then generates concise, accurate responses using GPT-4o-mini.

### Key features:

- **RAG Pipeline**: Combines semantic search over FAQs and program data with LLM generation.  
- **Multi-Namespace Retrieval**: Searches across separate Pinecone namespaces for FAQs and programs.  
- **User-Friendly UI**: Streamlit-based chat interface with custom styling (lime-green assistant responses).  
- **Data Ingestion Scripts**: Tools to load and embed data from JSON (FAQs) and CSV (programs) into Pinecone.

The app requires an **OpenAI API key** (for embeddings and LLM) and **Pinecone credentials** (via `.env`).

---

## Project Structure
```
├── streamlit_app.py          # Main Streamlit chatbot app
├── data_upsertion.py         # Retrieval module for querying Pinecone namespaces
├── umt_faqs_insertion.py     # Script to ingest FAQs from JSON into Pinecone
├── program_faqs_insertion.py # Script to ingest program data from CSV into Pinecone
├── umt_faqs.json             # Sample FAQ data (JSON format)
├── programs_faq.csv          # Sample program data (CSV format)
├── .env                      # Environment variables (API keys)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── screenshot.png            # App screenshot
```

## Prerequisites

- Python 3.10+
- OpenAI API key (for GPT-4o-mini and embeddings)
- Pinecone account and API key (for vector store)
- Streamlit installed

## Setup Instructions

### Clone or Download the Project:
```bash
git clone <your-repo-url>
cd umt-admission-chatbot
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Set Up Environment Variables:
Create a `.env` file in the root directory:
```text
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENV=your_pinecone_environment_here  # e.g., us-west1-gcp
```

### Prepare Data Files:

- `umt_faqs.json`: JSON array of FAQs with keys like question, answer, category.
- `programs_faq.csv`: CSV with columns like School, Program Name, Duration, Tentative Investment (PKR), Quarterly Installment (PKR).

### Ingest Data into Pinecone:
Run the ingestion scripts to embed and upload data:
```bash
python umt_faqs_insertion.py  # Loads FAQs into "umt-faqs-namespace"
python program_faqs_insertion.py  # Loads programs into "umt-programs-namespace"
```
This uses the Pinecone index named `irfan-gpt-index`. Create it in your Pinecone dashboard if it doesn't exist (use `text-embedding-3-small` dimensions: 1536). Each document gets a unique UUID and metadata for filtering.

### Run the Chatbot:
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. Enter your OpenAI API key in the sidebar. Start chatting! Example queries: "What are the admission requirements?" or "Tell me about BS Computer Science fees."

## Workflow

### 1. Data Ingestion
- **FAQs (JSON → Pinecone)**: `umt_faqs_insertion.py` loads `umt_faqs.json`, creates LangChain `Document` objects (answer as content, question/category as metadata), embeds with OpenAI's `text-embedding-3-small`, and upserts into Pinecone namespace `umt-faqs-namespace`.  
- **Programs (CSV → Pinecone)**: `program_faqs_insertion.py` loads `programs_faq.csv`, combines fields into content (e.g., "School: XYZ
Program Name: BS CS..."), embeds, and upserts into `umt-programs-namespace`.  
- **Output**: Vector store with ~100-500 vectors (depending on data size), searchable by semantic similarity.

### 2. Retrieval
`data_upsertion.py` defines `get_faqs_answers(query)`:

- Initializes Pinecone vector store with OpenAI embeddings.  
- Creates retrievers for each namespace (k=5 top matches).  
- Queries both namespaces in parallel and combines results (up to 10 docs total).  
- Returns raw `Document` objects with content and metadata.

### 3. Generation (Chatbot)
`streamlit_app.py`:

- **UI Setup**: Configures page with UMT logo, sticky header, sidebar for API key, and main image.  
- **Chat Loop**:  
  - User inputs query via `st.chat_input`.  
  - Retrieves chunks: `chunks = get_faqs_answers(prompt)`.  
  - Formats prompt template with retrieved content: `"Use only the data provided... If not in data: 'I don't know... visit https://admissions.umt.edu.pk'"`.  
  - Invokes `ChatOpenAI(model="gpt-4o-mini")` for response.  
  - Displays with lime-green background for assistant messages; maintains session state for history.

- **Error Handling**: Checks API key; catches exceptions (e.g., invalid key).  
- **Styling**: Custom HTML for lime-green assistant bubbles; responsive layout.

### 4. Response Guidelines
- Responses are concise, friendly, and in the user's language.  
- Sources only from retrieved data; fallback to UMT website if no match.  
- No external knowledge; ensures accuracy.

### Diagram: High-Level Workflow
```text
[User Query] → Streamlit Chat Input
              ↓
[Retrieval] → data_upsertion.py → Pinecone (Query "umt-faqs-namespace" + "umt-programs-namespace")
              ↓ (Top 5 docs each → Combined chunks)
[Augmented Prompt] → GPT-4o-mini (via LangChain)
              ↓
[Response] → Streamlit Chat Output (Lime-Green Bubble)
```

## Customization
- Add More Data: Extend JSON/CSV and re-run ingestion scripts.  
- Change Model: Update `model="gpt-4o-mini"` in app or embeddings.  
- Index Name: Edit `index_name = "irfan-gpt-index"` in scripts.  
- Retrieval Params: Adjust `k=5` in `data_upsertion.py` for more/fewer results.  
- UI Tweaks: Modify HTML in `streamlit_app.py` for colors/images.

## Troubleshooting
- Pinecone Connection Error: Verify API key/env in `.env`; ensure index exists.  
- OpenAI Rate Limit: Use a valid API key; monitor usage.  
- No Relevant Docs: Check data ingestion logs; ensure embeddings match.  
- Streamlit Issues: Run `streamlit hello` to test installation.

## Contributing
Fork the repo, add features (e.g., more data sources), and submit a PR. Ensure data privacy and API key security.

## License
MIT License - Feel free to use/modify.

Powered by Streamlit, LangChain, OpenAI, and Pinecone.
