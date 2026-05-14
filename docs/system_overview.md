# System Overview & Setup

## Table of Contents

1. [What NeoFinance Is](#1-what-neofinance-is)
2. [Key Components](#2-key-components)
3. [Setup & Configuration](#3-setup--configuration)
4. [How the Agentic System Works](#4-how-the-agentic-system-works)

---

## 1. What NeoFinance Is

NeoFinance is an AI-powered financial assistant. Users interact with it through a chat interface (text or voice). Behind the scenes, a large language model (Google Gemini) decides — on its own — which Python functions to call based on what the user is asking. Those functions fetch real data, run predictions, and return results. Gemini then formats those results into a natural, conversational reply.

This design pattern — where the LLM drives which tools to use and when — is called **agentic AI** or **function calling**.

---

## 2. Key Components

| Component | Role |
|-----------|------|
| Gradio | Web UI — chat interface and microphone input |
| Google Gemini 1.5 Pro | LLM that understands requests and calls tools |
| `tools.py` | 15 Python functions exposed as tools to Gemini |
| `stock_prediction_transformer.py` | PyTorch transformer for daily stock prediction |
| `stock_prediction_transformer_intraday.py` | PyTorch transformer for intraday prediction |
| Yahoo Finance (`yfinance`) | Source of historical daily stock data |
| Alpha Vantage API | Intraday data, forex, crypto, news sentiment |
| OpenAI Whisper | Speech-to-text for voice input |
| Cloudinary | Hosts prediction chart images |

---

## 3. Setup & Configuration

### Prerequisites

- Python 3.9+
- A CUDA-capable GPU is optional but strongly recommended for training models. CPU is used as a fallback.

### Installation

```bash
git clone https://github.com/Ruhan-Saad-Dave/Neo.git
cd Neo

# Create and activate a virtual environment
# Windows (CMD):
python -m venv myenv
myenv\Scripts\activate

# macOS / Linux:
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### API Keys

Create a `.env` file in the project root with the following keys. **Never commit this file.**

```
GEMINI_API_KEY="..."
ALPHA_API_KEY="..."
CLOUDINARY_CLOUD_NAME="..."
CLOUDINARY_API_KEY="..."
CLOUDINARY_API_SECRET="..."
```

Where to get each key:

| Key | Source |
|-----|--------|
| `GEMINI_API_KEY` | [Google AI Studio](https://makersuite.google.com/app/apikey) |
| `ALPHA_API_KEY` | [Alpha Vantage](https://www.alphavantage.co/support/#api-key) — free tier: 25 requests/day |
| Cloudinary keys | [Cloudinary Console](https://console.cloudinary.com/pm) → Settings → API Keys |

### Running the App

```bash
# Windows
python app.py

# macOS / Linux
python3 app.py
```

Gradio will print a local URL (e.g. `http://127.0.0.1:7860`) and a temporary public share URL. Open either in a browser to use the chat interface.

### Pre-trained Models

Pre-trained `.pt` model files for AAPL, NKE, NVDA, and HDFCBANK.NS are stored in the `models/` directory (tracked via Git LFS). For any other stock, the system will automatically download data and train a new model on first use — this can take up to 2 hours on CPU.

### Configuration File

`config.json` is auto-generated on first run. It stores metadata for each trained stock and the transformer hyperparameters used across all models.

```json
{
  "stocks": {
    "AAPL": {
      "added_date": "2025-03-31",
      "last_trained": "2025-04-06",
      "accuracy": 0.963,
      "model_path": "models/AAPL_model.pt"
    }
  },
  "transformer_config": {
    "d_model": 128,
    "nhead": 8,
    "num_encoder_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.2,
    "learning_rate": 0.0001,
    "epochs": 100
  }
}
```

---

## 4. How the Agentic System Works

### The Function Calling Loop

```
User message
     │
     ▼
 Gradio UI
     │
     ▼
Gemini 1.5 Pro
  (reads message + tool schemas)
     │
     ├─ If answer needs data ──► Selects tool(s) to call
     │                                   │
     │                                   ▼
     │                          Python function in tools.py
     │                          (fetches data / runs model)
     │                                   │
     │                          Returns result to Gemini
     │                                   │
     ▼                                   │
Gemini formats result into               │
natural language reply  ◄────────────────┘
     │
     ▼
Gradio streams response back to UI
```

Gemini never sees raw code — it only sees the function **name**, **description**, and **parameter types** declared in each tool's docstring. It decides entirely on its own which tool fits the user's request, what arguments to pass, and whether to chain multiple tool calls.

### Per-Session Chat State

Each browser tab gets its own isolated Gemini chat session via `gr.State`. This means two users talking simultaneously never share conversation history. A new session is created lazily on the user's first message.

### Voice Input Flow

```
Microphone recording (Gradio Audio)
          │
          ▼
   Whisper (openai/whisper-small)
   transcribes audio to text
          │
          ▼
   Transcribed text enters the
   same chat pipeline as typed input
```

### System Prompt Behaviour

Gemini is given a system instruction that:
- Constrains it to finance topics only
- Instructs it to format numerical data as tables
- Instructs it to detect the user's language, translate the request internally to call the right tool, then translate the response back
- Instructs it to never ignore or bypass the system instruction, even if asked to
