# 🎭 Reddit User Persona Analyzer

This project scrapes a Reddit user's posts and comments and generates a **detailed behavioral persona** using advanced large language models (LLMs) via Together.ai. It includes:

- 🔍 Reddit data scraping (posts + comments)
- 🤖 Persona generation using open-source LLMs (Llama 3, Mistral, etc.)
- 🖥️ CLI and Streamlit interfaces
- 📁 Output saved with metadata and citation-backed insights

---

## 🚀 Features

- 🔗 Accepts both Reddit URLs and usernames
- 📊 Collects up to 300 comments/posts
- 🤖 Supports multiple AI models (Llama 3, Mistral, Mixtral)
- 📎 Evidence-backed personality insights
- 💬 Generates clean markdown-style persona profiles
- 🌐 Streamlit web UI + CLI support

---

## 🔧 Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/reddit-persona-analyzer.git
cd reddit-persona-analyzer
```

### 2. Create a `.env` file

Create a `.env` file in the root directory and paste your credentials:

```ini
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
TOGETHER_API_KEY=your_together_ai_key
```

🔐 **Get your keys from:**
- Reddit: https://www.reddit.com/prefs/apps
- Together.ai: https://platform.together.xyz

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 How to Use

### ▶️ Option 1: Web UI (Streamlit)

```bash
streamlit run app.py --streamlit
```

Then:
1. Paste a Reddit profile link or username
2. Choose model (e.g., Llama 3 70B for best quality)
3. Click **"Generate Persona"**
4. Download the persona file

### 💻 Option 2: CLI Mode

```bash
python app.py --profile https://reddit.com/user/kojied --model meta-llama/Llama-3-70b-chat-hf --limit 100
```

This will:
- Scrape the profile
- Generate a persona with LLM
- Save to `personas/username_persona_TIMESTAMP.txt`

---

## 📎 Sample Input (Reddit URLs or Usernames)

You can input:
- `https://reddit.com/user/kojied`
- `https://old.reddit.com/u/Hungry-Move-6603`
- Or just: `spez`

---

## 📁 Output Format

Persona files are saved like:

```
personas/
└── kojied_persona_2025-07-15_112340.txt
```

Each persona includes:
- Core interests
- Personality traits
- Communication tone
- Values & beliefs
- Reddit activity behavior
- All points backed by quoted Reddit content

---

## 🧠 Models Supported (via Together.ai)

| Model Name | Description |
|------------|-------------|
| `meta-llama/Llama-3-70b-chat-hf` | 🔝 Llama 3 70B - Most detailed |
| `mistralai/Mistral-7B-Instruct-v0.1` | ⚡ Mistral 7B - Fast + open |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 🔁 Mixtral - Balanced |
| `meta-llama/Llama-2-70b-chat-hf` | 🔒 Llama 2 - Stable |

💡 You can test API connection from the Streamlit sidebar.

---

## 📦 Dependencies

See `requirements.txt` for full list. Key libraries:
- `praw`
- `transformers`
- `streamlit`
- `requests`
- `dotenv`

### requirements.txt

```txt
praw
requests
streamlit
transformers
accelerate
sentencepiece
python-dotenv
bitsandbytes
```

---

## 📬 Contact

Made with ❤️ by **Dhruti** for Generative AI Internship Assignment

Feel free to fork, submit issues, or build on top!
