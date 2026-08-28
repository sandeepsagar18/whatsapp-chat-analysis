
<div align="center">

# 💬 WhatsApp Chat Intelligence & Analytics Dashboard

**A full-featured, interactive analytics platform and NLP intelligence dashboard for exported WhatsApp conversations.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![NLTK VADER](https://img.shields.io/badge/NLTK-VADER%20NLP-green.svg?style=for-the-badge)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 🌟 Overview

**WhatsApp Chat Intelligence** transforms exported WhatsApp chat logs (`.txt`) into interactive, visually captivating dashboards. From deep conversational statistics and response delay metrics to **VADER sentiment tracking** and a fun **"WhatsApp Wrapped"** superlative award generator, this tool provides complete visibility into chat dynamics for both individual chats and group conversations.

---

## ✨ Key Features

### 🎁 1. "WhatsApp Wrapped" & Superlative Awards
- 👑 **The Grand Yapper**: Highest word count and longest conversations.
- ⚡ **Lightning Responder**: Fastest average reply speed.
- 👻 **The Certified Ghost**: Slowest replier in the group.
- 🌙 **Ultimate Night Owl**: Most messages sent between midnight and 5:00 AM.
- 🤣 **Chief Humor Officer**: Highest frequency of laughs (`haha`, `lol`, `😂`, `🤣`, `rofl`).
- 📸 **Media Mogul**: Most photos, stickers, and media shared.
- 🚀 **Conversation Igniter**: Initiates the most chats after long periods of silence.

### 🎭 2. Sentiment & Mood Analysis (VADER NLP)
- **Polarity Score Gauge**: Measures aggregate emotional tone from `-1.0` (Negative) to `+1.0` (Positive).
- **Sentiment Distribution Chart**: Percentage breakdown of Positive, Neutral, and Negative messages.
- **Participant Sentiment Ranking**: Compares positivity across participants.
- **Message-Level Polarity Inspector**: Inspect individual message scores and sentiment classifications.

### ⚡ 3. Interaction Dynamics & Response Times
- **Reply Delay Analytics**: Computes average and median response times in minutes.
- **Conversation Starters vs. Closers**: Tracks who breaks the silence ($> 4$ hours) versus who gets the last word.

### 📊 4. Interactive Timelines & Activity Patterns
- **Monthly Trend Volume**: Interactive area charts highlighting conversational growth over time.
- **Daily Message Timeline**: High-resolution line charts with range sliders, zooming, and pan controls.
- **Day of Week & Monthly Breakdown**: Identifies peak chatting days and seasonal patterns.
- **24-Hour Intensity Heatmap**: Identifies peak active chatting hours across each day of the week.

### 🔤 5. Words, Emojis & Domain Intelligence
- **Dark-Themed Word Cloud**: Clean word cloud filtered for both English and Hinglish stopwords.
- **Native Unicode Emoji Charts**: Interactive donut and bar charts rendering full Unicode emojis crisply (no missing `□` font boxes).
- **Shared Web Domains**: Extracts and ranks shared URLs (YouTube, Instagram, GitHub, Spotify, etc.).

### 🔍 6. Chat Explorer & CSV Export
- Real-time keyword search across all messages.
- Clean tabular message view with timestamp, sender, and content.
- One-click **Download Filtered Data as CSV** button.

---

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core application logic and data processing |
| **Streamlit** | Modern, responsive web frontend & UI |
| **Plotly** | High-performance interactive visualizations & charts |
| **NLTK (VADER)** | Natural Language Processing & Sentiment Analysis |
| **Pandas & NumPy** | Data manipulation and aggregation pipelines |
| **URLExtract** | Web link extraction and domain categorization |
| **Emoji** | Unicode emoji parsing and frequency distribution |
| **WordCloud & Matplotlib** | Text visualization and keyword clouds |

---

## 📂 Project Structure

```
whatsapp-chat-analysis/
│── app.py                 # Streamlit web application & UI dashboard
│── helper.py              # Statistical, sentiment, wrapped & plotting utilities
│── preprocessor.py        # Robust multi-format WhatsApp regex parser
│── requirements.txt       # Project dependencies
│── stop_hinglish.txt      # Stopwords for English & Hinglish cleaning
│── .gitignore             # Git ignore configuration
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sandeepsagar18/whatsapp-chat-analysis.git
cd whatsapp-chat-analysis
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📱 How to Export Your WhatsApp Chat

1. Open any individual or group chat on **WhatsApp**.
2. Tap the **Three Dots (⋮)** on Android or **Contact/Group Info** on iOS.
3. Select **More** $\rightarrow$ **Export Chat**.
4. Select **Without Media** (attaching the resulting `.txt` file into the web app).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/sandeepsagar18/whatsapp-chat-analysis/issues).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Sandeep Kumar**
- **GitHub**: [@sandeepsagar18](https://github.com/sandeepsagar18)
- **LinkedIn**: [Sandeep Kumar](https://www.linkedin.com/in/sandeep-kumar-2188722a7)



