🎨 Creative Text Generator — End-to-End NLP Project (PyTorch & Streamlit)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



A deployed deep learning application demonstrating creative text generation using PyTorch LSTMs, trained on a large, diverse corpus (1.5M sequences from song lyrics & news headlines) and served via an interactive Streamlit UI.
This project showcases the full lifecycle from data engineering and model training to deployment, including professional handling of large model files (>100MB) without Git LFS.

🚀 Live Demo

Interact with the deployed app here:
➡️ https://yourname-streamlit-app.streamlit.app

🖼️ Preview


Example: Screenshot or GIF of the Streamlit interface

Example Output (Top-K Sampling, k=5, T=0.8):

Seed: The city sleeps but
Generated: the city sleeps but i know that i'm not alone...

🎯 Objectives & Highlights

✅ End-to-End Pipeline: From raw CSVs → preprocessing → PyTorch LSTM → Streamlit app
✅ Large-Scale Data Handling (>50M tokens) with memory optimization
✅ 2-Layer LSTM (hidden_size=512) trained on 1.5M sequences
✅ No Git LFS: uses .gitignore + external hosting + download at runtime
✅ Interactive Streamlit UI with custom neumorphic dark theme

🧱 Architecture Overview
graph LR
    A[User Input] --> B[Streamlit UI - app.py]
    B --> C[Preprocessing - Tokenization]
    C --> D[PyTorch LSTM Model - model.py]
    D --> E[Top-K Sampling]
    E --> F[Generated Text]
    F --> B


🧩 Components

Frontend: Streamlit (UI + input handling)

Model: PyTorch 2-layer LSTM (in model.py)

Vocab: Precomputed vocab_objects.pth

Weights: creative_text_generator.pth hosted externally (e.g., Google Drive)

⚙️ Local Setup
1. Clone Repository
git clone https://github.com/yourusername/creative-text-generator.git
cd creative-text-generator

2. Create Environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m nltk.downloader punkt

3. Prepare Data (Run Once)

Download:

lyrics-data.csv

abcnews-date-text.csv

Then run:

python preprocess.py

4. Configure Model URL

Edit app.py → find MODEL_URL = ""
Replace with your direct download link for creative_text_generator.pth.

5. Run the App
streamlit run app.py


Your model will auto-download and cache on first run.

🧠 Model Details
Parameter	Value
Layers	2
Hidden Size	512
Embedding	256
Dropout	0.5
Sampling	Top-K (k=5), Temperature=0.8
Corpus	Lyrics + News (1.5M sequences)
🔮 Future Enhancements

Build a Flask/FastAPI backend for /generate endpoint

Deploy model as an API (Docker/Cloud Run)

Connect multiple UIs (Streamlit, web apps, etc.)

👨‍💻 Developed By

Mynk

Project built with ❤️ using PyTorch & Streamlit

🪪 License

This project is licensed under the MIT License
