import streamlit as st
import torch
import torch.nn as nn
# --- ADD NLTK IMPORT HERE ---
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re
import os
import requests # Make sure to add 'requests' to your requirements.txt
from tqdm import tqdm
from model import TextGenerator # Imports your model class from model.py

# --- NEW: Download NLTK Data on Startup ---
# This is CRUCIAL for Streamlit Cloud deployment
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer already downloaded.")
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    # Make sure 'nltk' is in your requirements.txt
    nltk.download('punkt', quiet=True)
    print("NLTK 'punkt' downloaded.")
# --- END NLTK DOWNLOAD ---
# --- Configuration & Constants ---
st.set_page_config(
    page_title="Creative Text Generator",
    page_icon="✍️",
    layout="wide", # Use wide layout for better spacing
    initial_sidebar_state="expanded"
)

# --- NEW Custom CSS for Neumorphic Dark Theme ---
st.markdown("""
<style>
    /* Base Background */
    .stApp {
        background-color: #2E3440; /* Nord dark background */
        background-attachment: fixed;
    }
    
    /* Main Content Area Styling */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Title Styling */
    h1 {
        color: #ECEFF4; /* Nord light text */
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Markdown Text Styling */
    .stMarkdown p, .stMarkdown li {
        color: #D8DEE9; /* Nord lighter text */
        font-size: 1.1rem; /* Slightly larger text */
    }
    
    /* Card-like effect for Input/Output Sections */
    div[data-testid="stVerticalBlock"] { /* Target vertical blocks */
       /* background-color: #3B4252; /* Nord slightly lighter dark */
       /* border-radius: 10px; */
       /* padding: 1.5rem; */
       /* margin-bottom: 1.5rem; */
       /* Neumorphic shadow */
       /* box-shadow: 5px 5px 10px #242933, -5px -5px 10px #383f4d; */
    }

    /* Input Label Styling */
    .stTextInput label, .stSlider label {
        color: #E5E9F0 !important; /* Nord light label */
        font-weight: bold;
    }
    
    /* Input Box Styling (Neumorphic inset) */
    .stTextInput input, .stTextArea textarea {
        background-color: #2E3440; /* Match background */
        color: #ECEFF4;
        border: 1px solid #4C566A; /* Nord grey border */
        border-radius: 8px;
        box-shadow: inset 3px 3px 6px #242933, inset -3px -3px 6px #383f4d;
    }

    /* Button Styling (Neumorphic raised) */
    .stButton>button {
        background-color: #5E81AC; /* Nord blue */
        color: #ECEFF4;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: bold;
        box-shadow: 3px 3px 6px #242933, -3px -3px 6px #383f4d;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #81A1C1; /* Lighter Nord blue */
        box-shadow: inset 2px 2px 4px #242933, inset -2px -2px 4px #383f4d;
    }
    .stButton>button:active {
         box-shadow: inset 3px 3px 6px #242933, inset -3px -3px 6px #383f4d;
    }

    /* Output Expander Styling */
    .stExpander {
        background-color: #3B4252; /* Nord slightly lighter dark */
        border-radius: 10px;
        border: none;
        box-shadow: 5px 5px 10px #242933, -5px -5px 10px #383f4d;
    }
    .stExpander header { /* Expander header */
        font-weight: bold;
        color: #88C0D0; /* Nord teal accent */
    }
    div[data-testid="stExpander"] div[role="button"] + div div[data-testid="stMarkdownContainer"] {
        background-color: transparent; /* Remove inner background */
        border: none;
        padding: 0.5rem 1rem 1rem 1rem; /* Adjust padding */
    }
    div[data-testid="stExpander"] div[role="button"] + div div[data-testid="stMarkdownContainer"] p {
        color: #D8DEE9; /* Light text inside */
        font-family: 'Courier New', Courier, monospace; /* Monospace font for output */
    }

    /* Sidebar Styling */
    .stSidebar {
        background-color: #3B4252; /* Nord slightly lighter dark */
        box-shadow: 5px 0px 10px #242933; /* Shadow on the right */
    }
    .stSidebar .stMarkdown p, .stSidebar .stMetric, .stSidebar h2, .stSidebar .stCaption {
        color: #ECEFF4; /* Light text */
    }
    .stSidebar .stMetric .stMetricLabel {
       color: #A3BE8C; /* Nord green accent for labels */
    }
    .stSidebar .stMetric .stMetricValue {
        color: #ECEFF4; /* Light value */
    }
    .stSidebar .stMarkdown a { /* Link color */
        color: #88C0D0; /* Nord teal accent */
    }

</style>
""", unsafe_allow_html=True)


# --- File Paths and Model URL ---
MODEL_PATH = 'creative_text_generator.pth'
VOCAB_PATH = 'vocab_objects.pth'
MODEL_URL = "https://www.dropbox.com/scl/fi/may99yg8hro9571prwlv8/creative_text_generator.pth?rlkey=4xft0gnpgq68r8bbietfpjtal&st=8bciulq6&dl=1" # Your working link

# --- Model Hyperparameters ---
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 512 # The "big brain"
NUM_LAYERS = 2
DROPOUT_PROB = 0.5
FINAL_TRAINING_LOSS = 3.2290 # Store the final loss
LOSS_THRESHOLD = 4.0 # Only show metrics if loss is below this

# --- Helper Functions ---

def download_model(url, file_path):
    """Downloads the large .pth model file if it doesn't exist."""
    if os.path.exists(file_path):
        return

    st.info(f"Model file not found. Downloading... (This happens once on first launch)")
    
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 8192 # Increased block size for potentially faster download
        progress_bar = st.progress(0, text="Downloading Model...")
        
        with open(file_path, 'wb') as f:
            downloaded_bytes = 0
            for data in r.iter_content(block_size):
                f.write(data)
                downloaded_bytes += len(data)
                if total_size_in_bytes > 0:
                    progress = min(downloaded_bytes / total_size_in_bytes, 1.0)
                    # Use markdown for better progress text formatting
                    progress_text = f"Downloading Model... {int(progress * 100)}% ({downloaded_bytes // (1024*1024)} MB / {total_size_in_bytes // (1024*1024)} MB)"
                    progress_bar.progress(progress, text=progress_text)
                else: # If total size is unknown, just show bytes downloaded
                     progress_bar.progress(0, text=f"Downloading Model... {downloaded_bytes // (1024*1024)} MB")

        progress_bar.empty()
        st.success("Model downloaded successfully!")

    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.error("Please make sure your MODEL_URL is a valid direct download link.")
        if os.path.exists(file_path):
            os.remove(file_path)
        st.stop()


@st.cache_resource
def load_resources():
    """Loads all the necessary resources (vocab, model) into memory."""
    device = torch.device("cpu")

    try:
        vocab_objects = torch.load(VOCAB_PATH)
        word_to_int = vocab_objects['word_to_int']
        int_to_word = vocab_objects['int_to_word']
        vocab_size = vocab_objects['vocab_size']
    except FileNotFoundError:
        st.error(f"Error: `{VOCAB_PATH}` not found. Did you run `preprocess.py` first?")
        st.stop()
    except Exception as e:
        st.error(f"Error loading `{VOCAB_PATH}`: {e}")
        st.stop()

    model = TextGenerator(
        vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB
    ).to(device)

    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=False)
        )
    except FileNotFoundError:
        st.error(f"Error: `{MODEL_PATH}` not found. Download may have failed.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.error("Check if hyperparameters match the trained model or if the file is corrupt.")
        st.stop()

    model.eval()
    return model, word_to_int, int_to_word, device, vocab_size

def generate_text(model, word_to_int, int_to_word, device, start_text, length, top_k=3): 
    """Generates text using the trained model with Top-K sampling."""
    model.eval()

    words = word_tokenize(start_text.lower())
    generated_words = []
    hidden = model.init_hidden(1, device)

    # Warm up hidden state
    for w in words:
        try:
            word_tensor = torch.tensor([[word_to_int.get(w, word_to_int['<PAD>'])]]).to(device)
            _, hidden = model(word_tensor, hidden)
        except Exception:
            continue

    # Initialize input tensor
    try:
        last_word = words[-1]
        ipt = torch.tensor([[word_to_int.get(last_word, word_to_int['<PAD>'])]]).to(device)
    except Exception:
        rand_index = torch.randint(1, len(word_to_int), (1,1)).item()
        ipt = torch.tensor([[rand_index]]).to(device)

    # Generation loop
    for _ in range(length):
        output, hidden = model(ipt, hidden)
        probs = torch.softmax(output.squeeze().div(0.8), dim=0) # Temperature for creativity
        
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        choice_idx = torch.multinomial(top_k_probs, 1).item()
        word_idx = top_k_indices[choice_idx].item()
        
        if word_idx == word_to_int['<PAD>']: break
        
        word = int_to_word.get(word_idx, "")
        generated_words.append(word)
        ipt = torch.tensor([[word_idx]]).to(device)
        
    return ' '.join(generated_words)

# --- Main Application UI ---

st.title("✍️ Creative Text Generator")
st.markdown(
    """
    Trained on **1.5 million** song lyrics & news headlines, this AI generates creative text. 
    Provide a starting thread and see the tapestry it creates!
    """
)

# --- CONDITIONAL PERFORMANCE DISPLAY (Sidebar) ---
# Logic remains the same: Only show if loss is good
if FINAL_TRAINING_LOSS <= LOSS_THRESHOLD:
    st.sidebar.header("📊 Model Performance")
    st.sidebar.markdown(
        """
        Final metrics from training. *Lower Loss = Better*.
        """
    )
    st.sidebar.metric(
        label="Final Training Loss", 
        value=f"{FINAL_TRAINING_LOSS:.4f}", 
        delta=f"-{round(5.0 - FINAL_TRAINING_LOSS, 4)} vs Initial", # More descriptive delta
        delta_color="inverse" 
    )
    st.sidebar.caption("Based on the last training epoch.")
else:
    st.sidebar.info("Performance metrics hidden (Loss > 4.0).")
# --- END PERFORMANCE DISPLAY ---


# 1. Ensure model is downloaded
download_model(MODEL_URL, MODEL_PATH)

# 2. Load resources (cached)
model, word_to_int, int_to_word, device, vocab_size = load_resources()

# 3. User Inputs (Main Area with Columns for Layout)
col1, col2 = st.columns([3, 1]) # Make input wider than slider

with col1:
    start_phrase = st.text_input(
        "Enter your starting phrase:",
        value="The city sleeps but" # Default value
    )

with col2:
    num_words = st.slider(
        "Words to generate:", 
        min_value=1, 
        max_value=30, 
        value=15, 
        label_visibility="collapsed" # Hide label if obvious
    )


# 4. Generate Button Logic
if st.button("✨ Generate Text"): # Changed button text slightly
    if not start_phrase:
        st.warning("Please enter a starting phrase.")
    elif MODEL_URL == "PASTE_YOUR_DIRECT_DOWNLOAD_LINK_HERE": # Keep this check
        st.error("CRITICAL ERROR: MODEL_URL not set in app.py!")
    else:
        with st.spinner("⏳ Generating..."):
            generated_text = generate_text(
                model, word_to_int, int_to_word, device,
                start_text=start_phrase,
                length=num_words
            )
            
            # Display Result in an expander for better layout
            with st.expander("AI Generated Text:", expanded=True):
                st.markdown(f"> **{start_phrase}** {generated_text}")

# --- Footer ---
st.markdown("---")
st.caption("Built with PyTorch & Streamlit.")

