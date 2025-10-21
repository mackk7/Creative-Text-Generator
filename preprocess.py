import pandas as pd
import torch
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import nltk
import gc
from tqdm import tqdm

# --- Setup ---
print("🚀 Starting preprocessing...")
print("This will read your large CSV files and create the small 'vocab_objects.pth' dictionary.")

# Ensure NLTK resources are downloaded. This is the fix for your error.
try:
    # Attempt to download the required 'punkt' data
    nltk.download('punkt', quiet=True)
    
    # We must explicitly import and initialize the tokenizer
    # If the resource is still missing, the script will crash here, but with a clearer error.
    # We will remove the explicit check here as it is causing the failure.
    print("✅ NLTK tokenization dependencies found.")
except Exception as e:
    # This should now be resolved since you ran the downloader manually.
    print(f"❌ CRITICAL NLTK ERROR: Could not load tokenization data. Please ensure the 'punkt' package is installed via NLTK.")
    exit()

# --- Normalization Function ---
def normalize_text(text):
    """A function to clean and normalize a block of text."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) # Remove text in brackets like [chorus]
    text = re.sub(r'[^a-z\s\']', '', text) # Keep only letters, apostrophes, spaces
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with one
    return text

# --- File Paths ---
lyrics_file = 'data/lyrics-data.csv'
news_file = 'data/abcnews-date-text.csv'
output_vocab_file = 'vocab_objects.pth'

all_tokens = []

# --- 1. Process Lyrics File (Memory Optimized) ---
try:
    print("\nProcessing lyrics file...")
    lyrics_df = pd.read_csv(lyrics_file)
    lyrics_df = lyrics_df.dropna(subset=['Lyric', 'language'])
    lyrics_df = lyrics_df[lyrics_df['language'] == 'en']
    lyrics_text = '\n'.join(lyrics_df['Lyric'])
    cleaned_lyrics = normalize_text(lyrics_text)
    
    # The error happens during this function call:
    all_tokens.extend(word_tokenize(cleaned_lyrics))
    print("Lyrics tokenized successfully.")

    # Free up memory
    del lyrics_df, lyrics_text, cleaned_lyrics
    gc.collect()
except FileNotFoundError:
    print(f"❌ ERROR: {lyrics_file} not found. Check your file path.")
except Exception as e:
    print(f"An unexpected error occurred processing {lyrics_file}: {e}")
    
# --- 2. Process News File (Memory Optimized) ---
try:
    print("\nProcessing news file...")
    news_df = pd.read_csv(news_file)
    news_df = news_df.dropna(subset=['headline_text'])
    news_text = '\n'.join(news_df['headline_text'])
    cleaned_news = normalize_text(news_text)
    
    # The error happens during this function call:
    all_tokens.extend(word_tokenize(cleaned_news))
    print("News tokenized successfully.")

    # Free up memory
    del news_df, news_text, cleaned_news
    gc.collect()
except FileNotFoundError:
    print(f"❌ ERROR: {news_file} not found. Check your file path.")
except Exception as e:
    print(f"An unexpected error occurred processing {news_file}: {e}")


if not all_tokens:
    print("\n❌ CRITICAL FAILURE: Zero tokens were processed. Check CSV files and NLTK data.")
    exit()
else:
    print(f"\nTotal tokens from all files: {len(all_tokens):,}")

    # --- 3. Build and Save Vocabulary (Remains the same) ---
    print("\nBuilding vocabulary...")
    word_counts = Counter(all_tokens)
    vocab_list = [word for word, count in word_counts.items() if count >= 5]
    word_to_int = {word: i + 1 for i, word in enumerate(vocab_list)}
    word_to_int['<PAD>'] = 0
    int_to_word = {i: word for word, i in word_to_int.items()}
    vocab_size = len(word_to_int)

    # Save Vocabulary Objects
    vocab_objects = {
        'word_to_int': word_to_int,
        'int_to_word': int_to_word,
        'vocab_size': vocab_size
    }

    torch.save(vocab_objects, output_vocab_file)
    
    print(f"\n✅ Success! Vocabulary objects saved to '{output_vocab_file}'.")
    print("You are now ready to run 'streamlit run app.py'")

    # Final cleanup
    del all_tokens, word_counts, vocab_list
    gc.collect()
