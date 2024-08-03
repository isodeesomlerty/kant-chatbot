import re
import requests
from nltk.tokenize import sent_tokenize
import nltk
from config import config, get_kant_urls
import pickle
import os

nltk.download('punkt', quiet=True)

PROCESSED_DATA_FILE = 'processed_data.pkl'

def fetch_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def clean_text(text):
    # Remove Project Gutenberg header and footer
    text = re.sub(r'^\s*The Project Gutenberg.*?\n\n', '', text, flags=re.DOTALL)
    text = re.sub(r'\n\n\s*End of the Project Gutenberg.*$', '', text, flags=re.DOTALL)
    
    # Remove line breaks and extra whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_title(text):
    match = re.search(r'Title: (.+)', text)
    return match.group(1) if match else "Unknown Title"

def split_into_chunks(text, max_chunk_size=config['max_chunk_size']):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_urls(urls):
    if os.path.exists(PROCESSED_DATA_FILE):
        print("Loading existing processed data...")
        with open(PROCESSED_DATA_FILE, 'rb') as f:
            return pickle.load(f)

    print("Processing texts...")
    processed_data = []
    
    for url in urls:
        try:
            raw_text = fetch_text_from_url(url)
            title = extract_title(raw_text)
            clean_text_content = clean_text(raw_text)
            chunks = split_into_chunks(clean_text_content)
            
            for chunk in chunks:
                processed_data.append({
                    'text': chunk,
                    'source': title,
                    'url': url
                })
            
            print(f"Successfully processed: {title}")
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    with open(PROCESSED_DATA_FILE, 'wb') as f:
        pickle.dump(processed_data, f)
    
    return processed_data

if __name__ == "__main__":
    kant_urls = get_kant_urls()
    processed_data = process_urls(kant_urls)
    print(f"Processed {len(processed_data)} chunks from Kant's works.")