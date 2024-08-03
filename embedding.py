from openai import OpenAI
import numpy as np
from typing import List, Dict, Any
from config import config, get_kant_urls
from data_preprocessing import process_urls
import json
import os
from tqdm import tqdm

client = OpenAI(api_key=config.get('openai_api_key'))

CACHE_FILE = 'embedding_cache.json'
ALL_EMBEDDINGS_FILE = 'all_embeddings.npy'

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {CACHE_FILE}. The file might be corrupted. Creating a new cache.")
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def reset_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    if os.path.exists(ALL_EMBEDDINGS_FILE):
        os.remove(ALL_EMBEDDINGS_FILE)
    print("Cache has been reset.")

embedding_cache = load_cache()

def get_embeddings(texts: List[str], model: str = config['embedding_model']) -> np.ndarray:
    uncached_texts = []
    uncached_indices = []
    embeddings = [None] * len(texts)

    for i, text in enumerate(texts):
        cache_key = f"{text[:100]}_{model}"  # Use first 100 chars as key
        if cache_key in embedding_cache:
            embeddings[i] = embedding_cache[cache_key]
        else:
            uncached_texts.append(text.replace("\n", " "))
            uncached_indices.append(i)

    if uncached_texts:
        response = client.embeddings.create(input=uncached_texts, model=model)
        for i, embedding_data in zip(uncached_indices, response.data):
            embedding = embedding_data.embedding
            embeddings[i] = embedding
            cache_key = f"{texts[i][:100]}_{model}"
            embedding_cache[cache_key] = embedding

    save_cache(embedding_cache)
    return np.array(embeddings)

def create_embeddings(processed_data: List[Dict[str, Any]]) -> np.ndarray:
    if os.path.exists(ALL_EMBEDDINGS_FILE):
        print("Loading existing embeddings...")
        return np.load(ALL_EMBEDDINGS_FILE)

    batch_size = 100  # Adjust based on your needs and API limits
    all_embeddings = []

    for i in tqdm(range(0, len(processed_data), batch_size), desc="Creating embeddings"):
        batch = processed_data[i:i+batch_size]
        texts = [item['text'] for item in batch]
        batch_embeddings = get_embeddings(texts)
        all_embeddings.extend(batch_embeddings)

    all_embeddings_array = np.array(all_embeddings)
    np.save(ALL_EMBEDDINGS_FILE, all_embeddings_array)
    return all_embeddings_array

if __name__ == "__main__":
    kant_urls = get_kant_urls()
    processed_data = process_urls(kant_urls)
    embeddings = create_embeddings(processed_data)
    print(f"Created {len(embeddings)} embeddings.")