import json
import os
from typing import List, Dict, Any

def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """Load the configuration from the JSON file and environment variables."""
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    # Override with environment variables if they exist
    config['openai_api_key'] = os.getenv('OPENAI_API_KEY', config.get('openai_api_key'))
    config['chat_model'] = os.getenv('CHAT_MODEL', config.get('chat_model'))
    config['embedding_model'] = os.getenv('EMBEDDING_MODEL', config.get('embedding_model'))
    
    return config

config = load_config()

def get_kant_urls() -> List[str]:
    """Return a list of Kant's work URLs."""
    return [item['url'] for item in config['kant_urls']]

def get_kant_works() -> List[Dict[str, str]]:
    """Return a list of dictionaries containing Kant's work URLs and descriptions."""
    return config['kant_urls']

if __name__ == "__main__":
    print("Kant's works:")
    for work in config['kant_urls']:
        print(f"- {work['description']}: {work['url']}")
    print(f"\nEmbedding model: {config['embedding_model']}")
    print(f"Chat model: {config['chat_model']}")
    print(f"Max chunk size: {config['max_chunk_size']}")
    print(f"Number of relevant chunks: {config['num_relevant_chunks']}")