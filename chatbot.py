from openai import OpenAI
import os
from typing import List, Dict, Any
from retrieval import create_retriever
from data_preprocessing import process_urls
from embedding import create_embeddings, get_embeddings
from config import config, get_kant_urls
import numpy as np

client = OpenAI(api_key=config.get('openai_api_key'))

if not client.api_key:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

class KantChatbot:
    def __init__(self):
        print("Initializing Kant Chatbot...")
        self.kant_urls = get_kant_urls()
        self.processed_data = None
        self.embeddings = None
        self.retriever = None

    def ensure_data_loaded(self):
        if self.processed_data is None:
            print("Loading processed data...")
            self.processed_data = process_urls(self.kant_urls)
            print(f"Loaded {len(self.processed_data)} text chunks.")

        if self.embeddings is None:
            print("Loading embeddings...")
            self.embeddings = np.load('all_embeddings.npy')
            print(f"Loaded {len(self.embeddings)} embeddings.")

        if self.retriever is None:
            print("Initializing retriever...")
            self.retriever = create_retriever(self.processed_data, self.embeddings)

    def generate_response(self, query: str) -> str:
        self.ensure_data_loaded()
        
        query_embedding = get_embeddings([query])[0]
        context = self.retriever.retrieve(query_embedding)
        prompt = self._create_prompt(query, context)
        
        try:
            response = client.chat.completions.create(
                model=config['chat_model'],
                messages=[
                    {"role": "system", "content": "You are a chatbot that answers questions about Immanuel Kant's philosophy based on his writings. Always cite the specific work you're referencing in your answer."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

    def _create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join([f"{item['text']} [Source: {item['source']}]" for item in context])
        return f"Based on the following excerpts from Kant's writings:\n\n{context_text}\n\nPlease answer the following question, citing the specific work(s) you're referencing: {query}"

def main():
    chatbot = KantChatbot()
    print("Kant Chatbot is ready. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        response = chatbot.generate_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()