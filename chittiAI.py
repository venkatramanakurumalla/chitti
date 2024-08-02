import streamlit as st
import threading
import math
import numpy as np
from gtts import gTTS
import pygame
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import logging
import requests
from bs4 import BeautifulSoup

# Ensure google-generativeai is installed and imported correctly
try:
    import google.generativeai as genai
    genai.configure(api_key="YOUR_API_KEY_HERE")
    model = genai.GenerativeModel('gemini-1.5-flash')
except ImportError as e:
    st.error("google-generativeai module not found. Please install it using 'pip install google-generativeai'")
    raise e
except Exception as e:
    st.error(f"An error occurred with google-generativeai: {e}")
    raise e

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize sentence transformer model for encoding
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logging.error(f"Failed to initialize SentenceTransformer: {e}")
    sentence_model = None

class MemoryStore:
    def __init__(self, max_memory=100, preloaded_memories=None):
        self.max_memory = max_memory
        self.memories = []
        self.embeddings = []

        if preloaded_memories:
            for memory in preloaded_memories:
                self.add_memory(memory)

    def add_memory(self, text):
        if len(self.memories) >= self.max_memory:
            self.memories.pop(0)
            self.embeddings.pop(0)

        self.memories.append(text)
        if sentence_model:
            try:
                embedding = sentence_model.encode([text])[0]
                self.embeddings.append(embedding)
            except Exception as e:
                logging.error(f"Failed to encode memory: {e}")
        else:
            logging.warning("SentenceTransformer not initialized. Skipping embedding.")

    def get_relevant_memories(self, query, top_k=3):
        if not self.memories or not self.embeddings:
            return []

        if not sentence_model:
            logging.warning("SentenceTransformer not initialized. Returning random memories.")
            return np.random.choice(self.memories, min(top_k, len(self.memories)), replace=False).tolist()

        try:
            query_embedding = sentence_model.encode([query])[0]
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.memories[i] for i in top_indices]
        except Exception as e:
            logging.error(f"Error in get_relevant_memories: {e}")
            return []

# Initialize memory store with pre-trained data
pretrained_data = [
    "User: What is the weather like today?\nChitti: The weather is sunny and warm with a slight breeze.",
    "User: Who won the football match last night?\nChitti: The home team won with a score of 2-1.",
    "User: Can you play some music?\nChitti: Sure, I can play your favorite songs.",
    # Add more pre-trained conversation pairs here...
]

memory_store = MemoryStore(preloaded_memories=pretrained_data)

def process_input(text):
    try:
        st.write(f"You: {text}")
        
        # Retrieve relevant memories
        relevant_memories = memory_store.get_relevant_memories(text)
        context = "\n".join(relevant_memories) if relevant_memories else "No previous context available."

        # Include relevant memories in the prompt
        prompt = f"Context from previous conversations:\n{context}\n\nUser: {text}\nChitti:"
        response = model.generate_content(prompt)
        ai_response = response.text

        st.write(f"Chitti: {ai_response}")
        
        # Add the interaction to memory
        memory_store.add_memory(f"User: {text}\nChitti: {ai_response}")
        
        # Speak the response
        threading.Thread(target=speak_text, args=(ai_response,)).start()
    except Exception as e:
        error_message = f"Error processing input: {e}"
        st.write(f"{error_message}")
        logging.error(error_message)

def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        error_message = f"Error in speak_text method: {e}"
        st.write(f"{error_message}")
        logging.error(error_message)

def browse(url):
    logging.debug(f"Opening URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    st.write(f"Scraped content from {url}:\n{text[:1000]}...")  # Show only first 1000 chars
    speak_text(f"Here is some information from the webpage: {text[:200]}")  # Speak first 200 chars

# Streamlit UI
st.title("Chitti")
st.write("Welcome to Chitti, your AI assistant!")

user_input = st.text_input("Ask Chitti", "")
if st.button("Submit"):
    if user_input.startswith("browse "):
        url = user_input.split(" ", 1)[1]
        browse(url)
    else:
        process_input(user_input)
