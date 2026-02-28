import streamlit as st
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up the Streamlit page layout
st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="centered")

# Cache the NLTK downloads so they only run once when the app starts
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

# Define the stop words globally
stop_words = set(stopwords.words('english'))

# 1. Preprocessing Function (Exactly like the notebook)
def preprocess_text(text):
    text = str(text)
    words = word_tokenize(text.lower())
    clean_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(clean_words)

# 2. Summarization Function (Exactly like the notebook)
def generate_summary(text, top_n=3):
    text = str(text)
    sentences = sent_tokenize(text)
    
    # If the text is already shorter than the requested summary, just return it
    if len(sentences) <= top_n:
        return text
    
    clean_sentences = [preprocess_text(s) for s in sentences]
    
    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(clean_sentences)
    
    # Calculate Cosine Similarity Matrix
    sim_matrix = cosine_similarity(sentence_vectors)
    
    # Rank sentences based on similarity scores
    scores = np.zeros(len(sentences))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                scores[i] += sim_matrix[i][j]
                
    # Sort and pick top sentences, maintaining their original order in the text
    top_sentence_indices = np.argsort(scores)[::-1][:top_n]
    top_sentence_indices.sort() # Sort indices to keep original paragraph flow
    
    ranked_sentences = [sentences[i] for i in top_sentence_indices]
    return " ".join(ranked_sentences)

# 3. Streamlit User Interface
st.title("📝 NLP Extractive Text Summarizer")
st.write("This application uses **TF-IDF and Cosine Similarity** to extract the most important sentences from your text.")

# User input text area
user_input = st.text_area("Input Text", height=250, placeholder="Paste your news article or long text here...")

# Slider to let the user choose how many sentences they want in the summary
num_sentences = st.slider("Select Summary Length (Number of Sentences)", min_value=1, max_value=10, value=3)

# Summarize button logic
if st.button("Generate Summary"):
    if len(user_input.split()) < 30:
        st.warning("Please enter a longer piece of text (at least 30 words) for a meaningful summary.")
    else:
        with st.spinner("Analyzing text and calculating vector similarities..."):
            try:
                # Call our function
                summary_text = generate_summary(user_input, top_n=num_sentences)
                
                st.success("Summary Generated Successfully!")
                st.write("### Your Summary:")
                st.info(summary_text)
            except Exception as e:
                st.error(f"An error occurred: {e}")