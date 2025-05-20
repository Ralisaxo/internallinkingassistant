import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import spacy
import google.generativeai as genai
import ast # For safely evaluating string representations of lists

# --- Column Name Constants (from user's existing CSV) ---
URL_COLUMN = "url"
TITLE_COLUMN = "title" # New, but highly recommended
EMBEDDING_COLUMN_NAME_FROM_CSV = "Vector Embedding text-embedding-004"
INTERNAL_EMBEDDING_COLUMN = "embedding_vector_processed" # Internal name after processing

# --- SpaCy Model Loading ---
@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    print(f"Attempting to load SpaCy model: {model_name}...")
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"SpaCy model '{model_name}' not found. Attempting to download...")
        try:
            spacy.cli.download(model_name)
            return spacy.load(model_name)
        except Exception as e:
            st.error(f"Failed to download/load SpaCy model '{model_name}'. Error: {e}")
            st.error("Please ensure you have internet connectivity and necessary permissions if running locally. On Streamlit Cloud, this might indicate a compatibility issue or network problem.")
            return None # Return None if model can't be loaded

nlp = load_spacy_model() # Load an instance of the model

# --- NLTK Downloader ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' not found. Downloading now...")
    nltk.download('punkt', quiet=True)

# --- Gemini API Configuration ---
@st.cache_resource # Cache the configuration result
def configure_gemini_api():
    print("Configuring Gemini API...")
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it in your Streamlit Cloud app settings.")
            return False
        genai.configure(api_key=api_key)
        # Test with a simple call if necessary, but configure() itself might suffice
        # or do a lightweight model list. For now, assume configure() is enough.
        # models = [m for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
        # if not models:
        #     st.error("No embedding models available with the provided API key.")
        #     return False
        print("Gemini API configured successfully.")
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        return False

# --- Embedding Function ---
@st.cache_data(show_spinner="Embedding text...") # Cache based on inputs
def get_embedding(text: str, model_name: str = "models/text-embedding-004", task_type="RETRIEVAL_DOCUMENT") -> np.array | None:
    if not text or not text.strip():
        # st.warning("Empty text provided for embedding.")
        return np.zeros(768) # Return zero vector for empty input to avoid breaking similarity math

    # Ensure API is configured before trying to embed
    if 'gemini_api_configured_successfully' not in st.session_state or not st.session_state.gemini_api_configured_successfully:
         # Attempt to configure if not already done (e.g. on first run of get_embedding)
        if not configure_gemini_api():
             st.error("Cannot generate embedding: Gemini API not configured.")
             return None # Indicate failure
        st.session_state.gemini_api_configured_successfully = True


    try:
        embedding = genai.embed_content(
            model=model_name,
            content=text,
            task_type=task_type
        )
        return np.array(embedding['embedding'])
    except Exception as e:
        st.error(f"Error generating embedding for text snippet ('{text[:30]}...'): {e}")
        return None # Indicate failure

# --- Load Pre-computed Embeddings from CSV ---
@st.cache_data(show_spinner="Loading existing page embeddings...")
def load_embeddings_from_csv(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: The CSV file '{csv_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading CSV file '{csv_path}': {e}")
        return pd.DataFrame()

    # Validate essential columns
    if URL_COLUMN not in df.columns:
        st.error(f"Column '{URL_COLUMN}' not found in the CSV. Please ensure it exists.")
        return pd.DataFrame()
    if EMBEDDING_COLUMN_NAME_FROM_CSV not in df.columns:
        st.error(f"Column '{EMBEDDING_COLUMN_NAME_FROM_CSV}' not found in the CSV. Please ensure it exists.")
        return pd.DataFrame()

    # Process embeddings
    def parse_embedding(embedding_str):
        try:
            return np.array(ast.literal_eval(embedding_str))
        except (ValueError, SyntaxError, TypeError): # Added TypeError for non-string inputs
            return None # Return None for malformed strings or non-string data

    df[INTERNAL_EMBEDDING_COLUMN] = df[EMBEDDING_COLUMN_NAME_FROM_CSV].astype(str).apply(parse_embedding)
    
    # Filter out rows where embedding parsing failed
    original_rows = len(df)
    df.dropna(subset=[INTERNAL_EMBEDDING_COLUMN], inplace=True)
    if len(df) < original_rows:
        st.warning(f"{original_rows - len(df)} rows were removed due to invalid embedding formats.")

    if df.empty:
        st.error("No valid embeddings found after processing the CSV.")
        return pd.DataFrame()

    # Check embedding dimension consistency (e.g., for text-embedding-004, it's 768)
    expected_dim = 768
    actual_dims = df[INTERNAL_EMBEDDING_COLUMN].apply(len).unique()
    if len(actual_dims) > 1 or (len(actual_dims) == 1 and actual_dims[0] != expected_dim):
        st.warning(f"Embeddings have inconsistent dimensions or do not match expected {expected_dim}. Found dimensions: {actual_dims}. This might cause issues.")
    
    # Handle the 'title' column - make it optional but preferred
    if TITLE_COLUMN not in df.columns:
        st.warning(f"Column '{TITLE_COLUMN}' not found in the CSV. Anchor text suggestions will rely on URL slugs if possible, which is less effective. Adding a '{TITLE_COLUMN}' is highly recommended.")
        # Create a fallback title from URL if 'title' column is missing
        df[TITLE_COLUMN] = df[URL_COLUMN].apply(
            lambda x: x.split('/')[-1].replace('-', ' ').replace('_', ' ').title() if pd.notna(x) and isinstance(x, str) else "Untitled Page"
        )
    else:
         # Ensure titles are strings, fill NaNs
        df[TITLE_COLUMN] = df[TITLE_COLUMN].astype(str).fillna("Untitled Page")


    return df[[URL_COLUMN, TITLE_COLUMN, INTERNAL_EMBEDDING_COLUMN]] # Select and reorder

# --- Find Top N Similar Pages ---
def find_top_n_similar_pages(input_embedding: np.array | None, all_pages_df: pd.DataFrame, top_n: int) -> list:
    if input_embedding is None or np.all(input_embedding == 0) or all_pages_df.empty or INTERNAL_EMBEDDING_COLUMN not in all_pages_df.columns:
        return []
    
    # Ensure input_embedding is 1D array of the correct dimension
    if input_embedding.ndim != 1 or input_embedding.shape[0] != 768: # Assuming 768 for text-embedding-004
        st.error(f"Input embedding has incorrect shape or dimension. Expected 1D array of 768, got shape {input_embedding.shape}.")
        return []

    all_embeddings = np.stack(all_pages_df[INTERNAL_EMBEDDING_COLUMN].values) # More robust stacking
    
    if all_embeddings.shape[1] != input_embedding.shape[0]:
         st.error(f"Dimension mismatch between input article embedding ({input_embedding.shape[0]}) and pre-computed page embeddings ({all_embeddings.shape[1]}).")
         return []

    similarities = cosine_similarity(input_embedding.reshape(1, -1), all_embeddings)[0]
    
    # Get indices of top N similarities, filtering out very low scores
    # Argsort sorts in ascending order, so we take the last 'top_n' and reverse them.
    # Also, only consider if similarity > threshold.
    min_similarity_threshold = 0.1 # Example threshold
    
    # Create a list of (index, similarity) tuples for scores above threshold
    qualified_indices_scores = [(i, sim) for i, sim in enumerate(similarities) if sim > min_similarity_threshold]
    
    # Sort these qualified results by score in descending order
    qualified_indices_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take the top_n from this sorted list
    top_n_results = qualified_indices_scores[:top_n]

    results = []
    for i, score in top_n_results:
        results.append({
            "url": all_pages_df.iloc[i][URL_COLUMN],
            "title": all_pages_df.iloc[i][TITLE_COLUMN],
            "score": score,
            # "page_embedding": all_pages_df.iloc[i][INTERNAL_EMBEDDING_COLUMN] # Not strictly needed by keyword approach
        })
    return results

# --- Anchor Text Suggestion (Keyword-Based) ---
MIN_ANCHOR_LENGTH_WORDS = 2
MAX_ANCHOR_LENGTH_WORDS = 7

def extract_keywords_from_title(title_text: str) -> list[str]:
    if not nlp: # Check if SpaCy model loaded
        st.warning("SpaCy model not available. Cannot extract keywords from title.")
        return []
    if not title_text or not isinstance(title_text, str):
        return []

    doc = nlp(title_text)
    keywords = [chunk.text.strip() for chunk in doc.noun_chunks]
    
    if not keywords: # Fallback to significant words if no noun chunks
        keywords = [
            token.text.strip()
            for token in doc
            if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN", "ADJ"]
        ]
    
    processed_keywords = []
    seen_keywords = set()
    for kw in keywords:
        word_count = len(kw.split())
        # Ensure keyword is not empty after stripping and meets length criteria
        if kw and MIN_ANCHOR_LENGTH_WORDS <= word_count <= MAX_ANCHOR_LENGTH_WORDS and kw.lower() not in seen_keywords:
            processed_keywords.append(kw)
            seen_keywords.add(kw.lower())
    return processed_keywords

def suggest_anchor_texts_for_page_keyword_based(
    input_article_text: str,
    target_page_title: str,
    num_anchors_to_suggest: int
) -> list:
    if not input_article_text or not target_page_title:
        return []

    title_keywords = extract_keywords_from_title(target_page_title)
    if not title_keywords:
        return []

    try:
        input_article_sentences = nltk.sent_tokenize(input_article_text)
    except Exception as e:
        st.warning(f"Could not tokenize input article into sentences: {e}")
        return [] # Or treat whole article as one sentence, less ideal
    
    candidate_anchors = []
    for sentence in input_article_sentences:
        for keyword in title_keywords: # keyword is already filtered by length etc.
            if keyword.lower() in sentence.lower(): # Case-insensitive search
                candidate_anchors.append({
                    "text": keyword, # Store the original casing of the keyword
                    "context_sentence": sentence,
                    "keyword_length": len(keyword.split()) # For potential sorting
                })
    
    if not candidate_anchors:
        return []
    
    # Sort by keyword length (descending) to prioritize more specific anchors
    candidate_anchors.sort(key=lambda x: x["keyword_length"], reverse=True)
    
    final_anchors = []
    seen_texts_lowercase = set() # To ensure unique anchor texts
    for anchor in candidate_anchors:
        if len(final_anchors) >= num_anchors_to_suggest:
            break
        anchor_text_lower = anchor['text'].lower()
        if anchor_text_lower not in seen_texts_lowercase:
            final_anchors.append({
                "text": anchor['text'], # Keep original casing from title extraction
                "context_sentence": anchor['context_sentence']
            })
            seen_texts_lowercase.add(anchor_text_lower)
            
    return final_anchors