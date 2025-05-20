import streamlit as st
import pandas as pd
import numpy as np
# NLTK is used by utils, but good to have here if you ever call its functions directly
# import nltk 

# Import functions from our utils.py file
from utils import (
    configure_gemini_api, # Changed name
    get_embedding,
    load_embeddings_from_csv,
    find_top_n_similar_pages,
    suggest_anchor_texts_for_page_keyword_based,
    URL_COLUMN, TITLE_COLUMN # Import constants if needed for clarity, though utils handles them
)

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Internal Linking Assistant")

# --- Initialize Gemini API ---
# This should be one of the first things to run.
# The result is cached, so configure_gemini_api() body runs once.
if 'gemini_api_configured_successfully' not in st.session_state:
    st.session_state.gemini_api_configured_successfully = configure_gemini_api()

# If API configuration failed, stop the app gracefully.
if not st.session_state.get('gemini_api_configured_successfully', False):
    st.error("Gemini API could not be configured. Please check Streamlit Cloud secrets for 'GEMINI_API_KEY' or logs for details.")
    st.stop()

# --- Constants ---
CSV_FILE_PATH = "embeddings.csv" # This file must exist in your GitHub repo

# --- Load Data ---
# This is cached by @st.cache_data in utils.py
pages_df = load_embeddings_from_csv(CSV_FILE_PATH)

if pages_df.empty:
    # Error messages are now more detailed within load_embeddings_from_csv
    st.warning("No data loaded from CSV or CSV data is invalid. Please check the file and its format.")
    st.stop() # Stop execution if no data to work with

# --- UI Elements ---
st.title("✍️ Internal Linking Assistant")
st.markdown("""
Paste the content of your new, unpublished article below.
The tool will suggest existing pages from our site to link to, along with relevant anchor text phrases
(based on keywords from target page titles) from your article.
""")

# Use session_state to preserve text area content across reruns
if 'article_content_input' not in st.session_state:
    st.session_state.article_content_input = ""

input_article_text = st.text_area(
    "Paste your new article content here:",
    value=st.session_state.article_content_input,
    height=400,
    key="article_content_key", # Use a key for the widget
    on_change=lambda: setattr(st.session_state, 'article_content_input', st.session_state.article_content_key)
)
# The on_change callback updates the session state variable when the text_area changes.
# Alternatively, you can just read from st.session_state.article_content_key directly.

col1, col2 = st.columns(2)
with col1:
    top_n_links = st.number_input("Number of link suggestions (Top N):", min_value=1, max_value=20, value=5, key="top_n_links")
with col2:
    max_anchors_per_link = st.number_input("Max anchor texts per suggested link:", min_value=1, max_value=5, value=3, key="max_anchors")

analyze_button = st.button("🔗 Analyze for Linking Opportunities")

# --- Processing and Display ---
if analyze_button:
    current_article_content = st.session_state.article_content_input # Get current content

    if not current_article_content.strip():
        st.warning("Please paste some article content before analyzing.")
    else:
        with st.spinner("Analyzing article... This may take a moment."):
            # 1. Embed the entire input article
            st.markdown("---") # Visual separator
            progress_text = st.empty() # For status updates

            progress_text.info("Embedding input article...")
            input_article_embedding = get_embedding(current_article_content, task_type="RETRIEVAL_DOCUMENT")

            if input_article_embedding is None or np.all(input_article_embedding == 0):
                progress_text.error("Could not generate embedding for the input article. Please check content or API status.")
                st.stop()

            # 2. Find Top N Similar Pages
            progress_text.info("Finding similar published pages...")
            suggested_pages = find_top_n_similar_pages(input_article_embedding, pages_df, top_n_links)

            if not suggested_pages:
                progress_text.warning("No relevant existing pages found based on the overall article content.")
            else:
                # 3. Generate Anchor Text Suggestions
                progress_text.info("Generating keyword-based anchor text suggestions...")
                results_with_anchors = []
                
                for i, page_info in enumerate(suggested_pages):
                    page_title_for_anchors = page_info.get(TITLE_COLUMN) # Use constant
                    if not page_title_for_anchors or pd.isna(page_title_for_anchors):
                        # This case should be handled by load_embeddings_from_csv creating a fallback title
                        # st.caption(f"Note: No specific title found for {page_info[URL_COLUMN]}, using derived title for anchors.")
                        page_title_for_anchors = "Untitled Page" # Fallback, though utils should provide one

                    anchors = suggest_anchor_texts_for_page_keyword_based(
                        input_article_text=current_article_content,
                        target_page_title=page_title_for_anchors,
                        num_anchors_to_suggest=max_anchors_per_link
                    )
                    results_with_anchors.append({**page_info, "anchors": anchors})
                
                progress_text.success("Analysis complete!")

                # 4. Display Results
                st.subheader("💡 Suggested Internal Links & Anchor Texts")

                for result in results_with_anchors:
                    title_display = result.get(TITLE_COLUMN, 'N/A')
                    url_display = result.get(URL_COLUMN, '#')
                    score_display = result.get('score', 0.0)

                    st.markdown(f"#### [{title_display}]({url_display})")
                    st.markdown(f"<small>Link to this page (Overall relevance: {score_display:.3f})</small>", unsafe_allow_html=True)

                    if result['anchors']:
                        st.markdown("**Suggested Anchor Texts from your article:**")
                        for anchor_suggestion in result['anchors']:
                            context = anchor_suggestion['context_sentence']
                            anchor_text_keyword = anchor_suggestion['text']
                            
                            highlighted_context = context
                            try:
                                # Case-insensitive find, then use original casing of matched text in context for bolding
                                start_index = context.lower().find(anchor_text_keyword.lower())
                                if start_index != -1:
                                    matched_text_in_context = context[start_index : start_index + len(anchor_text_keyword)]
                                    highlighted_context = (
                                        context[:start_index] +
                                        f"**{matched_text_in_context}**" +
                                        context[start_index + len(anchor_text_keyword):]
                                    )
                            except Exception: # Broad exception for safety
                                pass # Keep context as is if highlighting fails

                            st.markdown(f"- `{anchor_text_keyword}`") # Anchor text is the keyword
                            st.markdown(f"    ↳ Kontext: \"<em>...{highlighted_context}...</em>\"", unsafe_allow_html=True)
                        st.markdown("---") # Separator after each suggested link's anchors
                    else:
                        st.markdown(f"<small><i>No specific keyword-based anchor text suggestions found from the title '{title_display}' within your article.</i></small>", unsafe_allow_html=True)
                        st.markdown("---") # Separator even if no anchors