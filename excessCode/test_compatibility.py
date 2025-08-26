#!/usr/bin/env python3
"""
Test script to demonstrate compatibility between enhanced 2nd Draft.py and Vector Search.py
Now with sentence and paragraph-level search capabilities!
"""

# Import the functions from 2nd Draft.py
from importlib import import_module
import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
draft_module = import_module('2nd Draft')
vector_search_module = import_module('Vector Search')

def test_compatibility():
    """Test that the enhanced vectorization works with the search engine."""

    print("🔍 Testing Enhanced Search with Sentence & Paragraph-Level Results")
    print("=" * 70)

    # Step 1: Extract and vectorize using enhanced 2nd Draft.py
    pdf_path = "Ransomware paper `4.pdf"

    print("📄 Step 1: Extracting text from PDF...")
    text = draft_module.parse_pdf_to_text(pdf_path)

    print("📝 Step 2: Splitting into semantic paragraphs...")
    paragraphs = draft_module.split_into_paragraphs(text)
    print(f"   Extracted {len(paragraphs)} semantic paragraphs")

    print("🔢 Step 3: Vectorizing with enhanced concept capture...")
    tfidf_matrix, feature_names = draft_module.vectorize_paragraphs(paragraphs)
    print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"   Number of features: {len(feature_names)}")

    # Step 4: Create a new vectorizer for the search engine
    print("🔍 Step 4: Creating search engine compatible vectorizer...")

    # Recreate the vectorizer with the same parameters
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.stem import WordNetLemmatizer
    import re

    lemmatizer = WordNetLemmatizer()

    def custom_tokenizer(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 2]
        return lemmatized

    # Create vectorizer with same parameters as in 2nd Draft.py
    min_df = min(2, len(paragraphs))
    max_df = 0.95 if len(paragraphs) > 1 else 1.0

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 3),
        tokenizer=custom_tokenizer,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        analyzer='word'
    )

    # Fit the vectorizer on the same data
    vectorizer.fit(paragraphs)

    # Step 5: Initialize search engine
    print("🔍 Step 5: Initializing search engine...")
    search_engine = vector_search_module.TFIDFSearchEngine(
        tfidf_matrix,
        vectorizer,
        paragraphs
    )

    # Step 6: Test different search types
    print("\n🎯 Step 6: Testing different search granularities...")
    test_queries = [
        "ransomware attack",
        "cyber security",
        "board level governance",
        "recovery capability"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔎 Query: '{query}'")
        print(f"{'='*60}")

        # Test document-level search (original)
        print(f"\n📄 DOCUMENT-LEVEL RESULTS:")
        doc_results = search_engine.search(query, top_k=3)
        if doc_results:
            for rank, (idx, score, doc) in enumerate(doc_results, 1):
                print(f"   {rank}. Score: {score:.3f} - Doc #{idx}")
                print(f"      Preview: {doc[:150]}...")

        # Test paragraph-level search (new)
        print(f"\n📝 PARAGRAPH-LEVEL RESULTS:")
        para_results = search_engine.search_paragraphs(query, top_k=3)
        if para_results:
            for rank, (doc_idx, para_idx, score, paragraph) in enumerate(para_results, 1):
                print(f"   {rank}. Score: {score:.3f} - Doc #{doc_idx}, Para #{para_idx}")
                print(f"      Preview: {paragraph[:150]}...")

        # Test sentence-level search (new)
        print(f"\n💬 SENTENCE-LEVEL RESULTS:")
        sent_results = search_engine.search_sentences(query, top_k=3)
        if sent_results:
            for rank, (doc_idx, sent_idx, score, sentence) in enumerate(sent_results, 1):
                print(f"   {rank}. Score: {score:.3f} - Doc #{doc_idx}, Sent #{sent_idx}")
                print(f"      Preview: {sentence[:150]}...")

    print(f"\n✅ Enhanced Search Test Complete!")
    print(f"Now you can get specific sentences and paragraphs instead of entire documents!")

if __name__ == "__main__":
    test_compatibility()
