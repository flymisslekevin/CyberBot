#!/usr/bin/env python3
"""
Demonstration of how cosine similarity scores are calculated
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def demonstrate_cosine_similarity():
    """Show how cosine similarity works with simple examples."""

    print("🎯 Understanding Cosine Similarity Scores")
    print("=" * 50)

    # Simple example with 3 documents
    documents = [
        "ransomware attack cyber security",
        "machine learning artificial intelligence",
        "board governance risk management"
    ]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    print(f"📊 TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"📝 Documents:")
    for i, doc in enumerate(documents):
        print(f"   Doc {i}: {doc}")

    # Test queries
    test_queries = [
        "ransomware attack",
        "machine learning",
        "governance"
    ]

    print(f"\n🔍 Testing Queries:")

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Transform query to vector
        query_vector = vectorizer.transform([query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        print(f"Cosine Similarity Scores:")
        for i, score in enumerate(similarities):
            print(f"   Doc {i}: {score:.3f}")

        # Find best match
        best_match = np.argmax(similarities)
        print(f"   Best match: Doc {best_match} (score: {similarities[best_match]:.3f})")

    print(f"\n📈 Score Interpretation:")
    print(f"   1.0 = Perfect match (identical content)")
    print(f"   0.8+ = Very similar")
    print(f"   0.5-0.8 = Moderately similar")
    print(f"   0.2-0.5 = Somewhat similar")
    print(f"   0.0-0.2 = Weak similarity")
    print(f"   0.0 = No similarity")

def explain_why_low_scores():
    """Explain why your scores are relatively low."""

    print(f"\n🤔 Why Your Scores Are Low (0.031-0.079):")
    print(f"=" * 50)

    reasons = [
        "📄 Large Document: Your PDF is a full academic paper (~20+ pages)",
        "📊 Many Features: 5000 TF-IDF features dilute the similarity",
        "🎯 Specific Queries: Short queries vs. long documents",
        "📝 Academic Language: Formal academic text vs. simple queries",
        "🔍 N-gram Features: Multi-word features reduce exact matches"
    ]

    for reason in reasons:
        print(f"   {reason}")

    print(f"\n💡 To Get Higher Scores:")
    print(f"   • Use longer, more specific queries")
    print(f"   • Search for exact phrases from the document")
    print(f"   • Compare with shorter documents")
    print(f"   • Use more focused search terms")

def show_calculation_steps():
    """Show the mathematical steps of cosine similarity."""

    print(f"\n🧮 Cosine Similarity Calculation:")
    print(f"=" * 40)

    print(f"1. Convert documents to TF-IDF vectors")
    print(f"2. Convert query to TF-IDF vector")
    print(f"3. Calculate: cos(θ) = (A·B) / (||A|| × ||B||)")
    print(f"   where A = query vector, B = document vector")
    print(f"4. Result: 0 ≤ similarity ≤ 1")

    print(f"\n📐 Mathematical Formula:")
    print(f"   similarity = dot_product(query, doc) / (magnitude(query) × magnitude(doc))")

if __name__ == "__main__":
    demonstrate_cosine_similarity()
    explain_why_low_scores()
    show_calculation_steps()
