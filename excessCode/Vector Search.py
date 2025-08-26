import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class TFIDFSearchEngine:
    """
    A simple search engine that uses a precomputed TF-IDF matrix
    to find the most relevant documents (or paragraphs).

    Attributes:
        tfidf_matrix: scipy.sparse matrix of shape (n_docs, n_features)
        vectorizer: fitted TfidfVectorizer
        documents: list of original text documents (e.g., paragraphs)
    """
    def __init__(self, tfidf_matrix, vectorizer, documents):
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
        self.documents = documents

    def search(self, query, top_k=5):
        """
        Search for the query string and return the top_k matching documents.

        Args:
            query (str): User's search query.
            top_k (int): Number of top results to return.

        Returns:
            List of tuples: [(doc_index, score, document_text), ...]
        """
        # 1. Transform the query into the TF-IDF space
        query_vec = self.vectorizer.transform([query])  # shape (1, n_features)

        # 2. Compute cosine similarity between the query and all docs
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()  # (n_docs,)

        # 3. Get indices of top_k highest scores
        top_indices = np.argsort(sims)[::-1][:top_k]

        # 4. Build result list
        results = [(int(idx), float(sims[idx]), self.documents[idx])
                   for idx in top_indices]
        return results

    def search_sentences(self, query, top_k=5, min_sentence_length=20):
        """
        Search for the query and return the most relevant sentences from the documents.

        Args:
            query (str): User's search query.
            top_k (int): Number of top sentences to return.
            min_sentence_length (int): Minimum sentence length to consider.

        Returns:
            List of tuples: [(doc_index, sentence_index, score, sentence_text), ...]
        """
        # 1. Transform the query into the TF-IDF space
        query_vec = self.vectorizer.transform([query])

        # 2. Split documents into sentences and create sentence-level TF-IDF
        all_sentences = []
        sentence_to_doc = []  # Maps sentence index to document index

        for doc_idx, document in enumerate(self.documents):
            # Split into sentences using regex
            sentences = re.split(r'[.!?]+', document)
            sentences = [s.strip() for s in sentences if len(s.strip()) >= min_sentence_length]

            for sent_idx, sentence in enumerate(sentences):
                all_sentences.append(sentence)
                sentence_to_doc.append(doc_idx)

        if not all_sentences:
            return []

        # 3. Create TF-IDF matrix for sentences
        min_df = min(1, len(all_sentences))  # At least 1 sentence
        max_df = 0.95 if len(all_sentences) > 1 else 1.0  # Allow all terms if only 1 sentence

        sentence_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 3),
            min_df=min_df,
            max_df=max_df
        )

        sentence_tfidf = sentence_vectorizer.fit_transform(all_sentences)

        # 4. Transform query using sentence vectorizer
        query_sentence_vec = sentence_vectorizer.transform([query])

        # 5. Calculate similarities
        sentence_sims = cosine_similarity(query_sentence_vec, sentence_tfidf).flatten()

        # 6. Get top sentences
        top_sentence_indices = np.argsort(sentence_sims)[::-1][:top_k]

        # 7. Build results
        results = []
        for sent_idx in top_sentence_indices:
            doc_idx = sentence_to_doc[sent_idx]
            score = sentence_sims[sent_idx]
            sentence = all_sentences[sent_idx]
            results.append((doc_idx, sent_idx, float(score), sentence))

        return results

    def search_paragraphs(self, query, top_k=5, min_paragraph_length=50):
        """
        Search for the query and return the most relevant paragraphs.

        Args:
            query (str): User's search query.
            top_k (int): Number of top paragraphs to return.
            min_paragraph_length (int): Minimum paragraph length to consider.

        Returns:
            List of tuples: [(doc_index, paragraph_index, score, paragraph_text), ...]
        """
        # 1. Transform the query into the TF-IDF space
        query_vec = self.vectorizer.transform([query])

        # 2. Split documents into paragraphs
        all_paragraphs = []
        paragraph_to_doc = []

        for doc_idx, document in enumerate(self.documents):
            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', document)
            paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_paragraph_length]

            for para_idx, paragraph in enumerate(paragraphs):
                all_paragraphs.append(paragraph)
                paragraph_to_doc.append(doc_idx)

        if not all_paragraphs:
            return []

        # 3. Create TF-IDF matrix for paragraphs
        min_df = min(1, len(all_paragraphs))  # At least 1 paragraph
        max_df = 0.95 if len(all_paragraphs) > 1 else 1.0  # Allow all terms if only 1 paragraph

        paragraph_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 3),
            min_df=min_df,
            max_df=max_df
        )

        paragraph_tfidf = paragraph_vectorizer.fit_transform(all_paragraphs)

        # 4. Transform query using paragraph vectorizer
        query_paragraph_vec = paragraph_vectorizer.transform([query])

        # 5. Calculate similarities
        paragraph_sims = cosine_similarity(query_paragraph_vec, paragraph_tfidf).flatten()

        # 6. Get top paragraphs
        top_paragraph_indices = np.argsort(paragraph_sims)[::-1][:top_k]

        # 7. Build results
        results = []
        for para_idx in top_paragraph_indices:
            doc_idx = paragraph_to_doc[para_idx]
            score = paragraph_sims[para_idx]
            paragraph = all_paragraphs[para_idx]
            results.append((doc_idx, para_idx, float(score), paragraph))

        return results


if __name__ == "__main__":
    # Example usage (assuming you already have tfidf_matrix, vectorizer, and documents list):
    # from your_vectorization_module import tfidf_matrix, vectorizer, paragraphs

    # Initialize search engine
    search_engine = TFIDFSearchEngine(tfidf_matrix, vectorizer, paragraphs)

    # Interactive loop
    print("Simple TF-IDF Search Engine")
    print("Type your query and press Enter (or 'exit' to quit)")
    while True:
        user_query = input("\nQuery > ")
        if user_query.lower() in {'exit', 'quit'}:
            print("Goodbye!")
            break

        hits = search_engine.search(user_query, top_k=5)
        print(f"\nTop {len(hits)} matches:")
        for rank, (idx, score, doc) in enumerate(hits, start=1):
            print(f"{rank}. [Document #{idx}, score={score:.3f}]\n{doc}\n")
