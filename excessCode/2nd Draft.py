import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def parse_pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

def preprocess_text(text):
    """Clean and preprocess text for better concept capture."""
    # Convert to lowercase
    text = text.lower()

    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def split_into_paragraphs(text):
    """Improved paragraph splitting that captures semantic boundaries."""
    # Preprocess the text
    text = preprocess_text(text)

    # Split on multiple newlines or significant punctuation
    # This better captures semantic paragraph boundaries
    paragraphs = re.split(r'\n\s*\n|\n\s*[A-Z][^a-z]', text)

    # Clean up paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 50]

    return paragraphs

def vectorize_paragraphs(paragraphs, max_features=5000):
    """Convert paragraphs into TF-IDF matrix with enhanced concept capture."""

    # Initialize lemmatizer for better word normalization
    lemmatizer = WordNetLemmatizer()

    # Custom tokenizer that includes lemmatization
    def custom_tokenizer(text):
        # Simple word tokenization using regex
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Lemmatize tokens to capture word variations
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 2]
        return lemmatized

    # Adjust parameters for small document sets
    min_df = min(2, len(paragraphs))  # At least 2 docs or all docs if fewer
    max_df = 0.95 if len(paragraphs) > 1 else 1.0  # Allow all terms if only 1 doc

    # Enhanced TF-IDF vectorizer with concept-capturing features
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        ngram_range=(1, 3),  # Capture 1-3 word sequences (concepts)
        tokenizer=custom_tokenizer,
        min_df=min_df,  # Minimum document frequency
        max_df=max_df,  # Maximum document frequency
        sublinear_tf=True,  # Apply sublinear scaling
        analyzer='word'
    )

    tfidf_matrix = vectorizer.fit_transform(paragraphs)
    return tfidf_matrix, vectorizer.get_feature_names_out()

# Example usage:
if __name__ == "__main__":
    pdf_path = "Ransomware paper `4.pdf"  # Replace with your PDF path

    # Step 1: Extract text from PDF
    text = parse_pdf_to_text(pdf_path)

    # Step 2: Split text into semantic paragraphs
    paragraphs = split_into_paragraphs(text)
    print(f"Extracted {len(paragraphs)} semantic paragraphs")

    # Step 3: Vectorize paragraphs with enhanced concept capture
    tfidf_matrix, feature_names = vectorize_paragraphs(paragraphs)

    # Step 4: Print results
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print(f"RESULT: {tfidf_matrix}")
    print("Top 10 features:", feature_names[:10])

    # Optional: print first paragraph
    print("\n--- First paragraph ---")
    print(paragraphs[0])

    tfidf_array = tfidf_matrix.toarray()
    print("TF-IDF NumPy matrix shape:", tfidf_array.shape)
    print(tfidf_array)
