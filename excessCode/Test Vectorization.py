import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

def vectorize_text(text, max_features=5000):
    """Convert text into TF-IDF vector."""
    # You can customize stop_words and other params as needed
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform([text])  # single document in this case
    return tfidf_matrix, vectorizer.get_feature_names_out()

# Example usage:
if __name__ == "__main__":
    pdf_path = "HBR Draft 2.pdf"  # Replace with your PDF path
    text = parse_pdf_to_text(pdf_path)

    tfidf_matrix, feature_names = vectorize_text(text)

    print("TF-IDF shape:", tfidf_matrix.shape)
    print("Top 10 features:", feature_names[:10])
