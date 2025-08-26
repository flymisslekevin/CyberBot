###
Requirements:

pdfplumber  
tqdm  
numpy  
faiss-cpu  
ollama  
transformers  

##
CLI commands:  

open terminal  
cd to project directory

$pip install pdfplumber tqdm numpy faiss-cpu ollama transformers

##
Ollama Install:  
$curl -fsSL https://ollama.ai/install.sh | sh  
$ollama pull nomic-embed-text   
$ollama pull phi  

###
Project Folder Directory:  
```
├── 📄 Scripts
│   ├── processText copy.py          # PDF text extraction
│   ├── clean copy.py                # Text cleaning and preprocessing
│   ├── build_local_index copy.py    # Vector embedding and FAISS index creation
│   └── rag_local copy.py            # RAG query interface
│
├── 📚 pdf_storage/                  # Document storage and processing
│   ├── industryReports/             # Industry reports and whitepapers
│   │   ├── *.pdf                    # Original PDF documents
│   │   └── *.txt                    # Extracted text files
│   ├── regulations/                 # Regulatory documents
│   │   ├── *.pdf                   
│   │   └── *.txt                   
│   ├── securityFrameworks/          # Security frameworks and standards
│   │   ├── *.pdf                    
│   │   └── *.txt                    
│   ├── processed/                   # Intermediate text extraction
│   │   ├── industryReports/
│   │   ├── regulations/
│   │   └── securityFrameworks/
│   └── clean/                       # Cleaned and preprocessed text
│       ├── industryReports/
│       ├── regulations/
│       └── securityFrameworks/
│
├── 🔍 vectors/                      # Vector embeddings and search index
│   ├── faiss.index       
│   └── mapping.json                
│
├── 🐍 venv/                         # Python virtual environment
│   ├── bin/                   
│   ├── lib/                         # Installed packages
│   └── pyvenv.cfg                 
