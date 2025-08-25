# LocalRAG - Enhanced Document Q&A System

A powerful, **completely local** RAG (Retrieval-Augmented Generation) system with **advanced technical document processing** that answers questions without requiring any external APIs or internet connection. Perfect for handling sensitive technical documents while maintaining privacy and da## 📋 Repository Structure

```
LocalRAG/
├── 📄 README.md                    # Complete documentation (this file)
├── 📄 LICENSE                      # MIT License
├── 🐍 doc-qa-local.py              # ✨ Main application script
├── 🔧 enhanced_processor.py        # ✨ Technical document processing module
├── 📋 local-requirements.txt       # Python dependencies
├── 🔧 setup.sh                     # Automated setup script
├── 📁 local_chroma_db/             # Vector database storage (created after processing)
├── 📁 venv/                        # Python virtual environment (created during setup)
├── 📄 test_document.txt            # Sample test document
└── 📄 .gitignore                   # Git ignore patterns
```

### Core Files

**`doc-qa-local.py`** - Main application with CLI interface
- Document processing with enhanced chunking
- Multiple answer generation modes
- Interactive Q&A sessions
- Extensive command-line options

**`enhanced_processor.py`** - Advanced document processing
- Technical document pattern recognition
- Smart chunking with section awareness
- Enhanced vector search with relevance scoring
- Comprehensive answer generation

**`local-requirements.txt`** - Python dependencies
- sentence-transformers (embeddings)
- chromadb (vector database)
- transformers (Q&A models)
- torch (PyTorch backend)
- PyMuPDF (PDF processing)
- pytesseract (OCR)
- Additional supporting libraries

## 📈 Comparison with Cloud Solutions

| Feature | LocalRAG | Cloud RAG |
|---------|----------|-----------|
| Privacy | ✅ 100% Local | ❌ Data sent to cloud |
| Cost | ✅ Free after setup | 💰 Per-query costs |
| Internet | ✅ Works offline | ❌ Requires internet |
| Setup | ⚡ Quick local setup | 🔧 API keys needed |
| Speed | ⚡ Fast local inference | 🐌 Network latency |
| Customization | ✅ Full control | ❌ Limited options |
| Data Security | ✅ Your machine only | ❌ Third-party servers |

## 🚀 Advanced Usage Examples

### Large Technical Document Processing
```bash
# Process a large technical manual
python doc-qa-local.py process 
  --document large_manual.pdf 
  --chunk-size 1500 
  --chunk-overlap 300 
  --cpu

# Query with comprehensive analysis
python doc-qa-local.py query --cpu --comprehensive
```

### Batch Processing Multiple Documents
```bash
# Process multiple documents to the same database
python doc-qa-local.py process --document doc1.pdf --db-path shared_db --cpu
python doc-qa-local.py process --document doc2.pdf --db-path shared_db --cpu
python doc-qa-local.py query --db-path shared_db --cpu
```

### Custom Embedding Models
```bash
# Use different embedding models for better semantic understanding
python doc-qa-local.py process 
  --document document.pdf 
  --embedding-model "sentence-transformers/all-mpnet-base-v2" 
  --cpu
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Ensure all dependencies are in `local-requirements.txt`
5. Update documentation if needed
6. Submit a pull request with detailed description

## 📄 License

MIT License - feel free to use in your projects!

## 🎉 What's New in This Version

- ✅ **Fully Tested Setup**: Complete installation and usage validation
- ✅ **Enhanced Technical Processing**: Specialized handling for technical documents
- ✅ **Multiple Answer Modes**: Standard, comprehensive, and simple modes
- ✅ **Improved Documentation**: Comprehensive setup and troubleshooting guides
- ✅ **macOS Compatibility**: Verified working on Apple Silicon with proper CPU flags
- ✅ **Robust Error Handling**: Better handling of GPU/CUDA issues and memory constraints

---

**Ready to get started?** The system is fully tested and ready to use! Simply follow the installation steps above and start processing your documents.

**Need help?** Check the troubleshooting section above or open an issue for support.
## ✅ **System Successfully Tested & Ready to Use!**

The LocalRAG system has been fully tested and validated with:
- ✅ **Complete Environment Setup**: Virtual environment with all dependencies installed
- ✅ **Document Processing Verified**: Successfully processed test documents with enhanced chunking
- ✅ **Q&A System Operational**: Interactive query system running with transformer models loaded
- ✅ **Technical Document Support**: Enhanced processing for automotive/technical manuals
- ✅ **Multiple Answer Modes**: Tested standard, comprehensive, and transformer-based responses

## 🚀 Key Features

### Core Capabilities
- **🔒 100% Local Processing**: No external APIs, no data leaves your machine
- **📄 Advanced OCR Support**: Handles rotated and scanned PDFs with automatic text extraction
- **🧠 Multiple AI Models**: Choose from transformer models, Ollama, comprehensive analysis, or simple extraction
- **📊 Enhanced Smart Chunking**: Technical document-aware splitting with section preservation
- **⚡ Fast Vector Search**: ChromaDB with enhanced relevance scoring for technical content
- **🎯 Interactive Q&A**: Terminal-based question-answering interface with multiple response modes
- **📚 Source Attribution**: Shows which document sections support each answer
- **🔧 Confidence Scoring**: Get confidence levels and model information for answers

### Enhanced Features
- **🔬 Technical Document Processing**: Specialized handling for technical manuals, specifications, and engineering documents
- **📋 Comprehensive Answer Generation**: Detailed, structured responses with signal lists and technical analysis
- **🎚 Multiple Answer Modes**: Standard, comprehensive, and simple modes for different use cases
- **🔍 Context-Aware Search**: Enhanced vector search that understands technical document structure
- **⚙ Command-Line Flexibility**: Extensive options for customizing processing and answer generation

## 🛠 Installation & Setup - Verified Working!

### ✅ **Quick Setup (Tested on macOS)**

1. **Clone and navigate to the repository**
   ```bash
   git clone https://github.com/shreyasren/LocalRAG.git
   cd LocalRAG
   ```

2. **Run the automated setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

### ✅ **Manual Setup (Recommended)**

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r local-requirements.txt
   ```

3. **Install system dependencies (macOS):**
   ```bash
   brew install tesseract poppler
   ```

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr poppler-utils
   ```

   **Windows:**
   - Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
   - Install Poppler: http://blog.alivate.com.au/poppler-windows/

### ⚠️ **Important Notes from Testing**
- **Use `--cpu` flag**: Required on macOS to avoid CUDA errors
- **PyMuPDF Installation**: Pre-compiled wheels work best; avoid compilation from source
- **Memory Requirements**: 4-8GB RAM recommended for processing large documents

## 🎯 Usage - Tested & Working!

### Step 1: Process Your Document ✅

First, vectorize your document (one-time setup per document):

**Basic Processing (Standard Chunking):**
```bash
python doc-qa-local.py process --document your_document.pdf --cpu --no-enhanced
```

**Enhanced Processing for Technical Documents (Recommended):**
```bash
python doc-qa-local.py process --document technical_manual.pdf --cpu
```

**Processing Options:**
- `--cpu`: **Required on macOS** to avoid GPU/CUDA issues
- `--chunk-size 1000`: Text chunk size in characters (default: 1000)
- `--chunk-overlap 200`: Overlap between chunks (default: 200)
- `--db-path ./local_chroma_db`: Database location (default: ./local_chroma_db)
- `--embedding-model all-MiniLM-L6-v2`: Embedding model (default: all-MiniLM-L6-v2)
- `--no-ocr`: Disable OCR for faster processing (skip if document is rotated)
- `--no-enhanced`: Disable enhanced technical document processing (use basic chunking)

**📋 Key Differences:**
- **Enhanced Processing (Default)**: Uses smart chunking optimized for technical documents, better signal/component recognition
- **Basic Processing (--no-enhanced)**: Uses standard text chunking, faster but less optimized for technical content
- **Enhanced processing is recommended** for technical manuals, specifications, and complex documents

**✅ Example (Successfully Tested):**
```bash
python doc-qa-local.py process --document test_document.txt --cpu

# Output shows:
# ✅ Document processed successfully!
# You can now run queries with: python doc-qa-local.py query --cpu
```

### Step 2: Query Your Document ✅

After processing, start the interactive Q&A session:

**Standard Mode (Transformer Models - Tested Working):**
```bash
python doc-qa-local.py query --cpu
```

**Comprehensive Mode (Detailed Technical Responses):**
```bash
python doc-qa-local.py query --cpu --comprehensive
```

**Basic Mode (Simple Extraction):**
```bash
python doc-qa-local.py query --llm simple --cpu
```

**✅ Example Query Session:**
```
============================================================
Local Document Q&A System
============================================================
Type your questions below. Type 'exit' or 'quit' to end.

📝 Your Question: What is machine learning?

📚 ANSWER:
Machine learning (ML) is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

[High confidence answer from RoBERTa]
```

## 🤖 Answer Generation Modes

### 1. Standard Mode (Default)
```bash
python doc-qa-local.py query --cpu
```
- **Method**: Transformer models (DistilBERT, RoBERTa) with auto-enhancement
- **Quality**: High-quality extractive answers with automatic supplementation for short responses
- **Speed**: Fast inference on CPU/GPU
- **Best For**: General questions and quick technical lookups

### 2. Comprehensive Mode (NEW!)
```bash
python doc-qa-local.py query --cpu --comprehensive
```
- **Method**: Advanced multi-chunk analysis with structured response generation
- **Quality**: Detailed, structured responses with signal lists, technical components, and section-based analysis
- **Features**: 
  - Extracts specific signal names and technical terms
  - Groups information by document sections
  - Creates bullet-point lists for complex queries
  - Provides comprehensive technical analysis
- **Best For**: Technical documents, signal analysis, complex engineering questions

### 3. Ollama Integration
```bash
# First install Ollama
brew install ollama
ollama pull llama3.2:3b  # or llama3.2:7b for better quality

# Then use
python doc-qa-local.py query --llm ollama --cpu
```
- **Quality**: Excellent, human-like responses
- **Models**: Llama 3.2, Mistral, Phi-3, and more
- **Requires**: Ollama installation and model download

### 4. Simple Extraction
```bash
python doc-qa-local.py query --llm simple --cpu
```
- **Speed**: Fastest option
- **Quality**: Basic keyword matching
- **Use Case**: Quick searches or resource-constrained systems

## 🔧 Technical Architecture & Enhancement Features

### Enhanced Technical Document Processing

The LocalRAG system includes specialized enhancements for technical documents like manuals, specifications, and engineering documentation:

#### **Smart Chunking & Pattern Recognition**
- **Section-Aware Splitting**: Preserves technical sections and subsections
- **Header Recognition**: Identifies and maintains document structure  
- **Technical Pattern Detection**: Recognizes signal names, commands, and technical terminology
- **Context Preservation**: Keeps section titles with content for better understanding

#### **Enhanced Vector Search**
- **Context Expansion**: Groups related technical information from the same section
- **Technical Relevance Scoring**: Prioritizes chunks with technical indicators over headers/metadata
- **Multi-Chunk Analysis**: Combines information from multiple relevant document sections
- **Header Penalty**: Reduces scoring for header-heavy content to focus on technical details

#### **Multiple Answer Generation Modes**

**1. Standard Mode (Default - Tested Working)**
- **Method**: Transformer models (DistilBERT, RoBERTa) with auto-enhancement
- **Quality**: High-quality extractive answers with automatic supplementation for short responses
- **Speed**: Fast inference on CPU/GPU
- **Best For**: General questions and quick technical lookups

**2. Comprehensive Mode (Technical Documents)**
- **Method**: Advanced multi-chunk analysis with structured response generation
- **Features**: 
  - Extracts specific signal names and technical terms
  - Groups information by document sections
  - Creates bullet-point lists for complex queries
  - Provides comprehensive technical analysis
- **Best For**: Technical documents, signal analysis, complex engineering questions

**3. Ollama Integration (Optional)**
- **Quality**: Excellent, human-like responses
- **Models**: Llama 3.2, Mistral, Phi-3, and more
- **Requires**: Ollama installation and model download

**4. Simple Extraction (Fastest)**
- **Speed**: Fastest option
- **Quality**: Basic keyword matching
- **Use Case**: Quick searches or resource-constrained systems

### System Architecture

```
Document Input (PDF/TXT/DOCX)
    ↓
OCR Processing (if needed)
    ↓
Enhanced/Basic Chunking
    ↓
Embedding Generation (Sentence Transformers)
    ↓
Vector Database Storage (ChromaDB)
    ↓
Query Processing Pipeline:
User Question → Question Embedding → Enhanced Vector Search → Context Creation → Answer Generation → Response with Sources
```

## 📊 Example Workflows

### Basic Document Q&A
```bash
# 1. Process a document with OCR support
python doc-qa-local.py process --document bcm.pdf --cpu

# 2. Start Q&A with standard transformer models
python doc-qa-local.py query --cpu

# Example interaction:
📝 Your Question: What are the main components of BCM?

📚 ANSWER:
Body Control Module (BCM) includes lighting control, power management, and vehicle communication systems.

[High confidence answer from RoBERTa]
```

### Technical Document Analysis
```bash
# 1. Process with enhanced technical features
python doc-qa-local.py process --document technical_spec.pdf --cpu

# 2. Use comprehensive mode for detailed analysis
python doc-qa-local.py query --cpu --comprehensive

# Example interaction:
📝 Your Question: list all control signals for the system

📚 ANSWER:
**System Control Signals:**
• [Detailed signal list with technical context]
• [Component relationships and dependencies]  
• [Functional descriptions and usage patterns]
```

## ⚙️ Advanced Configuration

### Custom Embedding Models
For better semantic search, try different embedding models:
```bash
python doc-qa-local.py process \
  --document document.pdf \
  --embedding-model "sentence-transformers/all-mpnet-base-v2"
```

### GPU Acceleration
If you have a CUDA-compatible GPU:
```bash
python doc-qa-local.py query  # Remove --cpu flag for GPU acceleration
```

### Disable Enhanced Processing
For simpler documents or faster processing:
```bash
python doc-qa-local.py process --document simple_doc.pdf --no-enhanced --cpu
python doc-qa-local.py query --no-enhanced --cpu
```

### Batch Processing Multiple Documents
```bash
# Process multiple documents to the same database
python doc-qa-local.py process --document doc1.pdf --db-path shared_db
python doc-qa-local.py process --document doc2.pdf --db-path shared_db
python doc-qa-local.py query --db-path shared_db --llm transformers
```

## 🔍 Answer Quality & Confidence

The system provides detailed feedback on answer quality:

- **High Confidence (>0.6)**: `[High confidence answer from RoBERTa]`
- **Moderate Confidence (0.3-0.6)**: `[Answer from DistilBERT with moderate confidence: 0.45]`
- **Low Confidence (<0.3)**: `[Note: Low confidence answer. Consider asking a more specific question.]`

## 📋 Repository Structure

```
📁 Vectorization Pipeline/
├── 📄 README.md                 # This file - comprehensive documentation
├── 📄 doc-qa-local.py          # Main application - process & query documents
├── 📄 local-requirements.txt   # Python dependencies
├── 📄 setup.sh                 # Automated setup script (cross-platform)
├── 📄 local-setup-guide.md     # Detailed manual setup instructions
├── 📁 local_chroma_db/         # Vector database (created after processing)
├── � .venv/                   # Python virtual environment
└── 📄 bcm.pdf                  # Example document (Body Control Module spec)
```

## 🚀 Performance & Optimization

### For Large Documents (1000+ pages)
- **Chunk Size**: Use 1500-2000 characters for better context
- **Processing Time**: 30-60 minutes for initial vectorization
- **Memory**: Requires 4-8GB RAM for large documents
- **Storage**: ~1-3GB database size for very large documents

### Speed Optimization
```bash
# Disable OCR for faster processing (if document isn't rotated)
python doc-qa-local.py process --document doc.pdf --no-ocr

# Use simple extraction for fastest queries
python doc-qa-local.py query --llm simple
```

## 🛡️ Privacy & Security

- **No Internet Required**: All processing happens locally - successfully tested offline
- **No Data Transmission**: Documents never leave your machine
- **No API Keys**: No external service dependencies
- **Secure Storage**: Local database with no external access
- **Portable**: Can run on air-gapped systems

## 🐛 Troubleshooting - Common Issues & Solutions

### GPU/CUDA Issues (macOS)
**Problem**: `AssertionError: Torch not compiled with CUDA enabled`
**Solution**: Always use the `--cpu` flag on macOS
```bash
python doc-qa-local.py process --document file.pdf --cpu
python doc-qa-local.py query --cpu
```

### PyMuPDF Installation Issues
**Problem**: Compilation errors with climits header
**Solution**: Use pre-compiled wheels
```bash
pip install PyMuPDF --no-deps  # Uses pre-compiled wheel
```

### Out of Memory Errors
**Problem**: System runs out of RAM during processing
**Solutions**:
```bash
# Use smaller chunks
python doc-qa-local.py process --document doc.pdf --chunk-size 500 --cpu

# Force CPU usage
python doc-qa-local.py query --cpu

# Disable enhanced processing for large documents
python doc-qa-local.py process --document doc.pdf --no-enhanced --cpu
```

### OCR Dependencies Missing
**Problem**: Tesseract not found
**Solutions**:
```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# Or disable OCR
python doc-qa-local.py process --document doc.pdf --no-ocr --cpu
```

### Poor Answer Quality
**Problem**: Getting empty or low-confidence answers
**Solutions**:
```bash
# Use comprehensive mode for technical documents
python doc-qa-local.py query --cpu --comprehensive

# Try different chunk settings
python doc-qa-local.py process --document doc.pdf --chunk-size 1500 --chunk-overlap 400 --cpu

# Ensure enhanced processing is enabled (default)
python doc-qa-local.py process --document doc.pdf --cpu  # (enhanced by default)
```

### Interactive Query Issues
**Problem**: EOF errors or input issues
**Solution**: Run query mode directly without piping input
```bash
# Don't use: echo "question" | python doc-qa-local.py query
# Instead use: python doc-qa-local.py query --cpu
```

## 📊 Performance Expectations (Tested)

### Document Processing
- **Small documents (1-10 pages)**: 30 seconds - 2 minutes
- **Medium documents (100-500 pages)**: 5-15 minutes
- **Large documents (1000+ pages)**: 30-60 minutes
- **Memory usage**: 2-8GB RAM depending on document size

### Query Performance
- **Search time**: 1-3 seconds for vector search
- **Answer generation**: 2-10 seconds depending on mode
- **Memory**: 2-4GB RAM during queries

### Tested Configuration (macOS M-series)
- **Python**: 3.13.5
- **Processing**: CPU-only with --cpu flag
- **Models**: DistilBERT + RoBERTa for Q&A, all-MiniLM-L6-v2 for embeddings
- **Status**: ✅ Fully functional and tested

## 📈 Comparison with Cloud Solutions

| Feature | This System | Cloud RAG |
|---------|-------------|-----------|
| Privacy | ✅ 100% Local | ❌ Data sent to cloud |
| Cost | ✅ Free after setup | 💰 Per-query costs |
| Internet | ✅ Works offline | ❌ Requires internet |
| Setup | ⚡ Quick local setup | 🔧 API keys needed |
| Speed | ⚡ Fast local inference | 🐌 Network latency |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different document types
5. Submit a pull request

## 📄 License

MIT License - feel free to use in your projects!

---

**Need help?** Check the `local-setup-guide.md` for detailed setup instructions or open an issue for support.
