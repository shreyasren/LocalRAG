# LocalRAG - Enhanced Document Q&A System

A powerful, **completely local** RAG (Retrieval-Augmented Generation) system with **advanced technical document processing** that answers questions without requiring any external APIs or internet connection. Perfect for handling sensitive technical documents while maintaining privacy and data security.

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

### Enhanced Features (NEW!)
- **🔬 Technical Document Processing**: Specialized handling for technical manuals, specifications, and engineering documents
- **📋 Comprehensive Answer Generation**: Detailed, structured responses with signal lists and technical analysis
- **🎚 Multiple Answer Modes**: Standard, comprehensive, and simple modes for different use cases
- **🔍 Context-Aware Search**: Enhanced vector search that understands technical document structure
- **⚙ Command-Line Flexibility**: Extensive options for customizing processing and answer generation

## 🛠 Installation & Setup

### Quick Setup

1. **Clone and navigate to the repository**
2. **Run the automated setup script:**
   ```bash
   ./setup.sh
   ```

### Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r local-requirements.txt
   ```

3. **Install system dependencies (macOS):**
   ```bash
   brew install tesseract poppler
   ```

## 🎯 Usage

### Step 1: Process Your Document

First, vectorize your document (one-time setup per document):

### Step 1: Process Your Document

First, vectorize your document (one-time setup per document):

**Basic Processing:**
```bash
python doc-qa-local.py process --document path/to/your/document.pdf --cpu
```

**Enhanced Processing (Recommended for Technical Documents):**
```bash
python doc-qa-local.py process --document path/to/your/document.pdf --cpu
```

**Processing Options:**
- `--chunk-size 1000`: Text chunk size in characters (default: 1000)
- `--chunk-overlap 200`: Overlap between chunks (default: 200)
- `--db-path ./local_chroma_db`: Database location (default: ./local_chroma_db)
- `--embedding-model all-MiniLM-L6-v2`: Embedding model (default: all-MiniLM-L6-v2)
- `--no-ocr`: Disable OCR for faster processing (skip if document is rotated)
- `--no-enhanced`: Disable enhanced technical document processing (use basic chunking)
- `--cpu`: Force CPU usage even if GPU is available

**Example for large technical documents:**
```bash
python doc-qa-local.py process \
  --document technical_manual.pdf \
  --chunk-size 1500 \
  --chunk-overlap 300 \
  --cpu
```

### Step 2: Query Your Document

After processing, start the interactive Q&A session:

**Standard Mode (Transformer Models with Auto-Enhancement):**
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

**Q&A Options:**
- `--llm transformers`: Use transformer models (default, recommended)
- `--llm ollama`: Use Ollama (requires Ollama installation)
- `--llm simple`: Use simple keyword matching (fastest)
- `--llm comprehensive`: Use detailed analysis (same as --comprehensive)
- `--comprehensive`: Force comprehensive answer generation for longer responses
- `--no-enhanced`: Disable enhanced search and processing
- `--cpu`: Use CPU processing
- `--db-path ./local_chroma_db`: Specify database location

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

## � Enhanced Technical Document Processing

### Key Improvements for Technical Documents

The system includes specialized enhancements for technical documents like manuals, specifications, and engineering documentation:

#### **Smart Chunking**
- **Section-Aware Splitting**: Preserves technical sections and subsections
- **Header Recognition**: Identifies and maintains document structure
- **Technical Pattern Detection**: Recognizes signal names, commands, and technical terminology

#### **Enhanced Vector Search**
- **Context Expansion**: Groups related technical information
- **Relevance Scoring**: Prioritizes technical content over headers/metadata
- **Multi-Chunk Analysis**: Combines information from multiple relevant sections

#### **Comprehensive Answer Generation**
- **Signal Extraction**: Automatically identifies command signals and technical terms
- **Structured Responses**: Organizes answers with clear sections and bullet points
- **Technical Context**: Provides detailed explanations with component relationships

### Example: Technical Document Analysis

```bash
# Process a technical manual with enhanced features
python doc-qa-local.py process --document technical_manual.pdf --cpu

# Query with comprehensive analysis for detailed responses
python doc-qa-local.py query --cpu --comprehensive

# Example interaction:
📝 Your Question: give me all the signals used in headlights

📚 ANSWER:
**Headlight Control Signals:**
• AV_HeadLight_Rq - Autonomous vehicle headlight request
• HEADLGHTCTL_D_RQ_CHNL - Headlight control request channel  
• HeadLightSW_Rq - Headlight switch request
• FuSA_Headlight_Status_Arb - Functional safety headlight status arbitrator

**Related Components:**
• Low Beam Control
• High Beam Control  
• Turn Indicator Lights
• Hazard Lights
• Position/Park Lamps

**Detailed Information:**
• The CAN signals for AV lighting requests are arbitrated with active driver determination
• Final arbitrated signals interface with base Exterior Lighting Control
• System supports both autonomous and human control modes
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

- **No Internet Required**: All processing happens locally
- **No Data Transmission**: Documents never leave your machine
- **No API Keys**: No external service dependencies
- **Secure Storage**: Local database with no external access

## 🐛 Troubleshooting

### Common Issues

**OCR Dependencies Missing:**
```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# Windows
# Use installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Out of Memory:**
```bash
# Use smaller chunks
python doc-qa-local.py process --document doc.pdf --chunk-size 500

# Force CPU usage
python doc-qa-local.py query --cpu
```

**Poor Answer Quality:**
```bash
# Try different models
python doc-qa-local.py query --llm transformers  # vs --llm simple

# Increase chunk overlap
python doc-qa-local.py process --chunk-overlap 400
```

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
