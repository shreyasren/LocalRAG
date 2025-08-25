# LocalRAG - Project Structure

## 📁 Repository Overview

```
LocalRAG/
├── 📄 README.md                    # Main documentation
├── 📄 LICENSE                      # MIT License
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 ENHANCED_SYSTEM.md           # Technical enhancement documentation
├── 🐍 doc-qa-local.py              # Main application script
├── 🔧 enhanced_processor.py        # Enhanced document processing module
├── 📋 local-requirements.txt       # Python dependencies
├── 📋 local-setup-guide.md         # Detailed setup instructions
├── 🔧 setup.sh                     # Automated setup script
├── 📁 local_chroma_db/             # Vector database storage
└── 📄 bcm.pdf                      # Sample document (if present)
```

## 🐍 Core Python Files

### `doc-qa-local.py` - Main Application
**Purpose**: Primary script for document processing and Q&A functionality

**Key Classes:**
- `LocalDocumentVectorizer`: Main class handling document processing, vector storage, and Q&A
  - Document loading and chunking
  - Vector database management
  - Multiple answer generation modes
  - Interactive Q&A interface

**Key Features:**
- Command-line interface with extensive options
- Multiple LLM integration (Transformers, Ollama, Simple, Comprehensive)
- OCR support for rotated/scanned documents
- Enhanced vs basic processing modes
- GPU/CPU optimization

**Usage Patterns:**
```bash
# Document processing
python doc-qa-local.py process --document file.pdf [options]

# Q&A interface  
python doc-qa-local.py query [options]
```

### `enhanced_processor.py` - Technical Document Enhancement
**Purpose**: Specialized processing for technical documents

**Key Classes:**
- `TechnicalDocumentProcessor`: 
  - Smart text cleaning and section extraction
  - Technical pattern recognition
  - Section-aware chunking
  
- `EnhancedVectorSearch`:
  - Context-aware search with expansion
  - Technical relevance scoring
  - Multi-chunk analysis

**Key Functions:**
- `create_enhanced_qa_context()`: Optimized context creation for Q&A
- Technical content prioritization
- Signal and component extraction

## 📋 Configuration Files

### `local-requirements.txt`
Python dependencies including:
- `sentence-transformers`: Embedding models
- `chromadb`: Vector database
- `transformers`: Transformer models for Q&A
- `torch`: PyTorch backend
- `PyMuPDF`: PDF processing
- `pytesseract`: OCR functionality
- `Pillow`: Image processing
- `tqdm`: Progress bars
- `langchain`: Document processing utilities

### `setup.sh`
Automated setup script that:
- Creates Python virtual environment
- Installs Python dependencies
- Installs system dependencies (macOS)
- Provides setup verification

## 📁 Data Storage

### `local_chroma_db/`
ChromaDB persistent storage containing:
- `chroma.sqlite3`: Main database file
- `{collection-id}/`: Vector collection data
  - `data_level0.bin`: Vector data
  - `header.bin`: Collection metadata
  - `index_metadata.pickle`: Index configuration
  - `length.bin`: Document lengths
  - `link_lists.bin`: Hierarchical index

## 🔧 System Architecture

### Document Processing Pipeline
```
PDF/Text Input 
    ↓
OCR Processing (if needed)
    ↓
Enhanced/Basic Chunking
    ↓
Embedding Generation
    ↓
Vector Database Storage
```

### Query Processing Pipeline
```
User Question
    ↓
Question Embedding
    ↓
Enhanced/Standard Vector Search
    ↓
Context Creation
    ↓
Answer Generation (Transformer/Comprehensive/Simple/Ollama)
    ↓
Response with Sources
```

### Answer Generation Modes

1. **Standard Mode (Default)**:
   - DistilBERT + RoBERTa extractive Q&A
   - Auto-enhancement for short responses
   - Confidence scoring

2. **Comprehensive Mode**:
   - Multi-chunk analysis
   - Technical term extraction
   - Structured response generation
   - Signal and component identification

3. **Ollama Mode**:
   - External LLM integration
   - Human-like responses
   - Requires Ollama installation

4. **Simple Mode**:
   - Keyword-based extraction
   - Fastest processing
   - Basic relevance scoring

## 🔄 Enhancement Features

### Technical Document Processing
- **Smart Chunking**: Section-aware document splitting
- **Pattern Recognition**: Identifies technical terms and signals
- **Context Expansion**: Groups related information
- **Relevance Scoring**: Prioritizes technical content

### Command Line Interface
```bash
# Processing options
--document FILE         # Document to process
--chunk-size INT        # Chunk size in characters
--chunk-overlap INT     # Overlap between chunks
--no-ocr               # Disable OCR processing
--no-enhanced          # Disable enhanced processing
--cpu                  # Force CPU usage

# Query options
--llm MODEL            # transformers/ollama/simple/comprehensive
--comprehensive        # Force comprehensive mode
--no-enhanced          # Disable enhanced search
--db-path PATH         # Database location
```

## 🚀 Extension Points

### Adding New LLM Models
Modify `_initialize_llm()` method in `LocalDocumentVectorizer`

### Customizing Document Processing
Extend `TechnicalDocumentProcessor` class for domain-specific processing

### Adding New Answer Generation Modes
Implement new methods in the answer generation section

### Extending Search Capabilities
Enhance `EnhancedVectorSearch` class for specialized search patterns

## 🔒 Privacy & Security

- **100% Local Processing**: No external API calls
- **Data Isolation**: All data stays on local machine
- **Temporary Files**: Cleaned up automatically
- **Secure Storage**: Local vector database with no cloud sync

## 📚 Documentation Files

- `README.md`: User guide and feature overview
- `ENHANCED_SYSTEM.md`: Technical implementation details
- `local-setup-guide.md`: Detailed installation instructions
- `PROJECT_STRUCTURE.md`: This architectural overview
