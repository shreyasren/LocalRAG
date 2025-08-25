# Complete Local Document Q&A Setup Guide

## 🎯 Features

### Handles Rotated Text
- **Automatic rotation detection**: Detects if pages are rotated 90°, 180°, or 270°
- **OCR processing**: Extracts text from images and scanned documents
- **Hybrid approach**: Uses regular extraction when possible, OCR when needed

### 100% Local - No API Keys Required
- **Local embeddings**: Uses Sentence Transformers (no OpenAI)
- **Local LLM**: Uses Ollama or HuggingFace models (no GPT)
- **Private**: Your data never leaves your machine
- **Free**: No API costs after initial setup

## 📋 Prerequisites

### 1. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils python3-pip
```

#### macOS:
```bash
brew install tesseract poppler
```

#### Windows:
- Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Download Poppler: http://blog.alivate.com.au/poppler-windows/
- Add both to your PATH

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Local LLM (Choose One)

#### Option A: Ollama (Recommended - Easiest)
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull a model (choose based on your RAM)
ollama pull llama3.2        # 3B model, needs ~4GB RAM
ollama pull mistral         # 7B model, needs ~8GB RAM  
ollama pull phi3            # 3.8B model, needs ~4GB RAM
```

#### Option B: HuggingFace Models (More Options)
The script will automatically download models when you first run it. Choose based on your hardware:

**For 8GB RAM:**
- `microsoft/DialoGPT-medium` (lightweight, fast)
- `microsoft/phi-2` (2.7B parameters)

**For 16GB RAM:**
- `microsoft/Phi-3-mini-4k-instruct` (good quality)
- `mistralai/Mistral-7B-Instruct-v0.2` (excellent quality)

**For 32GB+ RAM or GPU:**
- `meta-llama/Llama-2-13b-chat-hf` (requires approval)
- `tiiuae/falcon-7b-instruct`

## 🚀 Quick Start

### Step 1: Process Your Document (with OCR for rotated pages)
```bash
# Basic usage with Ollama
python local_vectorizer.py process --document your_document.pdf

# With custom settings
python local_vectorizer.py process \
  --document document.pdf \
  --chunk-size 1500 \
  --chunk-overlap 300
```

### Step 2: Query Your Document
```bash
# Start Q&A session
python local_vectorizer.py query
```

## 🔧 Configuration Options

### Disable OCR (faster but won't handle rotated text)
```bash
python local_vectorizer.py process --document doc.pdf --no-ocr
```

### Use Different Embedding Models
```bash
# Faster, smaller embeddings
python local_vectorizer.py process \
  --document doc.pdf \
  --embedding-model all-MiniLM-L6-v2

# Better quality embeddings (slower, needs more RAM)
python local_vectorizer.py process \
  --document doc.pdf \
  --embedding-model all-mpnet-base-v2
```

### Use HuggingFace Instead of Ollama
```bash
# Use a specific HuggingFace model
python local_vectorizer.py process \
  --document doc.pdf \
  --llm "microsoft/Phi-3-mini-4k-instruct"
```

### Force CPU Usage (if you have GPU issues)
```bash
python local_vectorizer.py process --document doc.pdf --cpu
```

## 📊 Performance Expectations

### For a 13,000-page document:

#### With OCR (handles rotated pages):
- **Processing time**: 2-4 hours
- **RAM needed**: 16-32GB recommended
- **Storage**: 2-4GB for vector database

#### Without OCR (faster):
- **Processing time**: 30-60 minutes
- **RAM needed**: 8-16GB
- **Storage**: 2-4GB for vector database

### Query Performance:
- **Search time**: 1-3 seconds
- **Answer generation**: 5-30 seconds (depends on LLM)

## 🎯 Recommended Configurations

### For Maximum Quality (32GB+ RAM):
```bash
# Process with OCR
python local_vectorizer.py process \
  --document doc.pdf \
  --embedding-model all-mpnet-base-v2 \
  --chunk-size 1500 \
  --chunk-overlap 400

# Query with Ollama (after installing llama3.2)
python local_vectorizer.py query --llm ollama
```

### For Balance (16GB RAM):
```bash
# Process
python local_vectorizer.py process \
  --document doc.pdf \
  --embedding-model all-MiniLM-L6-v2

# Query
python local_vectorizer.py query --llm ollama
```

### For Speed (8GB RAM):
```bash
# Process without OCR
python local_vectorizer.py process \
  --document doc.pdf \
  --no-ocr \
  --chunk-size 800

# Query with small model
python local_vectorizer.py query \
  --llm "microsoft/DialoGPT-medium"
```

## 🔍 Handling Rotated Pages

The OCR system automatically:
1. Detects page rotation using Tesseract's OSD (Orientation and Script Detection)
2. Rotates pages to correct orientation before extraction
3. Falls back to image-based OCR for scanned pages

To verify OCR is working:
```bash
# Test OCR installation
tesseract --version

# Process a document with verbose output
python local_vectorizer.py process --document rotated_doc.pdf
# Look for messages like "Page X rotated 90°, correcting..."
```

## 🐛 Troubleshooting

### "Tesseract not found"
- Make sure Tesseract is installed and in PATH
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

### "Out of memory"
- Use smaller chunks: `--chunk-size 500`
- Use smaller embedding model: `--embedding-model all-MiniLM-L6-v2`
- Process without OCR: `--no-ocr`
- Use CPU instead of GPU: `--cpu`

### "Ollama connection refused"
- Make sure Ollama is running: `ollama serve`
- Check if model is installed: `ollama list`
- Use HuggingFace model instead: `--llm microsoft/DialoGPT-medium`

### "Slow processing"
- Disable OCR if not needed: `--no-ocr`
- Use smaller embedding model
- Reduce chunk overlap: `--chunk-overlap 100`

### "Poor answer quality"
- Increase chunk size: `--chunk-size 2000`
- Increase chunk overlap: `--chunk-overlap 400`  
- Use better LLM: Install `ollama pull llama3.2` or `mistral`
- Retrieve more chunks: Edit code to increase `k` parameter

## 💡 Tips for 13k Page Documents

1. **First test on a smaller sample** (e.g., first 100 pages) to verify settings
2. **Run processing overnight** - it will take several hours with OCR
3. **Monitor RAM usage** - close other applications during processing
4. **Use SSD storage** for the database path for better performance
5. **Consider splitting** the document into parts if RAM is limited

## 📈 Upgrading Performance

### Add GPU Support:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Use Better Models:
```bash
# Better embeddings (needs more RAM)
--embedding-model sentence-transformers/all-mpnet-base-v2

# Better LLM (needs more RAM/GPU)
ollama pull llama2:13b
ollama pull mixtral
```

## 🔒 Privacy & Security

- ✅ **100% offline**: No internet connection required after setup
- ✅ **No telemetry**: ChromaDB telemetry is disabled
- ✅ **Local storage**: All data stored in local directory
- ✅ **No API keys**: No external services used
- ✅ **Portable**: Can run on air-gapped systems