#!/bin/bash

# LocalRAG - Automated Setup Script
# Tested and verified on macOS with Python 3.13.5
# Supports Ubuntu/Debian, macOS, and provides instructions for Windows

set -e

echo "======================================"
echo "LocalRAG - Enhanced Document Q&A Setup"
echo "======================================"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    DISTRO=$(lsb_release -si 2>/dev/null || echo "unknown")
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies
install_system_deps() {
    echo "📦 Installing system dependencies..."
    
    if [ "$OS" == "linux" ]; then
        echo "Installing Tesseract OCR and Poppler..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils python3-pip python3-venv
        
    elif [ "$OS" == "macos" ]; then
        # Check if Homebrew is installed
        if ! command_exists brew; then
            echo "Homebrew not found. Installing Homebrew first..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        echo "Installing Tesseract OCR and Poppler..."
        brew install tesseract poppler
        
    elif [ "$OS" == "windows" ]; then
        echo "⚠️  Windows detected. Please install manually:"
        echo "1. Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
        echo "2. Poppler: http://blog.alivate.com.au/poppler-windows/"
        echo "3. Add both to your system PATH"
        echo ""
        read -p "Press Enter once you've installed these dependencies..."
    else
        echo "⚠️  Unknown OS. Please install Tesseract and Poppler manually."
        exit 1
    fi
    
    echo "✅ System dependencies installed"
    echo ""
}

# Create Python virtual environment
setup_python_env() {
    echo "🐍 Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    echo "✅ Python environment ready"
    echo ""
}

# Install Python dependencies
install_python_deps() {
    echo "📚 Installing Python dependencies..."
    echo "This may take a few minutes..."
    
    # Install dependencies from local-requirements.txt
    pip install -r local-requirements.txt
    
    echo "✅ Python dependencies installed"
    echo ""
}

# Install Ollama (optional)
install_ollama() {
    echo "🤖 Setting up Ollama for local LLM..."
    
    if command_exists ollama; then
        echo "Ollama already installed"
    else
        if [ "$OS" == "linux" ] || [ "$OS" == "macos" ]; then
            echo "Installing Ollama..."
            curl -fsSL https://ollama.ai/install.sh | sh
        elif [ "$OS" == "windows" ]; then
            echo "Please download and install Ollama from: https://ollama.ai/download"
            read -p "Press Enter once you've installed Ollama..."
        fi
    fi
    
    # Start Ollama service
    echo "Starting Ollama service..."
    if [ "$OS" == "linux" ] || [ "$OS" == "macos" ]; then
        # Start in background
        nohup ollama serve > /dev/null 2>&1 &
        sleep 3
    fi
    
    # Pull a model
    echo ""
    echo "Choose an LLM model to download:"
    echo "1) llama3.2 (3B parameters, ~2GB download, needs 4GB RAM)"
    echo "2) phi3 (3.8B parameters, ~2.3GB download, needs 4GB RAM)"
    echo "3) mistral (7B parameters, ~4GB download, needs 8GB RAM)"
    echo "4) Skip (use Transformer models instead - recommended for testing)"
    echo ""
    read -p "Enter choice (1-4): " model_choice
    
    case $model_choice in
        1)
            echo "Downloading llama3.2..."
            ollama pull llama3.2
            ;;
        2)
            echo "Downloading phi3..."
            ollama pull phi3
            ;;
        3)
            echo "Downloading mistral..."
            ollama pull mistral
            ;;
        4)
            echo "Skipping Ollama model download - using Transformer models"
            ;;
        *)
            echo "Invalid choice, skipping model download"
            ;;
    esac
    
    echo "✅ Ollama setup complete"
    echo ""
}

# Test installation
test_installation() {
    echo "🧪 Testing installation..."
    
    # Test Tesseract
    if command_exists tesseract; then
        echo "✅ Tesseract installed: $(tesseract --version | head -n1)"
    else
        echo "❌ Tesseract not found"
    fi
    
    # Activate venv for testing
    source venv/bin/activate
    
    # Test Python packages
    python3 -c "import sentence_transformers; print('✅ Sentence Transformers installed')" 2>/dev/null || echo "❌ Sentence Transformers not installed"
    python3 -c "import chromadb; print('✅ ChromaDB installed')" 2>/dev/null || echo "❌ ChromaDB not installed"
    python3 -c "import pytesseract; print('✅ PyTesseract installed')" 2>/dev/null || echo "❌ PyTesseract not installed"
    python3 -c "import torch; print(f'✅ PyTorch installed')" 2>/dev/null || echo "❌ PyTorch not installed"
    python3 -c "import fitz; print('✅ PyMuPDF installed')" 2>/dev/null || echo "❌ PyMuPDF not installed"
    
    # Test main script
    python3 doc-qa-local.py --help > /dev/null 2>&1 && echo "✅ Main script working" || echo "❌ Main script has issues"
    
    # Test Ollama
    if command_exists ollama; then
        echo "✅ Ollama installed"
        ollama list 2>/dev/null | grep -q "NAME" && echo "✅ Ollama models available" || echo "⚠️  No Ollama models installed"
    else
        echo "⚠️  Ollama not installed (optional)"
    fi
    
    echo ""
}

# Create example usage guide
create_usage_guide() {
    echo "📝 Creating usage examples..."
    
    cat > QUICK_START.md << 'EOF'
# LocalRAG Quick Start Guide

## ✅ System Ready!

Your LocalRAG system is installed and ready to use.

## 🚀 Quick Commands

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Process a Document
```bash
# Basic processing (always use --cpu on macOS)
python doc-qa-local.py process --document your_document.pdf --cpu

# For technical documents (recommended)
python doc-qa-local.py process --document technical_manual.pdf --cpu
```

### 3. Query Documents
```bash
# Start interactive Q&A
python doc-qa-local.py query --cpu

# Use comprehensive mode for detailed answers
python doc-qa-local.py query --cpu --comprehensive
```

## 📊 Example Session

```bash
# Process test document
python doc-qa-local.py process --document test_document.txt --cpu

# Start Q&A
python doc-qa-local.py query --cpu

# Example interaction:
📝 Your Question: What is machine learning?
📚 ANSWER: Machine learning (ML) is a method of data analysis...
```

## ⚠️ Important Notes

- **Always use `--cpu` flag on macOS** to avoid GPU errors
- **First run downloads models** - this may take time
- **Large documents** may take 10-30 minutes to process
- **Memory usage**: 2-8GB RAM depending on document size

## 🆘 If Something Goes Wrong

```bash
# Check if main script works
python doc-qa-local.py --help

# Reinstall dependencies if needed
pip install -r local-requirements.txt

# For PyMuPDF issues
pip install PyMuPDF --no-deps
```

## 📚 More Information

See README.md for complete documentation and troubleshooting.
EOF
    
    echo "✅ Created QUICK_START.md"
    echo ""
}

# Main installation flow
main() {
    echo "This script will set up the LocalRAG Enhanced Document Q&A System"
    echo "It will install:"
    echo "  - Tesseract OCR (for handling rotated text)"
    echo "  - Python dependencies (sentence-transformers, ChromaDB, etc.)"
    echo "  - Ollama (optional, for local LLM)"
    echo ""
    read -p "Continue with installation? (y/n): " confirm
    
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Installation cancelled"
        exit 0
    fi
    
    echo ""
    
    # Run installation steps
    install_system_deps
    setup_python_env
    install_python_deps
    
    # Ask about Ollama
    echo "Ollama provides high-quality local LLM responses."
    echo "Note: The system works great with built-in Transformer models too!"
    read -p "Install Ollama? (y/n): " install_ollama_choice
    
    if [ "$install_ollama_choice" == "y" ] || [ "$install_ollama_choice" == "Y" ]; then
        install_ollama
    else
        echo "Skipping Ollama installation. You can use Transformer models (recommended for testing)."
    fi
    
    # Test installation
    test_installation
    
    # Create usage guide
    create_usage_guide
    
    # Final instructions
    echo "======================================"
    echo "✅ LocalRAG Installation Complete!"
    echo "======================================"
    echo ""
    echo "🚀 To get started:"
    echo "1. Activate environment: source venv/bin/activate"
    echo "2. Process a document: python doc-qa-local.py process --document your_doc.pdf --cpu"
    echo "3. Ask questions: python doc-qa-local.py query --cpu"
    echo ""
    echo "📖 Quick reference: cat QUICK_START.md"
    echo "📚 Full documentation: cat README.md"
    echo ""
    echo "⚠️  Remember: Always use --cpu flag on macOS!"
    echo ""
}

# Run main installation
main