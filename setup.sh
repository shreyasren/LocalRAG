#!/bin/bash

# Local Document Q&A System - Automated Setup Script
# Supports Ubuntu/Debian, macOS, and provides instructions for Windows

set -e

echo "======================================"
echo "Local Document Q&A System Setup"
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
    
    # Create requirements.txt if it doesn't exist
    cat > requirements_local.txt << 'EOF'
# Core dependencies
sentence-transformers==2.2.2
chromadb==0.4.22
torch>=2.0.0
transformers==4.36.0
tqdm==4.66.1

# OCR dependencies for rotated text
pytesseract==0.3.10
pdf2image==1.16.3
PyMuPDF==1.23.8
Pillow==10.1.0

# Document processing
langchain==0.1.0
numpy==1.24.3

# Optional: For better PDF handling
pypdf==3.17.4

# Optional: For Ollama integration
requests==2.31.0
EOF
    
    pip install -r requirements_local.txt
    
    echo "✅ Python dependencies installed"
    echo ""
}

# Install Ollama
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
    echo "4) Skip (use HuggingFace models instead)"
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
            echo "Skipping Ollama model download"
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
    
    # Test Python packages
    python3 -c "import sentence_transformers; print('✅ Sentence Transformers installed')" 2>/dev/null || echo "❌ Sentence Transformers not installed"
    python3 -c "import chromadb; print('✅ ChromaDB installed')" 2>/dev/null || echo "❌ ChromaDB not installed"
    python3 -c "import pytesseract; print('✅ PyTesseract installed')" 2>/dev/null || echo "❌ PyTesseract not installed"
    python3 -c "import torch; print(f'✅ PyTorch installed (GPU: {torch.cuda.is_available()})')" 2>/dev/null || echo "❌ PyTorch not installed"
    
    # Test Ollama
    if command_exists ollama; then
        echo "✅ Ollama installed"
        ollama list 2>/dev/null | grep -q "NAME" && echo "✅ Ollama models available" || echo "⚠️  No Ollama models installed"
    else
        echo "⚠️  Ollama not installed (optional)"
    fi
    
    echo ""
}

# Create example script
create_example_script() {
    echo "📝 Creating example usage script..."
    
    cat > run_example.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Example commands
echo "Example Usage Commands:"
echo ""
echo "1. Process a document with OCR (handles rotated pages):"
echo "   python local_vectorizer.py process --document your_document.pdf"
echo ""
echo "2. Process without OCR (faster):"
echo "   python local_vectorizer.py process --document your_document.pdf --no-ocr"
echo ""
echo "3. Start Q&A session:"
echo "   python local_vectorizer.py query"
echo ""
echo "4. Process with custom settings:"
echo "   python local_vectorizer.py process --document doc.pdf --chunk-size 1500 --chunk-overlap 300"
echo ""

# Check if user wants to process a document
read -p "Do you have a document to process now? (y/n): " process_now

if [ "$process_now" == "y" ] || [ "$process_now" == "Y" ]; then
    read -p "Enter the path to your document: " doc_path
    
    if [ -f "$doc_path" ]; then
        echo "Processing document..."
        python local_vectorizer.py process --document "$doc_path"
        
        echo ""
        read -p "Document processed! Start Q&A session now? (y/n): " start_qa
        
        if [ "$start_qa" == "y" ] || [ "$start_qa" == "Y" ]; then
            python local_vectorizer.py query
        fi
    else
        echo "File not found: $doc_path"
    fi
fi
EOF
    
    chmod +x run_example.sh
    echo "✅ Created run_example.sh"
    echo ""
}

# Main installation flow
main() {
    echo "This script will set up the Local Document Q&A System"
    echo "It will install:"
    echo "  - Tesseract OCR (for handling rotated text)"
    echo "  - Python dependencies"
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
    echo "Ollama provides the best quality local LLM responses."
    read -p "Install Ollama? (y/n): " install_ollama_choice
    
    if [ "$install_ollama_choice" == "y" ] || [ "$install_ollama_choice" == "Y" ]; then
        install_ollama
    else
        echo "Skipping Ollama installation. You can use HuggingFace models instead."
    fi
    
    # Test installation
    test_installation
    
    # Create example script
    create_example_script
    
    # Final instructions
    echo "======================================"
    echo "✅ Installation Complete!"
    echo "======================================"
    echo ""
    echo "To get started:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Process a document: python local_vectorizer.py process --document your_doc.pdf"
    echo "3. Ask questions: python local_vectorizer.py query"
    echo ""
    echo "Or run: ./run_example.sh for guided usage"
    echo ""
    echo "For more options, see: python local_vectorizer.py --help"
    echo ""
}

# Run main installation
main