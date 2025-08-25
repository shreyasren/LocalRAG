#!/usr/bin/env python3
"""
LocalRAG - Enhanced Document Q&A System

A powerful, completely local RAG (Retrieval-Augmented Generation) system with advanced 
technical document processing capabilities. Processes documents and answers questions 
without requiring any external APIs or internet connection.

Key Features:
- 100% Local Processing
- Enhanced Technical Document Processing  
- Multiple Answer Generation Modes (Standard, Comprehensive, Ollama, Simple)
- OCR Support for Scanned Documents
- Interactive Q&A Interface
- GPU/CPU Optimization
- Smart Chunking and Context-Aware Search

Usage:
    # Process a document
    python doc-qa-local.py process --document file.pdf --cpu
    
    # Query with standard mode
    python doc-qa-local.py query --cpu
    
    # Query with comprehensive mode for detailed responses
    python doc-qa-local.py query --cpu --comprehensive

Author: LocalRAG Team
License: MIT
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import argparse
import warnings
warnings.filterwarnings("ignore")

# Disable ChromaDB telemetry to avoid version compatibility errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# Core dependencies
import numpy as np
from PIL import Image
import pytesseract
import pdf2image
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm
import logging

# Import enhanced processor
from enhanced_processor import TechnicalDocumentProcessor, EnhancedVectorSearch, create_enhanced_qa_context

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalDocumentVectorizer:
    """Local document vectorization and Q&A without external APIs"""
    
    def __init__(self, 
                 db_path: str = "./local_chroma_db",
                 collection_name: str = "local_documents",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "microsoft/Phi-3-mini-4k-instruct",
                 use_gpu: bool = None,
                 use_ocr: bool = True,
                 use_enhanced: bool = True):
        """
        Initialize the Local Document Vectorizer
        
        Args:
            db_path: Path to store the ChromaDB database
            collection_name: Name for the vector collection
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
            embedding_model: Sentence transformer model for embeddings
            llm_model: Local LLM model for Q&A (or 'ollama' to use Ollama)
            use_gpu: Whether to use GPU (auto-detect if None)
            use_ocr: Whether to use OCR for rotated/image-based text
            use_enhanced: Whether to use enhanced technical document processing
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ocr = use_ocr
        self.use_enhanced = use_enhanced
        
        # Detect GPU availability
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu
        
        logger.info(f"Using GPU: {self.use_gpu}")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        if self.use_gpu:
            self.embedding_model = self.embedding_model.cuda()
        
        # Initialize LLM
        self.llm_model_name = llm_model
        self.llm = None
        self.tokenizer = None
        self.use_comprehensive = False  # Initialize comprehensive mode flag
        
        # Initialize enhanced processor for technical documents based on flag
        if self.use_enhanced:
            try:
                self.enhanced_processor = TechnicalDocumentProcessor()
                self.enhanced_search = None  # Will be initialized after vector store is created
                self.use_enhanced_processing = True
                print("Enhanced processing enabled")
            except Exception as e:
                print(f"Enhanced processing not available: {e}")
                self.enhanced_processor = None
                self.enhanced_search = None
                self.use_enhanced_processing = False
        else:
            print("Enhanced processing disabled by user")
            self.enhanced_processor = None
            self.enhanced_search = None
            self.use_enhanced_processing = False
        self._initialize_llm()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
    
    def _initialize_llm(self):
        """Initialize the local LLM"""
        if self.llm_model_name == "ollama":
            # Use Ollama (requires Ollama to be installed and running)
            logger.info("Using Ollama for LLM (make sure Ollama is running)")
            self.use_ollama = True
            self.use_transformers = False
        elif self.llm_model_name == "simple":
            # Use simple answer extraction
            logger.info("Using simple answer extraction (no complex LLM)")
            self.use_ollama = False
            self.use_transformers = False
            self.llm = None
            self.tokenizer = None
        elif self.llm_model_name == "comprehensive":
            # Use comprehensive answer generation only
            logger.info("Using comprehensive answer generation for detailed responses")
            self.use_ollama = False
            self.use_transformers = False
            self.use_comprehensive = True
            self.llm = None
            self.tokenizer = None
        else:
            # Use transformers for better Q&A
            logger.info("Initializing transformer models for better Q&A...")
            self.use_ollama = False
            self.use_transformers = True
            self._initialize_transformer_models()
    
    def _initialize_transformer_models(self):
        """Initialize transformer models for better Q&A"""
        try:
            device = 0 if self.use_gpu and torch.cuda.is_available() else -1
            
            # Initialize Q&A models
            logger.info("Loading DistilBERT Q&A model...")
            self.qa_distilbert = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=device
            )
            
            logger.info("Loading RoBERTa Q&A model...")
            self.qa_roberta = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=device
            )
            
            logger.info("✅ Transformer models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load transformer models: {e}")
            logger.info("Falling back to simple answer extraction")
            self.use_transformers = False
    
    def extract_text_with_ocr(self, pdf_path: str, dpi: int = 200) -> List[Tuple[str, int]]:
        """
        Extract text from PDF using OCR, handling rotated pages
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for PDF to image conversion
            
        Returns:
            List of (text, page_number) tuples
        """
        logger.info("Using OCR to extract text (handles rotated pages)...")
        
        texts = []
        
        # First try PyMuPDF for regular text extraction
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in tqdm(range(len(pdf_document)), desc="Processing pages"):
                page = pdf_document[page_num]
                
                # Try regular text extraction first
                text = page.get_text()
                
                # If text is too short, likely needs OCR
                if len(text.strip()) < 50:
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Auto-detect and correct rotation
                    try:
                        # Try OCR with rotation detection
                        osd = pytesseract.image_to_osd(img)
                        rotation = int(osd.split('Rotate: ')[1].split('\n')[0])
                        
                        if rotation != 0:
                            logger.info(f"Page {page_num + 1} rotated {rotation}°, correcting...")
                            img = img.rotate(-rotation, expand=True)
                    except:
                        pass
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(img)
                
                if text.strip():
                    texts.append((text, page_num + 1))
            
            pdf_document.close()
            
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}, falling back to full OCR")
            
            # Fallback to pdf2image + OCR
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
            
            for page_num, img in enumerate(tqdm(images, desc="OCR processing")):
                # Auto-detect and correct rotation
                try:
                    osd = pytesseract.image_to_osd(img)
                    rotation = int(osd.split('Rotate: ')[1].split('\n')[0])
                    
                    if rotation != 0:
                        logger.info(f"Page {page_num + 1} rotated {rotation}°, correcting...")
                        img = img.rotate(-rotation, expand=True)
                except:
                    pass
                
                # Perform OCR
                text = pytesseract.image_to_string(img)
                if text.strip():
                    texts.append((text, page_num + 1))
        
        logger.info(f"Extracted text from {len(texts)} pages")
        return texts
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document from file with OCR support
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading document: {file_path}")
        
        documents = []
        
        if file_path.suffix.lower() == '.pdf' and self.use_ocr:
            # Use OCR for PDFs (handles rotated text)
            texts = self.extract_text_with_ocr(str(file_path))
            
            for text, page_num in texts:
                doc = Document(
                    page_content=text,
                    metadata={"source": str(file_path), "page": page_num}
                )
                documents.append(doc)
        else:
            # For non-PDF files, use simple text extraction
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                doc = Document(
                    page_content=text,
                    metadata={"source": str(file_path)}
                )
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} pages/sections")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using enhanced processor for technical documents"""
        logger.info("Splitting documents into chunks...")
        
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            # Try enhanced chunking first for technical documents if available and enabled
            if self.use_enhanced_processing and self.enhanced_processor:
                try:
                    enhanced_chunks = self.enhanced_processor.create_enhanced_chunks(
                        doc.page_content, 
                        chunk_size=self.chunk_size, 
                        chunk_overlap=self.chunk_overlap
                    )
                    
                    # Preserve original metadata and add document source
                    for chunk in enhanced_chunks:
                        if not chunk.metadata:
                            chunk.metadata = {}
                        chunk.metadata.update(doc.metadata)
                        
                    all_chunks.extend(enhanced_chunks)
                    
                except Exception as e:
                    logger.warning(f"Enhanced chunking failed, falling back to standard chunking: {e}")
                    # Fallback to standard chunking
                    doc_chunks = self.text_splitter.split_documents([doc])
                    all_chunks.extend(doc_chunks)
            else:
                # Use standard chunking when enhanced processing is disabled
                doc_chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        return all_chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts using local model"""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def create_vectorstore(self, chunks: List[Document], batch_size: int = 100):
        """Create or update vector store with document chunks"""
        logger.info("Creating vector store...")
        
        # Get or create collection
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        # Process in batches
        for i in tqdm(range(0, len(chunks), batch_size), desc="Vectorizing chunks"):
            batch = chunks[i:i + batch_size]
            
            # Extract texts and metadata
            texts = [chunk.page_content for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            ids = [f"chunk_{i+j}" for j in range(len(batch))]
            
            # Create embeddings
            embeddings = self.create_embeddings(texts)
            
            # Add to collection
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        self.collection = collection
        logger.info(f"Vector store created with {len(chunks)} chunks")
        
        # Initialize enhanced search after collection is created (if enabled)
        if self.use_enhanced_processing:
            self.enhanced_search = EnhancedVectorSearch(self.collection, self.embedding_model)
    
    def load_vectorstore(self):
        """Load existing vector store from disk"""
        logger.info(f"Loading vector store from {self.db_path}")
        
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info("Vector store loaded successfully")
            
            # Initialize enhanced search after collection is loaded (if enabled)
            if self.use_enhanced_processing:
                self.enhanced_search = EnhancedVectorSearch(self.collection, self.embedding_model)
        except Exception as e:
            raise ValueError(f"Could not load collection: {e}")
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        # Create query embedding
        query_embedding = self.create_embeddings([query])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else 0
            })
        
        return formatted_results
    
    def generate_answer_ollama(self, question: str, context: str) -> str:
        """Generate answer using Ollama"""
        import requests
        
        prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.2',  # or 'mistral', 'phi3', etc.
                    'prompt': prompt,
                    'stream': False
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return "Error: Could not connect to Ollama. Make sure it's running."
        except Exception as e:
            return f"Error with Ollama: {str(e)}. Make sure Ollama is installed and running."
    
    def generate_answer_transformers(self, question: str, context: str) -> str:
        """Generate answer using transformer models (better quality)"""
        if not self.use_transformers:
            return "Transformer models not available"
        
        try:
            # Truncate context if too long
            max_length = 3000  # Safe limit for most models
            if len(context) > max_length:
                context = context[:max_length] + "..."
            
            results = []
            
            # Try RoBERTa (usually better quality)
            if hasattr(self, 'qa_roberta'):
                try:
                    roberta_result = self.qa_roberta(
                        question=question,
                        context=context,
                        max_answer_len=200,  # Allow longer answers
                        handle_impossible_answer=True
                    )
                    results.append({
                        'answer': roberta_result['answer'],
                        'confidence': roberta_result['score'],
                        'model': 'RoBERTa'
                    })
                except Exception as e:
                    logger.warning(f"RoBERTa failed: {e}")
            
            # Try DistilBERT (faster, still good)
            if hasattr(self, 'qa_distilbert'):
                try:
                    distilbert_result = self.qa_distilbert(
                        question=question,
                        context=context,
                        max_answer_len=200  # Allow longer answers
                    )
                    results.append({
                        'answer': distilbert_result['answer'],
                        'confidence': distilbert_result['score'],
                        'model': 'DistilBERT'
                    })
                except Exception as e:
                    logger.warning(f"DistilBERT failed: {e}")
            
            if not results:
                return "No transformer models were able to generate an answer."
            
            # Get the highest confidence answer
            best_result = max(results, key=lambda x: x['confidence'])
            
            # Enhance the answer by adding relevant context
            enhanced_answer = self._enhance_answer_with_context(
                best_result['answer'], 
                question, 
                context,
                best_result['confidence'],
                best_result['model']
            )
            
            return enhanced_answer
                
        except Exception as e:
            logger.error(f"Error in transformer answer generation: {e}")
            return f"Error generating answer with transformers: {str(e)}. Falling back to simple extraction."
    
    def _enhance_answer_with_context(self, base_answer: str, question: str, context: str, confidence: float, model: str) -> str:
        """Enhance the base answer with additional relevant context"""
        
        # If confidence is very low, provide more context
        if confidence < 0.3:
            # Split context into sentences and find relevant ones
            sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
            
            # Look for sentences containing key terms from the question
            question_words = set(question.lower().split())
            relevant_sentences = []
            
            for sentence in sentences[:10]:  # Check first 10 sentences
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))
                if overlap >= 2:  # At least 2 word overlap
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                enhanced_context = '. '.join(relevant_sentences[:3])  # Top 3 relevant sentences
                return f"{base_answer}\n\nAdditional context: {enhanced_context}\n\n[Answer from {model} with low confidence: {confidence:.2f}. Additional context provided.]"
            else:
                return f"{base_answer}\n\n[Answer from {model} with low confidence: {confidence:.2f}. Consider asking a more specific question.]"
        
        elif confidence < 0.6:
            return f"{base_answer}\n\n[Answer from {model} with moderate confidence: {confidence:.2f}]"
        else:
            # Even for high confidence, add some context if the answer is very short
            if len(base_answer) < 50:
                # Find the sentence containing the answer in the context
                context_sentences = [s.strip() for s in context.split('.') if base_answer.lower() in s.lower()]
                if context_sentences:
                    full_sentence = context_sentences[0]
                    return f"{base_answer}\n\nFull context: {full_sentence}\n\n[High confidence answer from {model}]"
            
            return f"{base_answer}\n\n[High confidence answer from {model}]"
    
    def generate_simple_answer(self, question: str, context: str) -> str:
        """Generate a simple answer by extracting the most relevant part of context"""
        # Split context into sentences
        sentences = context.replace('\n', ' ').split('. ')
        
        # Convert question to lowercase for matching
        question_words = question.lower().split()
        
        # Find sentences that contain the most question words
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            sentence_lower = sentence.lower()
            score = sum(1 for word in question_words if word in sentence_lower)
            if score > 0:
                scored_sentences.append((score, sentence.strip()))
        
        # Sort by relevance score
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        if scored_sentences:
            # Return top 2-3 most relevant sentences
            top_sentences = [sent[1] for sent in scored_sentences[:3]]
            answer = '. '.join(top_sentences)
            
            # Clean up the answer
            if not answer.endswith('.'):
                answer += '.'
            
            return answer
        else:
            return "Based on the retrieved context, I couldn't find specific information that directly answers your question. Please check the source chunks below for relevant details."
    
    def generate_comprehensive_answer(self, question: str, source_chunks: List[Dict]) -> str:
        """Generate a comprehensive answer using multiple chunks and context"""
        if not source_chunks:
            return "No relevant information found in the documents."
        
        # Analyze question to determine if it's asking for a list or detailed explanation
        question_lower = question.lower()
        is_list_question = any(word in question_lower for word in ['list', 'signals', 'commands', 'types', 'what are'])
        
        # Group chunks by section or type if available
        sections = {}
        all_content = []
        
        for chunk in source_chunks[:10]:  # Use top 10 chunks for more comprehensive coverage
            metadata = chunk.get('metadata', {})
            section = metadata.get('section', 'General')
            
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk['text'])
            all_content.append(chunk['text'])
        
        # Extract key information
        question_words = set(word.lower() for word in question.split() if len(word) > 3)
        answer_parts = []
        
        # Look for specific signal names and technical terms
        signal_patterns = []
        technical_terms = []
        
        for content in all_content:
            # Find signal names (usually uppercase or with specific patterns)
            words = content.split()
            for i, word in enumerate(words):
                if any(kw in word.lower() for kw in ['signal', 'cmd', 'rq', 'status']):
                    # Get surrounding context
                    context_start = max(0, i-3)
                    context_end = min(len(words), i+4)
                    signal_context = ' '.join(words[context_start:context_end])
                    if len(signal_context) > 10:
                        signal_patterns.append(signal_context)
                
                # Find technical terms related to headlights
                if any(kw in word.lower() for kw in ['headl', 'beam', 'lamp', 'light']) and len(word) > 4:
                    if i < len(words) - 1:
                        technical_terms.append(f"{word} {words[i+1]}")
                    else:
                        technical_terms.append(word)
        
        # Build comprehensive answer
        if is_list_question and (signal_patterns or technical_terms):
            answer_parts.append("**Headlight Control Signals:**")
            
            # Add signal patterns
            unique_signals = list(set(signal_patterns[:6]))  # Limit to 6 unique patterns
            for signal in unique_signals:
                if any(qw in signal.lower() for qw in question_words):
                    answer_parts.append(f"• {signal}")
            
            # Add technical terms
            unique_terms = list(set(technical_terms[:8]))
            if unique_terms:
                answer_parts.append("\n**Related Components:**")
                for term in unique_terms:
                    answer_parts.append(f"• {term}")
        
        # Add detailed information from different sections
        answer_parts.append("\n**Detailed Information:**")
        
        section_count = 0
        for section, texts in sections.items():
            if section_count >= 3:  # Limit to 3 sections to avoid too long answers
                break
                
            combined_text = ' '.join(texts[:2])  # Use first 2 chunks from each section
            sentences = [s.strip() for s in combined_text.split('.') if len(s.strip()) > 25]
            
            # Find most relevant sentences in this section
            relevant_sentences = []
            for sentence in sentences[:8]:  # Check first 8 sentences
                sentence_words = set(word.lower() for word in sentence.split())
                score = len(question_words.intersection(sentence_words))
                if score >= 1:
                    relevant_sentences.append((score, sentence))
            
            # Sort by relevance and take top sentences
            relevant_sentences.sort(reverse=True, key=lambda x: x[0])
            
            if relevant_sentences:
                if section != 'General':
                    answer_parts.append(f"\n**{section}:**")
                
                # Add top 2 most relevant sentences from this section
                for _, sentence in relevant_sentences[:2]:
                    answer_parts.append(f"• {sentence.strip()}.")
                section_count += 1
        
        if len(answer_parts) <= 2:  # If we didn't find much structured content
            # Fallback to extracting key sentences from all content
            all_text = ' '.join(all_content[:5])
            sentences = [s.strip() for s in all_text.split('.') if len(s.strip()) > 30]
            
            scored_sentences = []
            for sentence in sentences[:15]:
                sentence_words = set(word.lower() for word in sentence.split())
                score = len(question_words.intersection(sentence_words))
                if score > 0:
                    scored_sentences.append((score, sentence))
            
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            
            answer_parts = ["**Key Information:**"]
            for _, sentence in scored_sentences[:4]:
                answer_parts.append(f"• {sentence.strip()}.")
        
        return '\n'.join(answer_parts) if answer_parts else "Unable to extract specific information from the available content."
    
    def query(self, question: str, k: int = 5) -> dict:
        """
        Query the document with a question
        
        Args:
            question: The question to ask
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        # Use enhanced search if available, otherwise fallback to standard search
        if self.enhanced_search:
            try:
                relevant_chunks = self.enhanced_search.search_with_context_expansion(
                    question, k=k, expand_context=True
                )
            except Exception as e:
                logger.warning(f"Enhanced search failed, falling back to standard search: {e}")
                relevant_chunks = self.search_similar(question, k=k)
        else:
            relevant_chunks = self.search_similar(question, k=k)
        
        # Create enhanced context
        context = create_enhanced_qa_context(relevant_chunks, max_length=3000)
        
        # Generate answer
        if self.use_ollama:
            answer = self.generate_answer_ollama(question, context)
        elif hasattr(self, 'use_comprehensive') and self.use_comprehensive:
            # Use comprehensive answer generation for detailed responses
            answer = self.generate_comprehensive_answer(question, relevant_chunks)
        elif self.use_transformers:
            answer = self.generate_answer_transformers(question, context)
            # Check if transformer answer is empty or too short
            main_answer = answer.split('\n')[0].strip() if answer else ""
            if len(main_answer) < 10 or not main_answer or main_answer.isspace():
                # Transformer failed, use comprehensive answer instead
                comprehensive_answer = self.generate_comprehensive_answer(question, relevant_chunks)
                answer = f"**Comprehensive Analysis:**\n{comprehensive_answer}\n\n[Note: Generated using enhanced analysis due to insufficient transformer response]"
            elif len(main_answer) < 50:  # Very short but valid answer
                comprehensive_answer = self.generate_comprehensive_answer(question, relevant_chunks)
                if len(comprehensive_answer) > len(main_answer):
                    answer = f"{answer}\n\n**Detailed Response:**\n{comprehensive_answer}"
        else:
            # Use comprehensive answer for better quality when not using transformers
            answer = self.generate_comprehensive_answer(question, relevant_chunks)
        
        return {
            'answer': answer,
            'source_chunks': relevant_chunks
        }
    
    def process_document(self, file_path: str):
        """Complete pipeline to process a document"""
        # Load document
        documents = self.load_document(file_path)
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Create vector store
        self.create_vectorstore(chunks)
        
        logger.info("Document processing complete!")

def interactive_qa_session(vectorizer: LocalDocumentVectorizer):
    """Run interactive Q&A session in terminal"""
    print("\n" + "="*60)
    print("Local Document Q&A System")
    print("="*60)
    print("Type your questions below. Type 'exit' or 'quit' to end.\n")
    
    while True:
        try:
            # Get user input
            question = input("\n📝 Your Question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            # Process question
            print("\n🔍 Searching for answer...")
            result = vectorizer.query(question)
            
            # Display answer
            print("\n" + "="*60)
            print("📚 ANSWER:")
            print("="*60)
            print(result['answer'])
            
            # Optionally show source chunks
            show_sources = input("\n📄 Show source chunks? (y/n): ").strip().lower()
            if show_sources == 'y':
                print("\n" + "="*60)
                print("📑 SOURCE CHUNKS:")
                print("="*60)
                for i, chunk in enumerate(result['source_chunks'], 1):
                    print(f"\n--- Chunk {i} ---")
                    text = chunk['text']
                    print(text[:500] + "..." if len(text) > 500 else text)
                    if chunk.get('metadata'):
                        print(f"Metadata: {chunk['metadata']}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")

def main():
    """Main function to run the program"""
    parser = argparse.ArgumentParser(description="Local Document Vectorization and Q&A System")
    parser.add_argument("action", choices=["process", "query"], 
                       help="Action to perform: 'process' to vectorize a document, 'query' to ask questions")
    parser.add_argument("--document", "-d", type=str, 
                       help="Path to document file (for 'process' action)")
    parser.add_argument("--db-path", type=str, default="./local_chroma_db",
                       help="Path to vector database directory")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Size of text chunks in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                       help="Overlap between chunks")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence transformer model for embeddings")
    parser.add_argument("--llm", type=str, default="transformers",
                       help="LLM to use: 'transformers' (recommended), 'ollama', or 'simple'")
    parser.add_argument("--no-ocr", action="store_true",
                       help="Disable OCR (faster but won't handle rotated text)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU usage even if GPU is available")
    parser.add_argument("--no-enhanced", action="store_true",
                       help="Disable enhanced technical document processing (use basic chunking)")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Use comprehensive answer generation instead of transformer models for longer responses")
    
    args = parser.parse_args()
    
    # Override LLM model if comprehensive flag is set
    if args.comprehensive:
        args.llm = "comprehensive"
    
    # Initialize vectorizer
    vectorizer = LocalDocumentVectorizer(
        db_path=args.db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        llm_model=args.llm,
        use_gpu=not args.cpu,
        use_ocr=not args.no_ocr,
        use_enhanced=not args.no_enhanced
    )
    
    if args.action == "process":
        if not args.document:
            print("❌ Error: Document path required for 'process' action. Use --document flag")
            sys.exit(1)
        
        print(f"\n📄 Processing document: {args.document}")
        print(f"🔄 OCR enabled: {not args.no_ocr} (handles rotated pages)")
        print(f"⚡ Enhanced processing: {not args.no_enhanced} (technical document optimization)")
        print(f"📊 Chunk size: {args.chunk_size} characters")
        print(f"💾 Database path: {args.db_path}")
        print(f"🧠 Embedding model: {args.embedding_model}")
        print(f"🤖 LLM: {args.llm}\n")
        
        vectorizer.process_document(args.document)
        
        print("\n✅ Document processed successfully!")
        print("You can now run queries with: python script.py query")
        
    elif args.action == "query":
        # Load existing vector store
        try:
            vectorizer.load_vectorstore()
            
            # Start interactive session
            interactive_qa_session(vectorizer)
            
        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
            print("Make sure you've processed a document first with: python script.py process --document <file>")
            sys.exit(1)

if __name__ == "__main__":
    main()