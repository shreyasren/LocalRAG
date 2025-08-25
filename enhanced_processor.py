#!/usr/bin/env python3
"""
Enhanced Document Processing Module for LocalRAG

This module provides specialized processing capabilities for technical documents,
including smart chunking, context-aware search, and comprehensive answer generation.

Classes:
    TechnicalDocumentProcessor: Handles intelligent document processing for technical content
    EnhancedVectorSearch: Provides context-aware search with relevance scoring
    
Functions:
    create_enhanced_qa_context: Creates optimized context for Q&A from search results

Key Features:
- Technical pattern recognition (signals, commands, etc.)
- Section-aware document chunking
- Context expansion for better Q&A
- Relevance scoring for technical content
- Smart text cleaning and preprocessing

Author: LocalRAG Team
License: MIT
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import re
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class TechnicalDocumentProcessor:
    """Enhanced processor for technical documents like automotive specs"""
    
    def __init__(self):
        self.header_patterns = [
            r'FS-RU5T-\d+\w*-\w+',  # Ford document IDs
            r'All copies of this document are uncontrolled',
            r'Originator: BCM Team',
            r'Date Created: \d{2}-\w{3}-\d{4}',
            r'Date Revised: \d{2}-\w{3}-\d{4}',
            r'Page \d+ of \d+',
            r'Copyright \d{4}',
            r'Confidential and Proprietary',
            r'Version: \d+\.\d+',
            r'FORD MOTOR COMPANY'
        ]
        
        self.technical_indicators = [
            r'R: \d+\.\d+\.\d+',  # Requirements
            r'Table \d+\.\d+',    # Tables
            r'Figure \d+\.\d+',   # Figures
            r'_Command\s*=',      # Signal commands
            r'_Status\s*=',       # Status signals
            r'_Cfg\s*=',          # Configuration
            r'Headlamps?_',       # Headlight related
            r'BCM\s+shall',       # Requirements language
            r'The\s+following\s+requirements',
            r'Feature\s+Behavior',
            r'Signal\s+Translation',
            r'CAN\s+Signal'
        ]
    
    def clean_text(self, text: str) -> str:
        """Remove excessive headers and metadata while preserving technical content"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip pure header/metadata lines
            is_header = any(re.search(pattern, line, re.IGNORECASE) for pattern in self.header_patterns)
            
            # Keep technical content even if it has some header elements
            has_technical_content = any(re.search(pattern, line, re.IGNORECASE) for pattern in self.technical_indicators)
            
            if not is_header or has_technical_content:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_technical_sections(self, text: str) -> List[Dict]:
        """Extract coherent technical sections from the document"""
        sections = []
        
        # Split by major section headers
        section_patterns = [
            r'(R:\s*\d+(?:\.\d+)*(?:\.\d+)*\w*[^\n]*)',     # Requirements like R: 2.3.18.1.7a
            r'(Table\s+\d+(?:\.\d+)*[^:]*:?[^\n]*)',        # Tables
            r'(Figure\s+\d+(?:\.\d+)*[^:]*:?[^\n]*)',       # Figures
            r'(Feature\s+Behavior\s+Detail[^\n]*)',         # Feature sections
            r'(\d+\.\d+(?:\.\d+)*\s+[A-Z][^\n\.]*)',        # Numbered sections
            r'(Requirements?(?:\s+\w+)*[^\n]*)',             # Requirements sections
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                start_pos = match.start()
                
                # Find the end of this section (next section or end of text)
                next_match = None
                for next_pattern in section_patterns:
                    next_matches = list(re.finditer(next_pattern, text[start_pos + len(match.group()):], re.MULTILINE | re.IGNORECASE))
                    if next_matches:
                        if next_match is None or next_matches[0].start() < next_match.start():
                            next_match = next_matches[0]
                
                if next_match:
                    end_pos = start_pos + len(match.group()) + next_match.start()
                else:
                    end_pos = min(start_pos + 2000, len(text))  # Limit section size
                
                section_text = text[start_pos:end_pos]
                cleaned_section = self.clean_text(section_text)
                
                if len(cleaned_section.strip()) > 100:  # Only keep substantial sections
                    sections.append({
                        'title': match.group(1),
                        'content': cleaned_section,
                        'type': self._classify_section(match.group(1)),
                        'start_pos': start_pos
                    })
        
        return sections
    
    def _classify_section(self, title: str) -> str:
        """Classify the type of section based on title"""
        title_lower = title.lower()
        
        if 'table' in title_lower:
            return 'table'
        elif 'figure' in title_lower:
            return 'figure'
        elif 'requirement' in title_lower or title.startswith('R:'):
            return 'requirement'
        elif 'feature' in title_lower or 'behavior' in title_lower:
            return 'feature'
        elif 'signal' in title_lower:
            return 'signal'
        else:
            return 'general'
    
    def create_enhanced_chunks(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Create enhanced chunks that preserve technical context"""
        
        # First, extract technical sections
        sections = self.extract_technical_sections(text)
        
        if not sections:
            # Fallback to cleaned regular chunking
            cleaned_text = self.clean_text(text)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            return splitter.create_documents([cleaned_text])
        
        # Create chunks from sections
        chunks = []
        
        for section in sections:
            section_text = f"{section['title']}\n\n{section['content']}"
            
            if len(section_text) <= chunk_size:
                # Section fits in one chunk
                chunks.append(Document(
                    page_content=section_text,
                    metadata={
                        'section_title': section['title'],
                        'section_type': section['type'],
                        'start_pos': section['start_pos']
                    }
                ))
            else:
                # Split large sections while preserving context
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size - len(section['title']) - 10,  # Reserve space for title
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                sub_chunks = splitter.create_documents([section['content']])
                
                for i, sub_chunk in enumerate(sub_chunks):
                    # Add section title to each sub-chunk
                    enhanced_content = f"{section['title']}\n\n{sub_chunk.page_content}"
                    chunks.append(Document(
                        page_content=enhanced_content,
                        metadata={
                            'section_title': section['title'],
                            'section_type': section['type'],
                            'start_pos': section['start_pos'],
                            'sub_chunk': i + 1,
                            'total_sub_chunks': len(sub_chunks)
                        }
                    ))
        
        return chunks


class EnhancedVectorSearch:
    """Enhanced vector search with technical document awareness"""
    
    def __init__(self, collection, embedding_model=None):
        self.collection = collection
        self.embedding_model = embedding_model
    
    def search_with_context_expansion(self, query: str, k: int = 5, expand_context: bool = True) -> List[Dict]:
        """Enhanced search that expands context for better technical answers"""
        
        # Initial search - use collection's query method directly
        if self.embedding_model:
            query_embedding = self.embedding_model.encode([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k * 2  # Get more results initially
            )
        else:
            # Fallback to text-based search if no embedding model available
            results = self.collection.query(
                query_texts=[query],
                n_results=k * 2
            )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            result = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else 0
            }
            formatted_results.append(result)
        
        if not expand_context:
            return formatted_results[:k]
        
        # Group results by section and expand context
        enhanced_results = self._expand_section_context(formatted_results)
        
        # Re-rank by relevance and technical content quality
        scored_results = self._score_technical_relevance(enhanced_results, query)
        
        return scored_results[:k]
    
    def _expand_section_context(self, results: List[Dict]) -> List[Dict]:
        """Expand context by including related chunks from the same section"""
        enhanced_results = []
        processed_sections = set()
        
        for result in results:
            metadata = result.get('metadata', {})
            section_title = metadata.get('section_title')
            
            if section_title and section_title not in processed_sections:
                # Find all chunks from the same section
                section_chunks = [r for r in results if r.get('metadata', {}).get('section_title') == section_title]
                
                if len(section_chunks) > 1:
                    # Combine chunks from the same section
                    combined_text = '\n\n'.join([chunk['text'] for chunk in section_chunks])
                    enhanced_result = {
                        'text': combined_text,
                        'metadata': metadata,
                        'distance': min([chunk['distance'] for chunk in section_chunks]),
                        'is_expanded': True,
                        'chunk_count': len(section_chunks)
                    }
                    enhanced_results.append(enhanced_result)
                    processed_sections.add(section_title)
                else:
                    enhanced_results.append(result)
            else:
                if not section_title or section_title not in processed_sections:
                    enhanced_results.append(result)
        
        return enhanced_results
    
    def _score_technical_relevance(self, results: List[Dict], query: str) -> List[Dict]:
        """Score results based on technical content quality"""
        
        query_lower = query.lower()
        technical_terms = ['signal', 'command', 'headlight', 'lamp', 'bcm', 'control', 'status']
        
        for result in results:
            text_lower = result['text'].lower()
            
            # Base score from vector similarity (lower distance = higher relevance)
            base_score = 1.0 - result['distance']
            
            # Boost for technical content
            technical_score = 0
            for term in technical_terms:
                if term in query_lower and term in text_lower:
                    technical_score += 0.1
            
            # Boost for requirements and structured content
            structure_score = 0
            if re.search(r'R: \d+\.\d+', result['text']):
                structure_score += 0.2
            if 'Table' in result['text'] or 'Figure' in result['text']:
                structure_score += 0.15
            
            # Penalize header-heavy content
            header_penalty = 0
            header_ratio = len(re.findall(r'(Page \d+|Copyright|Confidential)', result['text'])) / max(len(result['text'].split()), 1)
            if header_ratio > 0.1:
                header_penalty = header_ratio * 0.5
            
            final_score = base_score + technical_score + structure_score - header_penalty
            result['relevance_score'] = final_score
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)


def create_enhanced_qa_context(chunks: List[Dict], max_length: int = 3000) -> str:
    """Create enhanced context for Q&A that prioritizes technical content"""
    
    if not chunks:
        return ""
    
    # Sort chunks by relevance score if available
    sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', x.get('distance', 1)), reverse=True)
    
    context_parts = []
    current_length = 0
    
    for chunk in sorted_chunks:
        chunk_text = chunk['text']
        
        # Add section header if available
        metadata = chunk.get('metadata', {})
        section_title = metadata.get('section_title')
        if section_title and section_title not in chunk_text:
            chunk_text = f"[{section_title}]\n{chunk_text}"
        
        # Check if adding this chunk would exceed limit
        if current_length + len(chunk_text) > max_length and context_parts:
            break
        
        context_parts.append(chunk_text)
        current_length += len(chunk_text) + 2  # +2 for separator
    
    return '\n\n'.join(context_parts)
