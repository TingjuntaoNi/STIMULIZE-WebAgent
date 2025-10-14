#!/usr/bin/env python3
"""
b3_generate.py
RAG Generation Module: Generate answers based on retrieved document chunks

Features:
1. Context assembly and source information addition
2. Strict template prompts to ensure material-based answers
3. Length control and chunking processing
4. Anti-hallucination mechanisms and answerability determination
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from datetime import datetime

# 添加相关模块
from b1_recall import retrieve_for_query
from b2_rerank import rerank_candidates

# 如果有OpenAI或其他LLM API
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ContextAssembler:
    """Context assembler"""
    
    @staticmethod
    def assemble_context(chunks: List[Dict], 
                        max_length: int = 8000,
                        include_metadata: bool = True) -> Tuple[str, List[str]]:
        """
        Assemble retrieved document chunks into context
        
        Args:
            chunks: List of document chunks
            max_length: Maximum context length
            include_metadata: Whether to include metadata information
        
        Returns:
            (assembled_context, list_of_used_chunk_ids)
        """
        context_parts = []
        used_chunk_ids = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('id', f'chunk_{i}')
            text = chunk.get('text', '')
            page = chunk.get('metadata', {}).get('page', 'unknown')
            
            # Build text for single chunk
            if include_metadata:
                chunk_text = f"[Chunk {i+1}, ID:{chunk_id}, Page:{page}]\n{text}\n"
            else:
                chunk_text = f"[Chunk {i+1}]\n{text}\n"
            
            # Check length limitation
            if current_length + len(chunk_text) > max_length:
                if current_length == 0:  # If first chunk is too long, truncate it
                    truncated_text = text[:max_length-200] + "...\n"
                    if include_metadata:
                        chunk_text = f"[Chunk {i+1}, ID:{chunk_id}, Page:{page}]\n{truncated_text}"
                    else:
                        chunk_text = f"[Chunk {i+1}]\n{truncated_text}"
                    context_parts.append(chunk_text)
                    used_chunk_ids.append(chunk_id)
                break
            
            context_parts.append(chunk_text)
            used_chunk_ids.append(chunk_id)
            current_length += len(chunk_text)
        
        context = "\n".join(context_parts)
        return context, used_chunk_ids


class PromptTemplate:
    """Prompt template manager"""
    
    @staticmethod
    def get_qa_template() -> str:
        """Get Q&A template"""
        return """You are an intelligent assistant for the STIMULIZE user manual. Please answer user questions strictly based on the provided manual content.

【IMPORTANT CONSTRAINTS】
1. Only answer based on the provided manual chunks, do not fabricate information
2. If no relevant information is found in the manual, clearly state "No relevant information found in the manual"
3. Must cite specific chunk IDs and page numbers when answering
4. Keep answers concise and accurate, highlighting key information

【MANUAL CONTENT】
{context}

【USER QUESTION】
{question}

【ANSWER REQUIREMENTS】
- Answer based on the above manual content
- Cite specific chunk IDs (e.g., "According to Chunk 1 (ID:xxx, Page X)")
- If information is insufficient, explain what information is missing and suggest checking relevant pages
- Answer format: Direct answer + Information sources

请回答："""

    @staticmethod
    def get_answerability_template() -> str:
        """Get answerability determination template"""
        return """Please determine whether the user's question can be answered based on the following manual content.

【MANUAL CONTENT】
{context}

【USER QUESTION】
{question}

【DETERMINATION CRITERIA】
- Can answer: Manual contains direct or indirect relevant information
- Cannot answer: Manual contains no relevant information at all

Please only answer "Can answer" or "Cannot answer" with a brief explanation."""

    @staticmethod
    def get_source_summary_template() -> str:
        """Get source summary template"""
        return """Based on the following manual chunk information, provide lookup suggestions for the user:

【RELEVANT CHUNKS】
{chunks_info}

【USER QUESTION】
{question}

Please provide:
1. Most relevant page suggestions
2. Possible answer-containing section hints
3. Recommended reading order

Format requirements: Concise and clear, emphasizing page numbers and section names."""


class LLMGenerator:
    """LLM generator base class"""
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate answer"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if available"""
        raise NotImplementedError


class OpenAIGenerator(LLMGenerator):
    """OpenAI API generator"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        if api_key:
            openai.api_key = api_key
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate answer using OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Generation failed: {e}"
    
    def is_available(self) -> bool:
        return HAS_OPENAI and self.api_key is not None


class MockGenerator(LLMGenerator):
    """Mock generator (for testing)"""
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Mock generate answer"""
        if "answerability" in prompt.lower() or "can answer" in prompt.lower():
            return "Can answer. Manual contains relevant stimulus type setting information."
        
        # Simple rule-based mock answer generation
        return """According to the manual content, the steps to set up stimulus types are as follows:

1. In the Stimulize interface, select the "Choose Stimulus Types" option
2. You can choose image stimuli or text stimuli
3. For image stimuli, you need to upload them to an online server (such as Qualtrics library) to get the URL
4. For text stimuli, you can directly input them in the designated area

Information source: Based on Chunk 1 (ID:test1, Page 5) and related manual content.

For more detailed information, please refer to the "BASIC FUNCTIONS" section on page 5 of the user manual."""
    
    def is_available(self) -> bool:
        return True


class RAGGenerator:
    """RAG generator main class"""
    
    def __init__(self, llm_generator: LLMGenerator = None):
        self.llm_generator = llm_generator or MockGenerator()
        self.context_assembler = ContextAssembler()
        self.prompt_template = PromptTemplate()
    
    def generate_answer(self, 
                       question: str,
                       retrieval_top_k: int = 50,
                       rerank_top_k: int = 8,
                       max_context_length: int = 6000,
                       check_answerability: bool = True) -> Dict[str, Any]:
        """
        Generate Q&A answer
        
        Args:
            question: User question
            retrieval_top_k: Number of recalled candidates
            rerank_top_k: Number to keep after reranking
            max_context_length: Maximum context length
            check_answerability: Whether to check answerability
        
        Returns:
            Dictionary containing answer, sources and other information
        """
        result = {
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'answer': '',
            'sources': [],
            'context_chunks': [],
            'answerability': 'unknown',
            'processing_info': {}
        }
        
        try:
            # Step 1: Retrieval
            print(f"Starting retrieval of relevant document chunks...")
            candidates = retrieve_for_query(question, top_k=retrieval_top_k)
            result['processing_info']['retrieved_count'] = len(candidates)
            print(f"Retrieved {len(candidates)} candidate chunks")
            
            if not candidates:
                result['answer'] = "Sorry, no content related to your question was found in the user manual. Please try using different keywords or check the complete user manual."
                return result
            
            # Step 2: Reranking
            print(f"Reordering candidate chunks...")
            reranked_chunks = rerank_candidates(
                query=question,
                candidates=candidates,
                top_k=rerank_top_k,
                use_mmr=True
            )
            result['processing_info']['reranked_count'] = len(reranked_chunks)
            print(f"Kept {len(reranked_chunks)} chunks after reranking")
            
            if not reranked_chunks:
                result['answer'] = "After quality filtering, no sufficiently relevant content was found. Please check your question phrasing or refer to the user manual table of contents."
                return result
            
            # Step 3: Context assembly
            context, used_chunk_ids = self.context_assembler.assemble_context(
                reranked_chunks, 
                max_length=max_context_length
            )
            result['context_chunks'] = used_chunk_ids
            
            # Step 4: Answerability determination (optional)
            if check_answerability:
                print("Checking question answerability...")
                answerability_prompt = self.prompt_template.get_answerability_template().format(
                    context=context,
                    question=question
                )
                answerability = self.llm_generator.generate(answerability_prompt, max_tokens=200)
                result['answerability'] = answerability
                
                if "cannot answer" in answerability.lower():
                    # Provide lookup suggestions
                    chunks_info = self._format_chunks_for_suggestion(reranked_chunks)
                    suggestion_prompt = self.prompt_template.get_source_summary_template().format(
                        chunks_info=chunks_info,
                        question=question
                    )
                    suggestion = self.llm_generator.generate(suggestion_prompt, max_tokens=300)
                    result['answer'] = f"Based on the user manual content, your question cannot be directly answered.\n\nLookup suggestions:\n{suggestion}"
                    result['sources'] = self._extract_sources(reranked_chunks)
                    return result
            
            # Step 5: Generate answer
            print("Generating final answer...")
            qa_prompt = self.prompt_template.get_qa_template().format(
                context=context,
                question=question
            )
            
            answer = self.llm_generator.generate(qa_prompt, max_tokens=1000)
            result['answer'] = answer
            result['sources'] = self._extract_sources(reranked_chunks)
            
            print("Answer generation completed!")
            
        except Exception as e:
            result['answer'] = f"Error occurred while generating answer: {str(e)}"
            result['processing_info']['error'] = str(e)
        
        return result
    
    def _format_chunks_for_suggestion(self, chunks: List[Dict]) -> str:
        """Format chunk information for suggestion generation"""
        chunks_info = []
        for i, chunk in enumerate(chunks[:5], 1):  # Only take first 5
            page = chunk.get('metadata', {}).get('page', 'unknown')
            text_preview = chunk.get('text', '')[:100] + '...'
            chunks_info.append(f"Chunk {i} (Page {page}): {text_preview}")
        
        return "\n".join(chunks_info)
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information"""
        sources = []
        for chunk in chunks:
            source = {
                'chunk_id': chunk.get('id', ''),
                'page': chunk.get('metadata', {}).get('page', 'unknown'),
                'score': chunk.get('rerank_score', chunk.get('retrieval_score', 0)),
                'text_preview': chunk.get('text', '')[:200] + '...'
            }
            sources.append(source)
        return sources


def test_generation():
    """Test generation functionality"""
    print("=== RAG Generation Test ===")
    
    # Initialize generator
    generator = RAGGenerator()
    
    # Test questions
    test_questions = [
        "How to set up stimulus types in Stimulize?",
        "How does Stimulize integrate with Qualtrics?",
        "What is the basic workflow for experimental design?",
        "How to collect experimental data?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"{'='*50}")
        
        # Generate answer
        result = generator.generate_answer(
            question=question,
            retrieval_top_k=20,
            rerank_top_k=5,
            check_answerability=True
        )
        
        print(f"Processing info: {result['processing_info']}")
        print(f"Answerability: {result['answerability']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nUsed document chunks: {len(result['sources'])}")
        
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  {i}. Page {source['page']}, Score {source['score']:.3f}")
        
        print("\n" + "-"*50)


def answer_question(question: str) -> Dict[str, Any]:
    """
    Convenience function to answer a single question
    
    Args:
        question: User question
    
    Returns:
        Answer result dictionary
    """
    generator = RAGGenerator()
    return generator.generate_answer(question)


if __name__ == "__main__":
    test_generation()