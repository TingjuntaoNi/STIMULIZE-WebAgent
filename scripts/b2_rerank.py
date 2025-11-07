#!/usr/bin/env python3
"""
b2_rerank.py
RAG Reranking Module: Reorder recalled candidate chunks

Features:
1. Cross-Encoder reranking (using bge-reranker and other models)
2. MMR (Maximum Marginal Relevance) diversity optimization
3. Candidate chunk quality assessment and filtering
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys
from scipy.spatial.distance import cosine
import re

# 添加src路径到Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
except ImportError as e:
    print(f"Please install required dependencies: pip install sentence-transformers torch")
    sys.exit(1)


class CrossEncoderReranker:
    """Cross-Encoder重排器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        初始化Cross-Encoder模型
        
        Args:
            model_name: 重排模型名称，可选：
                - BAAI/bge-reranker-large
                - BAAI/bge-reranker-base
                - ms-marco-MiniLM-L-6-v2
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading reranking model: {model_name}")
        print(f"Using device: {self.device}")
        
        try:
            self.model = CrossEncoder(model_name, device=self.device)
            print("Reranking model loaded successfully!")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            print("Trying fallback model...")
            # 备用方案：使用sentence-transformer作为重排器
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.use_fallback = True
        else:
            self.use_fallback = False
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        对候选片段重新排序
        
        Args:
            query: 查询文本
            candidates: 候选片段列表
            top_k: 返回的片段数量
        
        Returns:
            重排后的片段列表
        """
        if not candidates:
            return []
        
        if self.use_fallback:
            return self._fallback_rerank(query, candidates, top_k)
        
        # 准备查询-文档对
        query_doc_pairs = []
        for candidate in candidates:
            query_doc_pairs.append([query, candidate['text']])
        
        # 获取重排分数
        try:
            scores = self.model.predict(query_doc_pairs)
            
            # 添加重排分数到候选片段
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(scores[i])
            
            # 按重排分数排序
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            
        except Exception as e:
            print(f"Error during reranking: {e}")
            print("Using retrieval scores as fallback sorting...")
            reranked = sorted(candidates, key=lambda x: x.get('retrieval_score', 0), reverse=True)
            for candidate in reranked:
                candidate['rerank_score'] = candidate.get('retrieval_score', 0)
        
        return reranked[:top_k]
    
    def _fallback_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Fallback reranking method using sentence transformer similarity"""
        print("Using fallback reranking method...")
        
        # 编码查询
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        
        # Encode candidate documents and calculate similarity
        candidate_texts = [candidate['text'] for candidate in candidates]
        candidate_embeddings = self.model.encode(candidate_texts, normalize_embeddings=True)
        
        # 计算相似度分数
        for i, candidate in enumerate(candidates):
            similarity = 1 - cosine(query_embedding, candidate_embeddings[i])
            candidate['rerank_score'] = float(similarity)
        
        # 排序并返回
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]


class MMRReranker:
    """最大边际相关性(MMR)重排器"""
    
    def __init__(self, lambda_param: float = 0.7, embeddings: np.ndarray = None):
        """
        初始化MMR重排器
        
        Args:
            lambda_param: 相关性vs多样性权衡参数，越大越注重相关性
            embeddings: 预计算的embeddings矩阵
        """
        self.lambda_param = lambda_param
        self.embeddings = embeddings
    
    def rerank(self, query_embedding: np.ndarray, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        使用MMR重新排序
        
        Args:
            query_embedding: 查询的embedding向量
            candidates: 候选片段（需要包含索引信息）
            top_k: 返回的片段数量
        
        Returns:
            MMR重排后的片段列表
        """
        if not candidates or len(candidates) <= top_k:
            return candidates[:top_k]
        
        selected = []
        remaining = candidates.copy()
        
        # 第一步：选择与查询最相关的文档
        if remaining:
            best_candidate = max(remaining, key=lambda x: x.get('rerank_score', x.get('retrieval_score', 0)))
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        
        # 迭代选择剩余文档
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_candidate = None
            
            for candidate in remaining:
                # 计算与查询的相关性
                relevance = candidate.get('rerank_score', candidate.get('retrieval_score', 0))
                
                # 计算与已选择文档的最大相似度（多样性）
                max_similarity = 0
                if self.embeddings is not None and 'chunk_idx' in candidate:
                    candidate_embedding = self.embeddings[candidate['chunk_idx']]
                    for selected_doc in selected:
                        if 'chunk_idx' in selected_doc:
                            selected_embedding = self.embeddings[selected_doc['chunk_idx']]
                            similarity = 1 - cosine(candidate_embedding, selected_embedding)
                            max_similarity = max(max_similarity, similarity)
                
                # MMR分数 = λ * 相关性 - (1-λ) * 最大相似度
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
            
            if best_candidate:
                best_candidate['mmr_score'] = best_score
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected


class QualityFilter:
    """候选片段质量过滤器"""
    
    @staticmethod
    def filter_candidates(candidates: List[Dict], 
                         min_length: int = 50,
                         max_length: int = 2000,
                         min_score: float = 0.1) -> List[Dict]:
        """
        过滤低质量候选片段
        
        Args:
            candidates: 候选片段列表
            min_length: 最小文本长度
            max_length: 最大文本长度
            min_score: 最小检索分数
        
        Returns:
            过滤后的候选片段列表
        """
        filtered = []
        
        for candidate in candidates:
            text = candidate.get('text', '')
            score = candidate.get('rerank_score', candidate.get('retrieval_score', 0))
            
            # 长度过滤
            if len(text) < min_length or len(text) > max_length:
                continue
            
            # 分数过滤
            if score < min_score:
                continue
            
            # 内容质量检查（简单版本）
            if QualityFilter._is_low_quality_content(text):
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    @staticmethod
    def _is_low_quality_content(text: str) -> bool:
        """检查是否为低质量内容"""
        # 检查是否主要是页眉页脚
        if re.match(r'^\s*Stimulize User Manual\s*Page \d+\s*$', text.strip()):
            return True
        
        # 检查是否过于简短且没有实质内容
        words = text.split()
        if len(words) < 5:
            return True
        
        # 检查是否主要是重复内容
        unique_words = set(word.lower() for word in words if word.isalpha())
        if len(unique_words) < len(words) * 0.3:  # 独特词汇少于30%
            return True
        
        return False


def rerank_candidates(query: str, 
                     candidates: List[Dict],
                     top_k: int = 10,
                     use_mmr: bool = True,
                     mmr_lambda: float = 0.7,
                     reranker_model: str = "model/bge-reranker-base") -> List[Dict]:
    """
    对候选片段进行重排
    
    Args:
        query: 查询文本
        candidates: 候选片段列表
        top_k: 最终返回的片段数量
        use_mmr: 是否使用MMR多样性优化
        mmr_lambda: MMR参数
        reranker_model: 重排模型名称
    
    Returns:
        重排后的片段列表
    """
    if not candidates:
        return []
    
    # 第一步：质量过滤
    print(f"Before quality filtering: {len(candidates)} candidates")
    filtered_candidates = QualityFilter.filter_candidates(candidates)
    print(f"After quality filtering: {len(filtered_candidates)} candidates")
    
    if not filtered_candidates:
        return []
    
    # 第二步：Cross-Encoder重排
    print("Performing Cross-Encoder reranking...")
    reranker = CrossEncoderReranker(reranker_model)
    
    # 取更多候选进行重排，然后应用MMR
    rerank_candidates_num = min(len(filtered_candidates), top_k * 3)
    reranked = reranker.rerank(query, filtered_candidates, rerank_candidates_num)
    
    # 第三步：MMR多样性优化（可选）
    if use_mmr and len(reranked) > top_k:
        print("Performing MMR diversity optimization...")
        
        # 为MMR准备embeddings（如果可用）
        try:
            data_dir = Path(__file__).parent / "data" / "processed"
            embeddings = np.load(data_dir / "embeddings" / "embeddings.npy")
            
            # 添加chunk索引
            chunks_file = data_dir / "manual_chunks_embedded.jsonl"
            chunk_id_to_idx = {}
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    chunk = json.loads(line.strip())
                    chunk_id_to_idx[chunk['id']] = idx
            
            # 为候选添加索引
            for candidate in reranked:
                chunk_id = candidate.get('id')
                if chunk_id in chunk_id_to_idx:
                    candidate['chunk_idx'] = chunk_id_to_idx[chunk_id]
            
            mmr_reranker = MMRReranker(mmr_lambda, embeddings)
            # MMR需要query embedding，这里简化处理
            final_results = mmr_reranker.rerank(None, reranked, top_k)
            
        except Exception as e:
            print(f"MMR optimization failed, using rerank results: {e}")
            final_results = reranked[:top_k]
    else:
        final_results = reranked[:top_k]
    
    return final_results


def test_reranking():
    """Test reranking functionality"""
    # Mock some candidate chunks
    mock_candidates = [
        {
            'id': 'test1',
            'text': 'Stimulize allows researchers to choose stimulus types for their experiments. You can select between image and text stimuli.',
            'metadata': {'page': 5},
            'retrieval_score': 0.8
        },
        {
            'id': 'test2', 
            'text': 'The software integrates with Qualtrics platform for online data collection and survey management.',
            'metadata': {'page': 29},
            'retrieval_score': 0.7
        },
        {
            'id': 'test3',
            'text': 'Stimulize User Manual Page 1',
            'metadata': {'page': 1},
            'retrieval_score': 0.6
        },
        {
            'id': 'test4',
            'text': 'Experimental design in Stimulize involves setting up trial parameters, configuring stimulus presentation timing, and defining response collection methods.',
            'metadata': {'page': 18},
            'retrieval_score': 0.75
        }
    ]
    
    query = "How to set up stimulus types in experiments?"
    
    print("=== Reranking Test ===")
    print(f"Query: {query}")
    print(f"Number of candidates: {len(mock_candidates)}")
    
    # Execute reranking
    reranked = rerank_candidates(
        query=query,
        candidates=mock_candidates,
        top_k=3,
        use_mmr=False,  # Simplified test, no MMR
        reranker_model="sentence-transformers/all-MiniLM-L6-v2"  # Use smaller model for testing
    )
    
    print(f"\nReranking Results (Top {len(reranked)}):")
    for i, result in enumerate(reranked, 1):
        print(f"{i}. Rerank score: {result.get('rerank_score', 'N/A'):.4f}")
        print(f"   Original score: {result.get('retrieval_score', 'N/A'):.4f}")
        print(f"   Page: {result['metadata']['page']}")
        print(f"   Content: {result['text']}")
        print()


if __name__ == "__main__":
    test_reranking()