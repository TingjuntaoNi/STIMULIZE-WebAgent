#!/usr/bin/env python3
"""
b1_recall.py
RAG召回模块：从向量库中检索相关文档片段

功能：
1. 稠密向量检索（基于embedding相似度）
2. BM25关键词检索（可选）
3. 混合检索结果合并去重
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.distance import cosine
import sys
import re
from collections import Counter, defaultdict
import math

# 添加src路径到Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

# 导入embedding相关模块
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Please install required dependencies: pip install sentence-transformers")
    sys.exit(1)


class BM25Retriever:
    """BM25关键词检索器"""
    
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_len = [len(doc.split()) for doc in documents]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # 构建倒排索引
        self.doc_freqs = []
        self.idf = {}
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Text tokenization for English documents"""
        # Simple English tokenization
        # Extract words and convert to lowercase
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Remove very short words (less than 2 characters)
        words = [word for word in words if len(word) >= 2]
        return words
    
    def _build_index(self):
        """构建BM25索引"""
        # 计算每个文档的词频
        for doc in self.documents:
            tokens = self._tokenize(doc)
            doc_freq = Counter(tokens)
            self.doc_freqs.append(doc_freq)
            
            # 统计包含每个词的文档数
            for token in doc_freq:
                self.idf[token] = self.idf.get(token, 0) + 1
        
        # 计算IDF
        num_docs = len(self.documents)
        for token in self.idf:
            self.idf[token] = math.log((num_docs - self.idf[token] + 0.5) / (self.idf[token] + 0.5))
    
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """BM25搜索"""
        query_tokens = self._tokenize(query)
        query_freq = Counter(query_tokens)
        
        scores = []
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            doc_len = self.doc_len[i]
            
            for token, freq in query_freq.items():
                if token in doc_freq:
                    tf = doc_freq[token]
                    idf = self.idf.get(token, 0)
                    
                    # BM25公式
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * (numerator / denominator)
            
            scores.append((i, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DenseRetriever:
    """稠密向量检索器"""
    
    def __init__(self, embeddings: np.ndarray, model_path: str = None):
        self.embeddings = embeddings
        self.model = None
        if model_path:
            self.model = SentenceTransformer(model_path)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query text to embedding vector"""
        if self.model:
            return self.model.encode([query], normalize_embeddings=True)[0]
        else:
            raise ValueError("Model path required for query encoding")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> List[Tuple[int, float]]:
        """向量相似度搜索"""
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = 1 - cosine(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class HybridRetriever:
    """混合检索器：结合稠密向量和BM25"""
    
    def __init__(self, chunks: List[Dict], embeddings: np.ndarray, model_path: str):
        self.chunks = chunks
        self.embeddings = embeddings
        
        # 初始化稠密检索器
        self.dense_retriever = DenseRetriever(embeddings, model_path)
        
        # 初始化BM25检索器
        documents = [chunk['text'] for chunk in chunks]
        self.bm25_retriever = BM25Retriever(documents)
    
    def search(self, 
               query: str, 
               top_k: int = 50,
               dense_weight: float = 0.7,
               bm25_weight: float = 0.3,
               use_bm25: bool = True) -> List[Dict]:
        """
        混合搜索
        
        Args:
            query: 查询文本
            top_k: 返回的候选数量
            dense_weight: 稠密向量权重
            bm25_weight: BM25权重
            use_bm25: 是否使用BM25
        """
        # 稠密向量检索
        query_embedding = self.dense_retriever.encode_query(query)
        dense_results = self.dense_retriever.search(query_embedding, top_k * 2)
        
        # 初始化结果字典
        candidate_scores = defaultdict(float)
        
        # 添加稠密向量结果
        max_dense_score = dense_results[0][1] if dense_results else 1.0
        for idx, score in dense_results:
            normalized_score = score / max_dense_score
            candidate_scores[idx] += dense_weight * normalized_score
        
        # BM25检索（如果启用）
        if use_bm25:
            bm25_results = self.bm25_retriever.search(query, top_k * 2)
            
            # 添加BM25结果
            max_bm25_score = bm25_results[0][1] if bm25_results else 1.0
            if max_bm25_score > 0:
                for idx, score in bm25_results:
                    normalized_score = score / max_bm25_score
                    candidate_scores[idx] += bm25_weight * normalized_score
        
        # 排序并返回Top-K
        sorted_candidates = sorted(candidate_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:top_k]
        
        # 构建结果
        results = []
        for idx, score in sorted_candidates:
            chunk = self.chunks[idx].copy()
            chunk['retrieval_score'] = score
            chunk['dense_similarity'] = None
            chunk['bm25_score'] = None
            
            # 添加详细分数信息
            for dense_idx, dense_score in dense_results:
                if dense_idx == idx:
                    chunk['dense_similarity'] = dense_score
                    break
            
            if use_bm25:
                for bm25_idx, bm25_score in bm25_results:
                    if bm25_idx == idx:
                        chunk['bm25_score'] = bm25_score
                        break
            
            results.append(chunk)
        
        return results


def load_data():
    """加载embeddings和chunks数据"""
    data_dir = Path(__file__).parent / "data" / "processed"
    
    # 加载chunks
    chunks = []
    with open(data_dir / "manual_chunks_embedded.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line.strip())
            # 移除embedding字段以节省内存（我们用numpy文件）
            if 'embedding' in chunk:
                del chunk['embedding']
            chunks.append(chunk)
    
    # 加载embeddings矩阵
    embeddings = np.load(data_dir / "embeddings" / "embeddings.npy")
    
    return chunks, embeddings


def test_retrieval():
    """Test retrieval functionality"""
    print("=== Loading Data ===")
    chunks, embeddings = load_data()
    model_path = project_root / "model" / "Qwen3-Embedding-8B"
    
    print(f"Loaded {len(chunks)} chunks")
    print(f"Embeddings matrix shape: {embeddings.shape}")
    
    # 初始化混合检索器
    print("\n=== Initializing Retriever ===")
    retriever = HybridRetriever(chunks, embeddings, str(model_path))
    
    # 测试查询
    test_queries = [
        "How to set up stimulus types?",
        "Qualtrics integration",
        "experimental design workflow",
        "data collection and recording"
    ]
    
    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        
        # 混合检索
        results = retriever.search(query, top_k=10, use_bm25=True)
        
        print(f"Found {len(results)} candidate chunks:")
        for i, result in enumerate(results[:5], 1):  # 只显示前5个
            print(f"{i}. Combined Score: {result['retrieval_score']:.4f}")
            print(f"   Page: {result['metadata']['page']}")
            if result['dense_similarity']:
                print(f"   Dense Similarity: {result['dense_similarity']:.4f}")
            if result['bm25_score']:
                print(f"   BM25 Score: {result['bm25_score']:.4f}")
            print(f"   Content: {result['text'][:100]}...")
            print()


def retrieve_for_query(query: str, 
                      top_k: int = 50,
                      dense_weight: float = 0.7,
                      use_bm25: bool = True) -> List[Dict]:
    """
    为给定查询执行检索
    
    Args:
        query: 查询文本
        top_k: 返回的候选数量
        dense_weight: 稠密向量权重
        use_bm25: 是否使用BM25
    
    Returns:
        检索结果列表
    """
    # 加载数据
    chunks, embeddings = load_data()
    model_path = project_root / "model" / "Qwen3-Embedding-8B"
    
    # 初始化检索器
    retriever = HybridRetriever(chunks, embeddings, str(model_path))
    
    # 执行检索
    bm25_weight = 1.0 - dense_weight if use_bm25 else 0.0
    results = retriever.search(
        query, 
        top_k=top_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        use_bm25=use_bm25
    )
    
    return results


if __name__ == "__main__":
    test_retrieval()