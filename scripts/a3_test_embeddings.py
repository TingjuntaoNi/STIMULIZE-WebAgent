#!/usr/bin/env python3
"""
a3_test_embeddings.py
测试生成的embeddings质量
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine

def load_embeddings():
    """加载embeddings数据"""
    # 加载JSONL格式的数据
    jsonl_path = Path(__file__).parent / "data" / "processed" / "manual_chunks_embedded.jsonl"
    chunks = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line.strip())
            chunks.append(chunk)
    
    # 加载numpy格式的数据
    numpy_dir = Path(__file__).parent / "data" / "processed" / "embeddings"
    embeddings_matrix = np.load(numpy_dir / "embeddings.npy")
    
    with open(numpy_dir / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return chunks, embeddings_matrix, metadata

def calculate_similarity(embedding1, embedding2):
    """计算两个向量的余弦相似度"""
    return 1 - cosine(embedding1, embedding2)

def find_similar_chunks(query_text, chunks, embeddings, top_k=5):
    """找到与查询文本最相似的chunks"""
    # 找到包含查询文本的chunk
    query_chunks = []
    for i, chunk in enumerate(chunks):
        if query_text.lower() in chunk['text'].lower():
            query_chunks.append((i, chunk))
    
    if not query_chunks:
        print(f"没有找到包含'{query_text}'的chunk")
        return []
    
    # 使用第一个匹配的chunk作为查询
    query_idx, query_chunk = query_chunks[0]
    query_embedding = embeddings[query_idx]
    
    print(f"查询chunk (ID: {query_chunk['id']}):")
    print(f"页面: {query_chunk['metadata']['page']}")
    print(f"内容: {query_chunk['text'][:200]}...")
    print()
    
    # 计算与所有其他chunks的相似度
    similarities = []
    for i, chunk in enumerate(chunks):
        if i != query_idx:  # 跳过自身
            similarity = calculate_similarity(query_embedding, embeddings[i])
            similarities.append((i, chunk, similarity))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    return similarities[:top_k]

def test_embeddings():
    """测试embeddings质量"""
    print("=== 测试STIMULIZE用户手册Embeddings ===")
    
    # 加载数据
    chunks, embeddings_matrix, metadata = load_embeddings()
    
    print(f"加载了 {len(chunks)} 个chunks")
    print(f"Embeddings矩阵形状: {embeddings_matrix.shape}")
    print()
    
    # 测试查询1: 寻找与"stimulus"相关的内容
    print("=== 测试1: 查找与'stimulus'相关的内容 ===")
    similar_chunks = find_similar_chunks("stimulus", chunks, embeddings_matrix, top_k=3)
    
    for i, (idx, chunk, similarity) in enumerate(similar_chunks, 1):
        print(f"{i}. 相似度: {similarity:.4f}")
        print(f"   页面: {chunk['metadata']['page']}")
        print(f"   内容: {chunk['text'][:150]}...")
        print()
    
    # 测试查询2: 寻找与"Qualtrics"相关的内容
    print("=== 测试2: 查找与'Qualtrics'相关的内容 ===")
    similar_chunks = find_similar_chunks("Qualtrics", chunks, embeddings_matrix, top_k=3)
    
    for i, (idx, chunk, similarity) in enumerate(similar_chunks, 1):
        print(f"{i}. 相似度: {similarity:.4f}")
        print(f"   页面: {chunk['metadata']['page']}")
        print(f"   内容: {chunk['text'][:150]}...")
        print()
    
    # 测试查询3: 寻找与"experiment"相关的内容
    print("=== 测试3: 查找与'experiment'相关的内容 ===")
    similar_chunks = find_similar_chunks("experiment", chunks, embeddings_matrix, top_k=3)
    
    for i, (idx, chunk, similarity) in enumerate(similar_chunks, 1):
        print(f"{i}. 相似度: {similarity:.4f}")
        print(f"   页面: {chunk['metadata']['page']}")
        print(f"   内容: {chunk['text'][:150]}...")
        print()
    
    # 计算一些统计信息
    print("=== Embeddings统计信息 ===")
    
    # 计算所有向量的平均相似度
    similarities = []
    for i in range(len(embeddings_matrix)):
        for j in range(i+1, len(embeddings_matrix)):
            sim = calculate_similarity(embeddings_matrix[i], embeddings_matrix[j])
            similarities.append(sim)
    
    similarities = np.array(similarities)
    print(f"平均相似度: {similarities.mean():.4f}")
    print(f"相似度标准差: {similarities.std():.4f}")
    print(f"最高相似度: {similarities.max():.4f}")
    print(f"最低相似度: {similarities.min():.4f}")
    
    # 检查向量是否已归一化
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    print(f"向量模长 - 平均值: {norms.mean():.6f}, 标准差: {norms.std():.6f}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_embeddings()