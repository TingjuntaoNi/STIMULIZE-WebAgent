#!/usr/bin/env python3
"""
a2_embedding_chunks.py
使用 Qwen3-Embedding-8B 模型将文档切片编码成向量

功能：
1. 加载 manual_chunks_test.jsonl 文件
2. 使用 Qwen3-Embedding-8B 模型将每个chunk编码成向量
3. 保存向量化结果到文件
"""

import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm

# 添加src路径到Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

# 导入transformers和sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    print(f"请安装必要的依赖：pip install sentence-transformers torch")
    sys.exit(1)


class EmbeddingProcessor:
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化embedding处理器
        
        Args:
            model_path: 模型路径
            device: 设备选择 ("auto", "cpu", "cuda")
        """
        self.model_path = Path(model_path)
        
        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print(f"加载模型: {self.model_path}")
        try:
            self.model = SentenceTransformer(str(self.model_path), device=self.device)
            print(f"模型加载成功!")
            print(f"模型维度: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def encode_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        批量编码chunks
        
        Args:
            chunks: chunk列表
            batch_size: 批处理大小
            
        Returns:
            带有embeddings的chunk列表
        """
        print(f"开始编码 {len(chunks)} 个chunks...")
        
        # 准备文本列表
        texts = []
        for chunk in chunks:
            # 使用document prompt（如果有的话）
            text = chunk["text"]
            texts.append(text)
        
        # 批量编码
        all_embeddings = []
        
        with tqdm(total=len(texts), desc="编码进度") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 编码当前批次
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # 归一化向量
                    )
                    all_embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    print(f"批次 {i//batch_size + 1} 编码失败: {e}")
                    # 如果批量失败，尝试逐个编码
                    for text in batch_texts:
                        try:
                            embedding = self.model.encode(
                                [text],
                                show_progress_bar=False,
                                convert_to_numpy=True,
                                normalize_embeddings=True
                            )[0]
                            all_embeddings.append(embedding)
                        except Exception as e2:
                            print(f"单个文本编码失败: {e2}")
                            # 使用零向量作为fallback
                            zero_embedding = np.zeros(self.model.get_sentence_embedding_dimension())
                            all_embeddings.append(zero_embedding)
                
                pbar.update(len(batch_texts))
        
        # 将embeddings添加到chunks中
        embedded_chunks = []
        for chunk, embedding in zip(chunks, all_embeddings):
            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding.tolist()  # 转换为列表以便JSON序列化
            embedded_chunk["embedding_dim"] = len(embedding)
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks


def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """加载JSONL格式的chunks文件"""
    chunks = []
    print(f"加载chunks文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析错误: {e}")
                continue
    
    print(f"成功加载 {len(chunks)} 个chunks")
    return chunks


def save_embeddings(embedded_chunks: List[Dict[str, Any]], output_path: Path):
    """保存embedding结果"""
    print(f"保存embeddings到: {output_path}")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSONL格式
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in embedded_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"成功保存 {len(embedded_chunks)} 个embedded chunks")


def save_embeddings_numpy(embedded_chunks: List[Dict[str, Any]], output_dir: Path):
    """保存embeddings的numpy格式（用于快速加载）"""
    print(f"保存numpy格式到: {output_dir}")
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取embeddings和metadata
    embeddings = np.array([chunk["embedding"] for chunk in embedded_chunks])
    
    # 保存embeddings矩阵
    np.save(output_dir / "embeddings.npy", embeddings)
    
    # 保存chunk metadata (不包含embedding)
    metadata = []
    for chunk in embedded_chunks:
        meta = {k: v for k, v in chunk.items() if k not in ["embedding"]}
        metadata.append(meta)
    
    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Numpy格式保存完成:")
    print(f"  - embeddings.npy: {embeddings.shape}")
    print(f"  - metadata.json: {len(metadata)} 条记录")


def main():
    # 配置路径
    input_file = project_root / "scripts" / "data" / "processed" / "manual_chunks_test.jsonl"
    model_path = project_root / "model" / "Qwen3-Embedding-8B"
    output_file = project_root / "scripts" / "data" / "processed" / "manual_chunks_embedded.jsonl"
    output_numpy_dir = project_root / "scripts" / "data" / "processed" / "embeddings"
    
    print("=== Qwen3-Embedding-8B 文档向量化 ===")
    print(f"输入文件: {input_file}")
    print(f"模型路径: {model_path}")
    print(f"输出文件: {output_file}")
    print(f"Numpy输出: {output_numpy_dir}")
    
    # 检查输入文件
    if not input_file.exists():
        print(f"错误: 输入文件不存在 {input_file}")
        return
    
    # 检查模型路径
    if not model_path.exists():
        print(f"错误: 模型路径不存在 {model_path}")
        return
    
    # 加载chunks
    try:
        chunks = load_chunks(input_file)
        if not chunks:
            print("错误: 没有找到有效的chunks")
            return
    except Exception as e:
        print(f"加载chunks失败: {e}")
        return
    
    # 显示一些统计信息
    text_lengths = [len(chunk["text"]) for chunk in chunks]
    print(f"\nChunks统计:")
    print(f"  - 总数: {len(chunks)}")
    print(f"  - 平均长度: {np.mean(text_lengths):.1f} 字符")
    print(f"  - 最小长度: {min(text_lengths)} 字符")
    print(f"  - 最大长度: {max(text_lengths)} 字符")
    
    # 初始化embedding处理器
    try:
        processor = EmbeddingProcessor(model_path, device="auto")
    except Exception as e:
        print(f"初始化embedding处理器失败: {e}")
        return
    
    # 开始编码
    start_time = time.time()
    try:
        embedded_chunks = processor.encode_chunks(chunks, batch_size=8)  # 使用较小的batch_size避免内存问题
    except Exception as e:
        print(f"编码过程失败: {e}")
        return
    
    end_time = time.time()
    print(f"\n编码完成! 用时: {end_time - start_time:.2f} 秒")
    
    # 验证结果
    if embedded_chunks:
        first_embedding = embedded_chunks[0]["embedding"]
        print(f"向量维度: {len(first_embedding)}")
        print(f"示例向量前5个值: {first_embedding[:5]}")
    
    # 保存结果
    try:
        # 保存JSONL格式
        save_embeddings(embedded_chunks, output_file)
        
        # 保存numpy格式
        save_embeddings_numpy(embedded_chunks, output_numpy_dir)
        
    except Exception as e:
        print(f"保存结果失败: {e}")
        return
    
    print("\n=== 处理完成 ===")
    print(f"向量化了 {len(embedded_chunks)} 个文档切片")
    print(f"向量维度: {embedded_chunks[0]['embedding_dim'] if embedded_chunks else 'N/A'}")
    print(f"输出文件: {output_file}")
    print(f"Numpy文件: {output_numpy_dir}")


if __name__ == "__main__":
    main()