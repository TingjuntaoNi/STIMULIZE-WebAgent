# build_index_qwen3_emb.py
import os, json, ujson, math
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel

# ========= 配置 =========
MODEL_NAME = "models/Qwen3-Embedding-8B"
DATA_JSONL = "data/output_chunks.jsonl"  # 切片输出
INDEX_DIR = "vector_index"
BATCH_SIZE = 16
MAX_LEN = 1024  # 保险起见限制一下长度
USE_4BIT = False  # 显存吃紧可开
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

# ========= 加载模型 =========
print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

load_kwargs = {
    "trust_remote_code": True
}
if DEVICE == "cuda":
    if USE_4BIT:
        load_kwargs.update(dict(
            device_map="auto",
            load_in_4bit=True
        ))
    else:
        load_kwargs.update(dict(
            device_map="auto",
            torch_dtype=torch.float16
        ))
else:
    load_kwargs.update(dict(
        torch_dtype=torch.float32
    ))

model = AutoModel.from_pretrained(MODEL_NAME, **load_kwargs)
model.eval()

# 一些 Qwen Embedding 模型会提供 pooled 输出；如果没有，就自己做 mean-pooling
@torch.no_grad()
def encode_texts(texts: List[str]) -> np.ndarray:
    batch = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**batch)
    # 优先尝试 'pooler_output'；若无，则对 last_hidden_state 做 mask mean
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        emb = out.pooler_output
    else:
        last = out.last_hidden_state  # [B, T, H]
        mask = batch["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    # L2 归一化，便于用内积做余弦
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().float().numpy()

# ========= 读取切片 =========
print("Loading chunks...")
metas = []   # 存每个 chunk 的元数据（id、页码、类型等）
texts = []   # 存待编码文本
with open(DATA_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = ujson.loads(line)
        # 你之前的字段名可能是 "text" / "content"；按你的产物改一下
        content = obj.get("text") or obj.get("content") or ""
        if not content.strip():
            continue
        # 最好保留 id/source/page/ctype 等，方便回溯
        metas.append({
            "id": obj.get("id"),
            "source": obj.get("source"),
            "page": obj.get("page"),
            "ctype": obj.get("ctype"),
            "text": content
        })
        texts.append(content)

print(f"Total chunks: {len(texts)}")

# ========= 批量编码 =========
all_vecs = []
for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    vecs = encode_texts(batch_texts)  # [B, D]
    all_vecs.append(vecs)
    if (i // BATCH_SIZE) % 10 == 0:
        print(f"Encoded {i+len(batch_texts)}/{len(texts)}")

emb_matrix = np.vstack(all_vecs).astype("float32")  # [N, D]
dim = emb_matrix.shape[1]
print("Embedding shape:", emb_matrix.shape)

# ========= 建立 FAISS 索引 =========
# 向量已归一化 → 用内积(IndexFlatIP)，等价于 cosine 相似度
index = faiss.IndexFlatIP(dim)
index.add(emb_matrix)
faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

# 保存元数据（与向量行号一一对应）
with open(os.path.join(INDEX_DIR, "meta.jsonl"), "w", encoding="utf-8") as f:
    for m in metas:
        f.write(ujson.dumps(m, ensure_ascii=False) + "\n")

print("✅ Index built and saved.")

# ========= 查询示例 =========
def search(query: str, topk=5):
    qv = encode_texts([query])  # [1, D]
    sims, ids = index.search(qv.astype("float32"), topk)  # 内积分数
    results = []
    # 读取元数据（简单起见，每次都读；实际可加载到内存）
    meta_list = [ujson.loads(x) for x in open(os.path.join(INDEX_DIR,"meta.jsonl"),"r",encoding="utf-8").read().splitlines()]
    for score, idx in zip(sims[0], ids[0]):
        m = meta_list[idx]
        results.append({"score": float(score), **m})
    return results

if __name__ == "__main__":
    demo = search("How do I adjust the size of image?", topk=5)
    for r in demo:
        print(f"[{r['score']:.3f}] p{r.get('page')} {r.get('source')}  ->  {r['text'][:80]}...")
