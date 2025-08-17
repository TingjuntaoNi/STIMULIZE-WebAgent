from typing import List, Dict
import re

def sent_split(text: str) -> List[str]:
    # 轻量句子切分；若用到中文/多语言可替换成 syntok 或 spaCy
    parts = re.split(r"(?<=[。！？!?\.])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(
    text: str,
    size: int = 800,           # 约等于 ~800 字/英文 token 的混合上限
    overlap: int = 150
) -> List[str]:
    sents = sent_split(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) > size and cur:
            chunks.append(" ".join(cur))
            # 重叠：从尾部取若干字符
            tail = (" ".join(cur))[-overlap:]
            cur, cur_len = [tail, s], len(tail) + len(s)
        else:
            cur.append(s); cur_len += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def make_chunk_objs(chunks: List[str], source: str, page: int, ctype: str) -> List[Dict]:
    out = []
    for i, ch in enumerate(chunks, 1):
        out.append({
            "id": f"{source}-p{page}-{ctype}-{i}",
            "text": ch,
            "metadata": {
                "source": source,
                "page": page,
                "type": ctype
            }
        })
    return out
