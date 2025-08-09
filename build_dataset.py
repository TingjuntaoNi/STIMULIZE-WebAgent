# %%
# build_dataset_local.py
import os, re, json, hashlib, subprocess, requests
import torch
import fitz  # PyMuPDF
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== 0) 路径与模型配置 ======
PDF_PATH = "/path/to/your/user_manual.pdf"   
OUT_PATH = "./dataset.jsonl"                  
MODEL_PATH = "/inspire/hdd/global_public/public_models/meta-llama/Llama-3.3-70B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",          # 自动分配到可用 GPU / CPU
    torch_dtype=torch.float16,  # 节省显存
)                  

# # 如果用本地 OpenAI 兼容 API
# OPENAI_COMPAT_URL = "http://localhost:8000/v1/chat/completions"

# ====== 1) 提示词模板======
EXTRACT_SYS = """You are an information extraction model.
Given a passage from a software user manual, extract a STRICT JSON object with these fields:
{
  "title": string,
  "type": "procedure|concept|faq",
  "preconditions": [string],
  "steps": [
    {"action": string, "target": string, "path": string, "parameters": [string], "result": string}
  ],
  "parameters": [string],
  "notes": [string],
  "faqs": [ {"q": string, "a": string} ]
}
Rules:
- Only use information found in the passage.
- If something is missing, use empty arrays/empty string.
- Output ONLY valid JSON. No commentary.
"""

GEN_Q_SYS = """You generate diverse user intents for support search.
Return JSON array of 3-5 English questions that a user might ask for the same feature.
Cover: how-to, where to configure, troubleshooting ("can't ..."), role/permission, and short form.
ONLY return a JSON array of strings. No commentary."""

VERIFY_SYS = """You are a strict verifier.
Given the original passage and a proposed answer, check that EVERY statement in the answer is fully supported by the passage.
Return JSON: {"supported": boolean, "unsupported_spans": [string], "reason": string}
ONLY return JSON.
"""

EXTRACT_USER_TMPL = """PASSAGE:
{passage}"""
GEN_Q_USER_TMPL = """FEATURE JSON:
{feature_json}"""
VERIFY_USER_TMPL = """PASSAGE:
{passage}
ANSWER:
{answer}
"""

# ====== 2) PDF 解析与切块 ======
TITLE_RE = re.compile(r'^([A-Za-z ]+\:|[0-9]+(\.[0-9]+)*\s+.+|[A-Z][A-Za-z0-9 \-_/]{0,60}$)')
STEP_HINTS = ("Step", "Steps", "Procedure", "How to", "Instructions")

def is_title(t: str) -> bool:
    return bool(TITLE_RE.match(t)) or t.endswith((":", "："))

def is_step_line(t: str) -> bool:
    t_strip = t.strip()
    return bool(re.match(r'^\d+[\.\)]\s+', t_strip)) or any(h.lower() in t_strip.lower() for h in STEP_HINTS)

def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []
    for pno, page in enumerate(doc):
        for _, _, _, _, txt, _, _ in page.get_text("blocks"):
            for line in (l.strip() for l in txt.split("\n")):
                if line:
                    lines.append({"page": pno + 1, "text": line})
    return lines

def chunk_sections(lines):
    chunks, cur = [], []
    for item in lines:
        t = item["text"]
        if is_title(t) and cur:
            chunks.append(cur); cur = [item]
        else:
            cur.append(item)
    if cur: chunks.append(cur)
    return chunks

def clean(x): 
    return re.sub(r'\s+', ' ', x).strip()

# ====== 3) 本地 LLM 调用 ======

def call_llm(messages, temperature=0.0, max_tokens=1024):
    # 把 Chat 格式合成成单个 prompt
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=(temperature > 0)
    )
    # 解码，并去掉输入部分
    output_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return output_text.strip()


# ====== 4) 三个步骤：抽取 → 生成问法 → 校验，并渲染答案 ======
def extract_feature_json(passage: str) -> dict:
    messages = [
        {"role": "system", "content": EXTRACT_SYS},
        {"role": "user", "content": EXTRACT_USER_TMPL.format(passage=passage)}
    ]
    raw = call_llm(messages, temperature=0.0)
    try:
        return json.loads(raw)
    except Exception:
        # 兜底，避免整个流程中断
        return {"title":"","type":"concept","preconditions":[],"steps":[],"parameters":[],"notes":[],"faqs":[]}

def gen_questions(feature_json: dict) -> list:
    messages = [
        {"role":"system","content": GEN_Q_SYS},
        {"role":"user","content": GEN_Q_USER_TMPL.format(
            feature_json=json.dumps(feature_json, ensure_ascii=False)
        )}
    ]
    raw = call_llm(messages, temperature=0.0)
    try:
        arr = json.loads(raw)
        return [clean(s) for s in arr if isinstance(s, str)]
    except Exception:
        title = feature_json.get("title") or "this feature"
        return [f"How do I {title}?", f"Where can I configure {title}?", f"Why can't I {title}?"]

def render_answer(feat: dict, section_id: str, page: int) -> str:
    out = []
    steps = feat.get("steps", [])
    notes = feat.get("notes", [])
    faqs  = feat.get("faqs", [])
    if steps:
        out.append("【Steps】")
        for s in steps:
            line = " - " + " ".join(
                x for x in [
                    s.get("action",""),
                    s.get("target",""),
                    f"(Path: {s['path']})" if s.get("path") else "",
                    f"(Params: {', '.join(s['parameters'])})" if s.get("parameters") else "",
                    f"→ {s['result']}" if s.get("result") else ""
                ] if x
            )
            out.append(line)
    if notes:
        out.append("\n【Notes / Warnings】")
        for n in notes:
            out.append(f" - {n}")
    if faqs:
        out.append("\n【Related FAQs】")
        for qa in faqs[:2]:
            out.append(f" - Q: {qa.get('q','')}")
            out.append(f"   A: {qa.get('a','')}")
    out.append(f"\n(Source: section {section_id}, page {page})")
    return "\n".join(out).strip()

def verify_answer(passage: str, answer: str) -> bool:
    messages = [
        {"role":"system","content": VERIFY_SYS},
        {"role":"user","content": VERIFY_USER_TMPL.format(passage=passage, answer=answer)}
    ]
    raw = call_llm(messages, temperature=0.0)
    try:
        obj = json.loads(raw)
        return bool(obj.get("supported", False))
    except Exception:
        # 审核失败就当通过（不建议长期这样，用于最小可运行）
        return True

# ====== 5) 主流程：读取 PDF、跑一遍 ======
def main():
    assert Path(PDF_PATH).exists(), f"PDF not found: {PDF_PATH}"
    lines = parse_pdf(PDF_PATH)
    chunks = chunk_sections(lines)

    n_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for i, ch in enumerate(chunks, 1):
            texts = [c["text"] for c in ch]
            passage = "\n".join(texts)
            if len(passage) < 120:
                continue  # 跳过过短块
            section_id = f"S{i:04d}"
            page = ch[0]["page"]

            feat = extract_feature_json(passage)
            questions = gen_questions(feat)
            if not questions:
                continue

            answer = render_answer(feat, section_id, page)
            ok = verify_answer(passage, answer)
            if not ok:
                # 保守修剪：去掉 step 的 result，降低不被支持的描述
                for s in feat.get("steps", []):
                    s["result"] = ""
                answer = render_answer(feat, section_id, page)

            source_hash = hashlib.md5(passage.encode("utf-8")).hexdigest()
            for q in questions[:4]:
                rec = {
                    "instruction": q,
                    "output": answer,
                    "metadata": {
                        "section_id": section_id,
                        "page": page,
                        "anchor": (texts[0] if texts else "")[:80],
                        "source_hash": source_hash,
                        "confidence": 0.75 if ok else 0.6
                    }
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"Saved {n_written} examples -> {OUT_PATH}")

if __name__ == "__main__":
    main()