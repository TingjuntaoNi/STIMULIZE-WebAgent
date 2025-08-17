from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Iterable, List, Dict
import typer
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from .framework.ingestion.loaders import load_pdf
from .framework.ingestion.cleaners import basic_clean
from .framework.ingestion.chunker import chunk_text

app = typer.Typer(help="STIMULIZE WebAgent CLI")

def _hash_id(text: str, source: str, page: int, ctype: str, idx: int) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    h.update(f"|{source}|{page}|{ctype}|{idx}".encode("utf-8"))
    return h.hexdigest()[:16]

def _dedup(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for c in chunks:
        key = hashlib.md5(c["text"].encode("utf-8")).hexdigest()  # 轻量文本去重
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def _iter_pdf_files(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*.pdf")):
            yield p
    else:
        raise typer.BadParameter(f"Path {path} is neither a PDF nor a directory of PDFs.")

@app.command(help="Ingest PDF(s) into JSONL chunks. Accepts a file or a directory.")
def ingest(
    src: Path = typer.Argument(..., help="Path to a PDF file or a directory containing PDFs."),
    out: Path = typer.Argument(..., help="Output JSONL file path."),
    chunk_size: int = typer.Option(800, help="Approx chunk size (chars)."),
    overlap: int = typer.Option(150, help="Overlap size (chars)."),
    ocr_images: bool = typer.Option(True, help="Enable OCR for page images."),
    ocr_lang: str = typer.Option("eng+chi_sim", help="Tesseract languages, e.g., 'eng', 'chi_sim', or 'eng+chi_sim'."),
    dedup: bool = typer.Option(True, help="Enable simple text de-duplication."),
):
    out.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict] = []
    files = list(_iter_pdf_files(src))
    if not files:
        raise typer.BadParameter(f"No PDFs found under {src}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        t = progress.add_task("Ingesting PDFs", total=len(files))
        for pdf in files:
            try:
                data = load_pdf(str(pdf), ocr_images=ocr_images, ocr_lang=ocr_lang)
            except Exception as e:
                typer.secho(f"[warn] Failed to load {pdf}: {e}", fg=typer.colors.YELLOW)
                progress.advance(t)
                continue

            # 文本页
            for p in data.get("text_pages", []):
                clean = basic_clean(p.text)
                if not clean:
                    continue
                pieces = chunk_text(clean, size=chunk_size, overlap=overlap)
                for i, ch in enumerate(pieces, 1):
                    all_chunks.append({
                        "id": _hash_id(ch, pdf.name, p.page, "text", i),
                        "text": ch,
                        "metadata": {"source": pdf.name, "page": p.page, "type": "text"},
                    })

            # 图片 OCR
            for im in data.get("image_ocr", []):
                clean = basic_clean(im.ocr_text)
                if not clean:
                    continue
                pieces = chunk_text(clean, size=chunk_size, overlap=overlap)
                for i, ch in enumerate(pieces, 1):
                    all_chunks.append({
                        "id": _hash_id(ch, pdf.name, im.page, "image_ocr", i),
                        "text": ch,
                        "metadata": {"source": pdf.name, "page": im.page, "type": "image_ocr"},
                    })

            progress.advance(t)

    if dedup:
        before = len(all_chunks)
        all_chunks = _dedup(all_chunks)
        typer.echo(f"De-duplicated {before - len(all_chunks)} duplicate chunks.")

    with out.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    typer.secho(f"✅ Ingested {len(all_chunks)} chunks → {out}", fg=typer.colors.GREEN)

def main():
    app()

if __name__ == "__main__":
    main()
