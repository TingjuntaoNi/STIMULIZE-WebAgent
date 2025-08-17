import io
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError("Please install PyMuPDF: pip install pymupdf") from e

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class PageImageOCR:
    page: int
    ocr_text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1


def load_pdf_text(pdf_path: str) -> List[PageText]:
    """抽取每一页的“可复制文本”（保留段落换行）。"""
    pages: List[PageText] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            # 使用 page.get_text("text") 比 'blocks' 更稳妥，保持阅读顺序
            txt = page.get_text("text")
            pages.append(PageText(page=i+1, text=txt))
    return pages


def load_pdf_images_ocr(pdf_path: str, dpi: int = 300, lang: str = "eng") -> List[PageImageOCR]:
    """对页面中的位图做 OCR，未安装 pytesseract 时返回空列表。"""
    if not _HAS_TESS or Image is None:
        return []

    results: List[PageImageOCR] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            # 把整页渲染成位图再 OCR，简单可靠；如需更细粒度可遍历 page.get_images()
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes))
            try:
                ocr_text = pytesseract.image_to_string(pil_img, lang=lang)
            except Exception:
                ocr_text = ""
            if ocr_text.strip():
                results.append(PageImageOCR(page=i+1, ocr_text=ocr_text, bbox=(0, 0, pil_img.width, pil_img.height)))
    return results


def load_pdf(pdf_path: str, ocr_images: bool = True, ocr_lang: str = "eng") -> Dict[str, Any]:
    """统一的 PDF 加载接口：返回 {'text_pages': [...], 'image_ocr': [...]}。"""
    text_pages = load_pdf_text(pdf_path)
    image_ocr = load_pdf_images_ocr(pdf_path, lang=ocr_lang) if ocr_images else []
    return {"text_pages": text_pages, "image_ocr": image_ocr}
