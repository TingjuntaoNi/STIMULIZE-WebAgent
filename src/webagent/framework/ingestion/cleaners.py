import re

WHITESPACE_RE = re.compile(r"[ \t\u00A0]+")

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 合并多空格 → 单空格
    s = WHITESPACE_RE.sub(" ", s)
    # 合并多行空行
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def drop_headers_footers(s: str) -> str:
    """
    极简示例：删除常见页眉页脚模式（比如 'Stimulize User Manual Page 12'）。
    真实项目建议把模式配置化。
    """
    s = re.sub(r"(?im)^\s*Stimulize User Manual Page \d+\s*$", "", s)
    return s

def basic_clean(s: str) -> str:
    s = drop_headers_footers(s)
    s = normalize_whitespace(s)
    return s
