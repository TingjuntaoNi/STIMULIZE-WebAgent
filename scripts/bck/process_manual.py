#!/usr/bin/env python3
"""
处理 STIMULIZE 用户手册PDF的脚本
流程：loaders.py -> cleaners.py -> chunker.py
"""

import os
import sys
import json
from pathlib import Path

# 添加src路径到Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from webagent.framework.ingestion.loaders import load_pdf
from webagent.framework.ingestion.cleaners import basic_clean
from webagent.framework.ingestion.chunker import chunk_text, make_chunk_objs


def main():
    # 定义文件路径
    manual_path = project_root / "files" / "STIMULIZE_UserManual_08042025.pdf"
    output_path = script_dir / "manual_chunks.json"
    
    print(f"项目根目录: {project_root}")
    print(f"处理文件: {manual_path}")
    print(f"输出文件: {output_path}")
    
    # 检查输入文件是否存在
    if not manual_path.exists():
        print(f"错误: 文件不存在 {manual_path}")
        return
    
    print("\n=== 步骤1: 使用 loaders.py 加载PDF ===")
    try:
        # 加载PDF，包含文本和OCR
        pdf_data = load_pdf(str(manual_path), ocr_images=True, ocr_lang="eng")
        text_pages = pdf_data["text_pages"]
        image_ocr = pdf_data["image_ocr"]
        
        print(f"成功加载PDF:")
        print(f"  - 文本页面数: {len(text_pages)}")
        print(f"  - OCR图像数: {len(image_ocr)}")
        
        # 显示前几页的文本长度
        for i, page in enumerate(text_pages[:3]):
            print(f"  - 第{page.page}页文本长度: {len(page.text)} 字符")
            
    except Exception as e:
        print(f"加载PDF时出错: {e}")
        return
    
    print("\n=== 步骤2: 使用 cleaners.py 清洗文本 ===")
    cleaned_pages = []
    
    # 清洗文本页面
    for page in text_pages:
        cleaned_text = basic_clean(page.text)
        if cleaned_text.strip():  # 只保留有内容的页面
            cleaned_pages.append({
                "page": page.page,
                "text": cleaned_text,
                "type": "text"
            })
    
    # 清洗OCR文本
    for ocr in image_ocr:
        cleaned_text = basic_clean(ocr.ocr_text)
        if cleaned_text.strip():
            cleaned_pages.append({
                "page": ocr.page,
                "text": cleaned_text,
                "type": "ocr",
                "bbox": ocr.bbox
            })
    
    print(f"清洗后保留的页面/OCR块数: {len(cleaned_pages)}")
    
    # 显示清洗效果示例
    if cleaned_pages:
        sample_page = cleaned_pages[0]
        original_length = len(text_pages[0].text) if text_pages else 0
        cleaned_length = len(sample_page["text"])
        print(f"清洗效果示例 (第{sample_page['page']}页):")
        print(f"  - 原始长度: {original_length} 字符")
        print(f"  - 清洗后长度: {cleaned_length} 字符")
        print(f"  - 清洗后前200字符: {sample_page['text'][:200]}...")
    
    print("\n=== 步骤3: 使用 chunker.py 进行文档切片 ===")
    all_chunks = []
    
    for page_data in cleaned_pages:
        page_num = page_data["page"]
        text = page_data["text"]
        content_type = page_data["type"]
        
        # 对每页文本进行切片
        chunks = chunk_text(
            text=text,
            size=800,      # 每个chunk约800字符
            overlap=150    # 重叠150字符
        )
        
        # 创建chunk对象
        chunk_objs = make_chunk_objs(
            chunks=chunks,
            source="STIMULIZE_UserManual_08042025.pdf",
            page=page_num,
            ctype=content_type
        )
        
        all_chunks.extend(chunk_objs)
        
        if chunks:
            print(f"第{page_num}页 ({content_type}): {len(chunks)} 个chunks")
    
    print(f"\n总共生成 {len(all_chunks)} 个文档切片")
    
    # 显示一些统计信息
    if all_chunks:
        chunk_lengths = [len(chunk["text"]) for chunk in all_chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        
        print(f"切片统计:")
        print(f"  - 平均长度: {avg_length:.1f} 字符")
        print(f"  - 最小长度: {min_length} 字符")
        print(f"  - 最大长度: {max_length} 字符")
    
    print("\n=== 步骤4: 保存结果 ===")
    try:
        # 保存为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"成功保存 {len(all_chunks)} 个文档切片到: {output_path}")
        
        # 显示第一个chunk作为示例
        if all_chunks:
            print(f"\n示例切片 (ID: {all_chunks[0]['id']}):")
            print(f"内容: {all_chunks[0]['text'][:300]}...")
            print(f"元数据: {all_chunks[0]['metadata']}")
            
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return
    
    print("\n=== 处理完成 ===")
    print(f"输出文件: {output_path}")
    print(f"总切片数: {len(all_chunks)}")


if __name__ == "__main__":
    main()