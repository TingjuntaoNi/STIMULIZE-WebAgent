#!/usr/bin/env python3
"""
rag_pipeline.py
简化的RAG问答批处理脚本

功能：
1. 从data/test_questions.json读取问题
2. 直接调用b3_generate.py中的answer_question函数
3. 输出结果到output_answer文件夹，按时间戳命名
"""

import json
import os
from datetime import datetime
from b3_generate import answer_question


def batch_generate(input_path: str, output_path: str):
    """
    批量生成问答结果并保存为JSON
    Args:
        input_path: 输入问题文件路径
        output_path: 输出结果文件路径
    """
    # 读取输入问题
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    total = len(questions)
    print(f"Loaded {total} questions from {input_path}")

    # 循环调用 RAG 生成器
    for i, q in enumerate(questions, start=1):
        qid = q.get("id", f"Q{i}")
        question_text = q["question"]
        print(f"({i}/{total}) Generating answer for: {question_text}")

        try:
            result = answer_question(question_text)  # 使用已有函数
            answer_text = result.get("answer", "")
            
            # 从sources中提取used_chunks
            sources = result.get("sources", [])
            used_chunks = [source.get("chunk_id", "") for source in sources if source.get("chunk_id")]
            
        except Exception as e:
            print(f"Failed to answer {qid}: {e}")
            answer_text = f"Error: {e}"
            used_chunks = []

        # 保存结果
        results.append({
            "id": qid,
            "question": question_text,
            "answer": answer_text,
            "used_chunks": used_chunks
        })

    # 写入结果文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path.replace(".json", f"_{timestamp}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"All results saved to: {output_file}")
    return output_file


def main():
    """主函数"""
    print("=== RAG Pipeline Started ===")
    
    # 设置路径
    input_file = "data/test_questions.json"
    output_file = "output_answer/rag_answers.json"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    try:
        result_file = batch_generate(input_file, output_file)
        
        print(f"\n=== Pipeline Completed ===")
        print(f"Output file: {result_file}")
        
        # 显示输出文件内容预览
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"\nOutput Preview:")
        for result in results[:2]:  # 显示前两个结果
            print(f"- {result['id']}: {result['question'][:50]}...")
            print(f"  Answer: {result['answer'][:100]}...")
            print(f"  Used chunks: {len(result['used_chunks'])}")
            print()
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()