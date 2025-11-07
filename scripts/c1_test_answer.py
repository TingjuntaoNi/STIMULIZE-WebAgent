import json
import os
import argparse
from datetime import datetime
from b3_generate import RAGGenerator

def batch_generate(input_path: str, output_path: str, model_path: str = None):
    """
    批量生成问答结果并保存为JSON
    Args:
        input_path: 输入问题文件路径
        output_path: 输出结果文件路径
        model_path: 大语言模型路径（可选）
    """
    # 初始化生成器
    print("[INFO] Initializing RAG generator...")
    generator = RAGGenerator(model_path=model_path) if model_path else RAGGenerator()
    print(f"[INFO] Using model: {generator.llm_path}")

    # 取输入问题
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    total = len(questions)
    print(f"[INFO] Loaded {total} questions from {input_path}")

    # 循环调用生成器
    for i, q in enumerate(questions, start=1):
        qid = q.get("id", f"Q{i}")
        question_text = q["question"]
        print(f"\n[INFO] ({i}/{total}) Generating answer for: {question_text}")

        try:
            result = generator.answer_question(question_text)
            answer_text = result.get("answer", "")
            used_chunks = result.get("used_chunks", [])
        except Exception as e:
            print(f"[ERROR] Failed to answer {qid}: {e}")
            answer_text = f"Error: {e}"
            used_chunks = []

        results.append({
            "id": qid,
            "question": question_text,
            "answer": answer_text,
            "used_chunks": used_chunks,
            "model": generator.llm_path
        })

    # 写入输出文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_path.replace(".json", f"_{timestamp}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n All results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch RAG Question Answering")
    parser.add_argument("--input", type=str, default="data/test_questions.json", help="Input JSON file path")
    parser.add_argument("--output", type=str, default="outputs/rag_answers.json", help="Output JSON file path")
    parser.add_argument("--model", type=str, default=None, help="Local LLM model path")
    args = parser.parse_args()

    batch_generate(args.input, args.output, args.model)



