import json
import random

# 基础句子模板：新增【相似度分数】列（强对齐=0.9，弱对齐=0.6，强矛盾=0.1）
base_sentences = [
    ("The cat sat on the mat.", "A cat was seated on the mat.", "ALIGNED", 0.9),  # 强对齐
    ("I love eating apples.", "I hate eating apples.", "CONTRADICT", 0.1),       # 强矛盾
    ("She went to the park yesterday.", "Yesterday, she visited the park.", "ALIGNED", 0.9),
    ("Python is a programming language.", "Python is used for data analysis.", "ALIGNED", 0.6),  # 弱对齐
    ("The temperature is 30°C today.", "It's freezing cold outside today.", "CONTRADICT", 0.1),
    ("He plays basketball every day.", "He never plays basketball.", "CONTRADICT", 0.1),
    ("The book is on the table.", "The table has a book on it.", "ALIGNED", 0.9)
]

# 生成1000条数据（保留原有逻辑，新增similarity_score字段）
paper_data = []
for i in range(1000):
    idx = random.randint(0, len(base_sentences)-1)
    text_a, text_b, label, similarity_score = base_sentences[idx]
    paper_data.append({
        "text_a": text_a,
        "text_b": text_b,
        "label": label,
        "similarity_score": similarity_score  # 新增字段
    })

# 保存（覆盖原有数据文件）
with open("data/train_paper.jsonl", "w", encoding="utf-8") as f:
    for item in paper_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 生成{len(paper_data)}条数据，新增相似度分数字段")