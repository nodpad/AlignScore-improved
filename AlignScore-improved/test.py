import torch
from train import AlignScoreModel  # 从train.py导入改进后的模型类

# 加载改进后的模型
model = AlignScoreModel(vocab_size=10000, embed_dim=128, hidden_dim=256)
model.load_state_dict(torch.load("models/alignscore_dynamic_weight.pth"))
model.eval()  # 切换到评估模式

# 测试样本（重点验证弱语义对齐样本）
test_cases = [
    # (text_a, text_b, 预期标签)
    ("Python is a programming language.", "Python is used for data analysis.", "ALIGNED"),  # 弱对齐（核心验证点）
    ("I love eating apples.", "I hate eating apples.", "CONTRADICT"),  # 强矛盾
    ("The cat sat on the mat.", "A cat was seated on the mat.", "ALIGNED"),  # 强对齐
    ("The temperature is 30°C today.", "It's freezing cold outside today.", "CONTRADICT")  # 强矛盾
]

# 逐一样本测试
print("===== 改进后模型测试结果 =====")
for text_a, text_b, expected_label in test_cases:
    with torch.no_grad():  # 禁用梯度计算，提升速度
        logits, _ = model([text_a], [text_b])
        pred_label_id = torch.argmax(logits).item()
        pred_label = "ALIGNED" if pred_label_id == 0 else "CONTRADICT"
        # 输出结果
        print(f"文本A：{text_a}")
        print(f"文本B：{text_b}")
        print(f"预期标签：{expected_label} | 预测标签：{pred_label}")
        print("-" * 80)