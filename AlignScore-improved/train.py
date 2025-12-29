import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ===================== 1. 数据集类定义（含相似度分数读取） =====================
class AlignDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        # 读取数据（新增similarity_score字段）
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append({
                    "text_a": item["text_a"],
                    "text_b": item["text_b"],
                    "label": 0 if item["label"] == "ALIGNED" else 1,  # 标签映射：ALIGNED=0，CONTRADICT=1
                    "similarity_score": item["similarity_score"]  # 新增：读取相似度分数
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 返回：文本a、文本b、标签、相似度分数
        return item["text_a"], item["text_b"], item["label"], item["similarity_score"]


# ===================== 2. 数据加载器函数（含自定义collate_fn） =====================
def get_train_loader(data_path, batch_size=4):
    dataset = AlignDataset(data_path)

    # 自定义collate_fn：统一批次数据格式（解决字符串+张量混合问题）
    def collate_fn(batch):
        text_a = [x[0] for x in batch]  # 字符串列表
        text_b = [x[1] for x in batch]  # 字符串列表
        labels = torch.tensor([x[2] for x in batch], dtype=torch.long)  # 标签张量
        similarity_scores = torch.tensor([x[3] for x in batch], dtype=torch.float)  # 相似度分数张量
        return text_a, text_b, labels, similarity_scores

    # 构建DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn  # 使用自定义collate_fn
    )
    return loader


# ===================== 3. 极简版ALIGNSCORE模型（无需下载预训练模型） =====================
class AlignScoreModel(torch.nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256):
        super().__init__()
        # 简单文本编码层（替代RoBERTa）
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim * 2, hidden_dim)  # 拼接text_a+text_b的嵌入
        self.fc2 = torch.nn.Linear(hidden_dim, 2)  # 二分类
        self.dropout = torch.nn.Dropout(0.1)
        # 简单分词器（按空格分词，映射为数字）
        self.vocab = {}
        self.vocab_size = vocab_size
        self.max_len = 128

    def simple_tokenize(self, text_list):
        """极简分词：按空格分词，映射为数字ID"""
        token_ids = []
        for text in text_list:
            tokens = text.lower().split()
            ids = []
            for token in tokens:
                if token not in self.vocab:
                    if len(self.vocab) < self.vocab_size:
                        self.vocab[token] = len(self.vocab) + 1  # 0留给padding
                    else:
                        ids.append(0)
                        continue
                ids.append(self.vocab[token])
            # 固定长度（前128个token）
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            else:
                ids += [0] * (self.max_len - len(ids))
            token_ids.append(ids)
        return torch.tensor(token_ids, dtype=torch.long)

    def encode_text(self, text_a_list, text_b_list):
        """编码文本对"""
        a_ids = self.simple_tokenize(text_a_list)
        b_ids = self.simple_tokenize(text_b_list)
        # 嵌入+均值池化
        a_embed = self.embedding(a_ids).mean(dim=1)  # [batch, embed_dim]
        b_embed = self.embedding(b_ids).mean(dim=1)  # [batch, embed_dim]
        # 拼接文本a和b的嵌入
        concat_embed = torch.cat([a_embed, b_embed], dim=1)  # [batch, embed_dim*2]
        return concat_embed

    def forward(self, text_a, text_b):
        """前向传播：返回logits和embeddings"""
        concat_embed = self.encode_text(text_a, text_b)
        # 特征提取
        hidden = self.fc1(concat_embed)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        # 分类输出
        logits = self.fc2(hidden)
        # embeddings用concat_embed（替代原CLS token）
        return logits, concat_embed


# ===================== 4. 动态加权损失计算函数 =====================
def get_contrast_weight(similarity_score):
    """动态计算对比损失权重：弱语义样本权重高，强语义样本权重低"""
    if 0.3 < similarity_score < 0.7:
        return 0.5  # 弱语义样本（重点学）
    else:
        return 0.1  # 强语义样本（少花精力）


def compute_contrast_loss(embeddings):
    """计算对比损失（InfoNCE）"""
    batch_size = embeddings.size(0)
    pos_mask = torch.eye(batch_size, device=embeddings.device)  # 正样本掩码（自身）
    neg_mask = 1 - pos_mask  # 负样本掩码（其他样本）
    # 计算余弦相似度
    sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    # 对比损失计算
    contrast_loss = -torch.log(
        torch.exp(sim * pos_mask).sum(1) / (torch.exp(sim * neg_mask).sum(1) + 1e-8)
    ).mean()
    return contrast_loss


def compute_total_loss(logits, labels, embeddings, similarity_scores):
    """总损失：交叉熵损失 + 动态加权对比损失"""
    # 交叉熵损失（分类损失）
    ce_loss = F.cross_entropy(logits, labels)
    # 对比损失
    contrast_loss = compute_contrast_loss(embeddings)
    # 动态权重计算
    contrast_weights = [get_contrast_weight(score.item()) for score in similarity_scores]
    avg_contrast_weight = torch.tensor(contrast_weights, device=logits.device).mean()
    # 总损失
    total_loss = ce_loss * 0.7 + contrast_loss * avg_contrast_weight
    return total_loss


# ===================== 5. 训练函数 =====================
def train_model(model, train_loader, epochs=10, lr=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # 读取批次数据：text_a, text_b, 标签, 相似度分数
            text_a, text_b, labels, similarity_scores = batch

            # 前向传播
            logits, embeddings = model(text_a, text_b)

            # 计算动态加权损失
            loss = compute_total_loss(logits, labels, embeddings, similarity_scores)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印每轮损失
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "models/alignscore_dynamic_weight.pth")
    print("✅ 模型训练完成，已保存为alignscore_dynamic_weight.pth")


# ===================== 6. 主函数（执行训练） =====================
if __name__ == "__main__":
    # 初始化极简模型
    model = AlignScoreModel(vocab_size=10000, embed_dim=128, hidden_dim=256)
    # 加载数据
    train_loader = get_train_loader("data/train_paper.jsonl", batch_size=4)
    # 开始训练
    train_model(model, train_loader, epochs=10)