import json
import random
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    data_list = []
    for value in data["data"].values():
        data_list.append(value)
        
    return data_list

class GinkaDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BertTokenizer, max_len=128):
        self.data = load_data(data_path)  # 自定义数据加载函数
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_size = 32

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 文本处理
        text = random.choice(item["text"])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 噪声生成
        w, h = item["size"]
        noise = torch.randn(h, w, 1)
        
        # 目标矩阵填充
        target = torch.full((self.max_size, self.max_size), -100)  # 使用-100忽略填充区域
        target[:h, :w] = torch.tensor(item["map"])
        
        return {
            "noise": noise,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "map_size": torch.tensor([h, w]),
            "target": target
        }