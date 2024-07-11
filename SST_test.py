from datasets import load_dataset
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import time
import pandas as pd
import csv

def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

out_file_path = "results.csv"
model_dir = '/root/autodl-tmp/ZhipuAI/glm-4-9b-chat'
data_path = "data/test.tsv"
df = pd.read_csv(data_path, delimiter='\t', header=None, names=['label', 'text'])
"""
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
"""
tokenizer = DistilBertTokenizer.from_pretrained("model")
model = DistilBertForSequenceClassification.from_pretrained("model")
texts = df['text'].tolist()
labels = df['label'].tolist()

ds = load_dataset("linxinyuan/cola")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


total_time = 0
total_samples = 0
correct_predictions = 0
info_write = []
for text, true_lable in zip(texts, labels):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.time()
    once_time = end_time - start_time
    total_time += once_time
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy_epoch = (predicted_classes == true_lable).item()
    correct_predictions += accuracy_epoch
    current_memory_usage = torch.cuda.memory_allocated() / 1024
    total_samples += 1
    print(f"Pred:{predicted_classes.item()}, GD:{true_lable}")
    info_write.append(["SST-2", "Sentiment Analysis", once_time, current_memory_usage, accuracy_epoch])

with open(out_file_path, mode="w", newline="") as file:
    fieldnames = ["Dataset", "Task", "Inference Time (seconds)", "Memory Usage (MB)", "Accuracy (%)"]
    writer = csv.writer(file)
    for info in info_write:
        writer.writerow(info)


accuracy_total = correct_predictions / total_samples
throughput = total_samples / total_time
print(accuracy_total, throughput)
