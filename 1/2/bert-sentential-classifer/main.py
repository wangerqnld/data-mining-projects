import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier
import torch.nn as nn
import os
import gc

# 设置CUDA环境变量以支持GPU训练
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用第一个GPU，可根据需要修改

def set_hf_mirrors():
    """
    设置Hugging Face镜像，加速模型下载
    """
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 可选的其他镜像
    # os.environ['HF_ENDPOINT'] = 'https://huggingface.tuna.tsinghua.edu.cn'
    # os.environ['HF_ENDPOINT'] = 'https://mirror.sjtu.edu.cn/hugging-face'
    
    # 设置模型缓存目录（可选）
    os.environ['HF_HOME'] = './hf_cache'
    
# 设置镜像
set_hf_mirrors()

def evaluate(model, eval_loader, device):
    """
    评估模型性能
    
    参数:
        model: 模型对象
        eval_loader: 评估数据加载器
        device: 计算设备（CPU/GPU）
        
    返回:
        Tuple[float, float]: 平均损失和准确率
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            total_loss += loss.item()
            
    return total_loss / len(eval_loader), correct_predictions.double() / total_predictions

def train(train_texts, train_labels, val_texts=None, val_labels=None):
    """
    训练模型
    
    参数:
        train_texts (List[str]): 训练文本列表
        train_labels (List[int]): 训练标签列表
        val_texts (List[str], optional): 验证文本列表
        val_labels (List[int], optional): 验证标签列表
        
    返回:
        model: 训练好的模型
    """
    # 清理 GPU 缓存和内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # 加载配置
    config = Config()
    
    # 确保模型保存目录存在
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # 打印GPU信息
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print("未检测到GPU，使用CPU进行训练")
    
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # 使用正确的SentimentClassifier模型而不是BertModel
    model = SentimentClassifier("bert-base-chinese", config.num_classes)
    model.to(device)
    
    # 检查是否有多个GPU可用，如果有则使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"检测到多个GPU: {torch.cuda.device_count()}个")
        model = torch.nn.DataParallel(model)
    
    # 启用混合精度训练以加速GPU计算
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # 准备训练数据
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 准备验证数据
    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # 学习率调度器
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),  # 10%的预热步数
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_accuracy = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for batch in train_loop:
            optimizer.zero_grad()
            
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # 梯度缩放以防止下溢
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率调度器
            scheduler.step()
            
            total_loss += loss.item()
            # 更新进度条显示
            train_loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{config.num_epochs}')
        print(f'Average training loss: {avg_train_loss:.4f}')
        
        if val_texts is not None and val_labels is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                model.save_model(config.model_save_path)
    
    return model
if __name__ == "__main__":
    # 设置Hugging Face镜像
    set_hf_mirrors()
    
    # 加载配置
    config = Config()
    
    # 加载数据
    data_loader = DataLoaderClass(config)
    
     # 分别加载训练集、验证集和测试集（带进度条）
    print("正在加载训练数据...")
    train_texts, train_labels = data_loader.load_csv(config.train_path)
    
    print("正在加载验证数据...")
    val_texts, val_labels = data_loader.load_csv(config.dev_path)
    
    print("正在加载测试数据...")
    test_texts, test_labels = data_loader.load_csv(config.test_path)
    
    # 打印数据统计信息
    print(f"训练集: {len(train_texts)} 条样本")
    print(f"验证集: {len(val_texts)} 条样本") 
    print(f"测试集: {len(test_texts)} 条样本")
    
    # 训练模型
    train(train_texts, train_labels, val_texts, val_labels)
