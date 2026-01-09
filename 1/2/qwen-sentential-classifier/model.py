import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(SentimentClassifier, self).__init__()
        # 加载模型并使用8-bit量化减少内存占用，同时启用CPU offload
        from transformers import BitsAndBytesConfig
        
        # 配置4-bit量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_classes)
        # 将分类器层移动到与模型相同的设备上
        self.classifier = self.classifier.to(next(self.qwen.parameters()).device)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # 获取最后一层隐藏状态
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state.mean(dim=1)
        
        # 确保pooled_output在与分类器相同的设备和数据类型上
        pooled_output = pooled_output.to(self.classifier.weight.device, dtype=self.classifier.weight.dtype)
        
        output = self.dropout(pooled_output)
        return self.classifier(output)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)