# 医疗知识图谱实体提取工具

## 项目简介

本项目是一个用于从医学文本中提取实体（症状、疾病、药物、检查）的工具，基于Qwen2.5-3B模型实现。

## 功能特点

- 支持从英文医学文本中提取中文实体
- 自动将英文文本翻译成中文
- 支持使用本地模型，无需连接Hugging Face
- 提供命令行参数配置

## 环境要求

- Python 3.8+
- PyTorch 2.1.0+
- 其他依赖见requirements.txt

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 使用默认模型

```bash
python main.py
```

这将使用Qwen/Qwen2.5-3B模型从默认输入文件中提取实体。

### 2. 使用本地模型

如果您无法连接到Hugging Face网站，可以使用本地下载好的模型：

```bash
python main.py --model-path /path/to/your/local/model
```

### 3. 自定义输入输出

```bash
python main.py --model-path /path/to/your/local/model --input-file /path/to/input.jsonl --output-dir /path/to/output
```

## 命令行参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| --model-path | str | Qwen/Qwen2.5-3B | 模型名称或本地模型路径 |
| --input-file | str | data/raw/Open-Patients.jsonl | 输入文件路径 |
| --output-dir | str | data/processed | 输出目录路径 |

## 本地模型准备

1. 从Hugging Face下载Qwen2.5-3B模型到本地
2. 确保模型文件结构完整，包含：
   - config.json
   - model.safetensors 或 pytorch_model.bin
   - tokenizer.json
   - tokenizer_config.json
   - ...等必要文件

## 项目结构

```
exp03-medical-knowledge-graph/
├── models/              # 模型相关代码
│   └── entity_extractor.py   # 医学实体提取器
├── utils/               # 工具函数
│   ├── data_processor.py     # 数据处理
│   └── translation.py        # 翻译功能
├── data/                # 数据目录
│   ├── raw/             # 原始数据
│   └── processed/       # 处理后的数据
├── main.py              # 主程序入口
├── requirements.txt     # 依赖列表
└── README.md            # 项目说明
```

## 注意事项

1. 首次运行时会自动下载模型（如果使用在线模型）
2. 模型较大（约7GB），请确保有足够的磁盘空间
3. 使用CPU运行会比较慢，建议使用CUDA GPU加速
4. 如果使用本地模型，请确保模型文件完整且与代码兼容

## 故障排除

### 无法连接Hugging Face

请使用本地模型：
```bash
python main.py --model-path /path/to/your/local/model
```

### 模型加载失败

1. 检查本地模型路径是否正确
2. 确保模型文件完整
3. 尝试使用较低的精度运行（CPU模式下会自动使用float32）

### 内存不足

1. 尝试在CPU上运行
2. 减少输入文本长度
3. 增加系统内存或使用更大内存的机器

## 联系方式

如有问题，请联系项目维护者。