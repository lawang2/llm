
import subprocess
import os
# test file, ignore this

# 要下载的模型和数据集名称
models = [
    "openai/whisper-large-v3-turbo",
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    "microsoft/table-transformer-detection",
    "bularisai/multilingual-sentiment-analysis",
    "dslim/bert-base-NER",
    "deepset/roberta-base-squad2",
    "acebook/bart-large-cnn",
    "Falconsai/nsfw_image_detection",
    # 
    "bert-base-cased",
    "distilbert-base-uncased",
    "facebook/opt-125m",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "THUDM/chatglm3-6b",
    "openai/whisper-large-v2"
]

datasets = [
    "yelp_review_full",
    "squad_v2",
    "squad",
    "mozilla-foundation/common_voice_11_0"
]

# 下载模型
for model in models:
    model_dir = os.path.join("models", model)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Downloading model: {model}")
    subprocess.run([
        "huggingface-cli", "repo", "download", model,
        "--type", "model"
    ], check=True)

# 下载数据集
for dataset in datasets:
    dataset_dir = os.path.join("datasets", dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Downloading dataset: {dataset}")
    subprocess.run([
        "huggingface-cli", "repo", "download", dataset,
        "--type", "dataset"
    ], check=True)



# from deepseek import DeepSeek
# client = DeepSeek(api_key="your_deepseek_api_key")

# from dashscope import Generation
# client = Generation()
