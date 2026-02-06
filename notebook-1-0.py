# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# -*- coding: utf-8 -*-
"""
阿尔茨海默症多模态预测系统 
功能：结合文本和音频特征，使用深度学习模型预测认知状态
分类标签：CTRL（健康）、ADs（阿尔茨海默症）
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import pickle
import warnings
import requests
from collections import defaultdict
warnings.filterwarnings('ignore')

# 设置中文字体，解决绘图乱码问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 手动下载BERT模型文件的函数
def download_bert_files(cache_dir="./bert_cache"):
    """手动下载BERT模型必要文件，跳过聊天模板检查"""
    os.makedirs(cache_dir, exist_ok=True)

    files = [
        "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
        "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"
    ]

    for url in files:
        filename = url.split('/')[-1]
        filepath = os.path.join(cache_dir, filename)

        if not os.path.exists(filepath):
            print(f"正在下载: {filename}")
            response = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"已存在: {filename}")

    return cache_dir

# ========================== 数据加载与预处理 ==========================
class AlzheimerDataset(Dataset):
    """阿尔茨海默症数据集类"""
    def __init__(self, text_data, audio_features, labels):
        self.text_data = text_data
        self.audio_features = audio_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.text_data[idx],
            'audio': torch.tensor(self.audio_features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_text_data(tsv_dir):
    """加载并预处理文本数据"""
    text_data = {}
    tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith('.tsv')]

    for file in tqdm(tsv_files, desc="加载文本数据"):
        uuid = file[:-4]
        try:
            df = pd.read_csv(os.path.join(tsv_dir, file), sep='\t')
            if 'value' in df.columns:
                # 更细致的文本清洗
                texts = df['value'].dropna().tolist()
                cleaned_texts = []
                for text in texts:
                    # 移除特殊字符但保留必要标点
                    cleaned = re.sub(r'[^\w\s.,!?]', '', str(text).lower())
                    # 移除多余空格
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    cleaned_texts.append(cleaned)
                # 保留句子结构，用句号分隔
                full_text = '. '.join(cleaned_texts)
                text_data[uuid] = full_text if full_text else "no text available"
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            text_data[uuid] = "no text available"

    return text_data

def load_audio_features(egemaps_dir):
    """加载并预处理音频特征 """
    audio_features = {}
    egemaps_files = [f for f in os.listdir(egemaps_dir) if f.endswith('.csv')]

    for file in tqdm(egemaps_files, desc="加载音频特征"):
        uuid = file[:-4]
        try:
            df = pd.read_csv(os.path.join(egemaps_dir, file), sep=';')
            df = df.drop(['name', 'frameTime'], axis=1, errors='ignore')

            # 处理缺失值
            df = df.fillna(df.mean())

            # 提取更多统计特征
            mean_features = df.mean(axis=0).values
            std_features = df.std(axis=0).values
            max_features = df.max(axis=0).values
            min_features = df.min(axis=0).values
            median_features = df.median(axis=0).values
            skew_features = df.skew(axis=0).values

            combined = np.concatenate([
                mean_features, std_features, max_features, 
                min_features, median_features, skew_features
            ])
            audio_features[uuid] = combined
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            if 'df' in locals():
                audio_features[uuid] = np.zeros(6 * df.shape[1])
            else:
                audio_features[uuid] = np.zeros(150)  # 增加默认特征维度

    return audio_features

def prepare_all_data(label_df, text_data, audio_features):
    """准备所有数据用于交叉验证"""
    label_df = label_df[label_df['label'].isin(['CTRL', 'AD'])]
    label_encoder = LabelEncoder()
    label_df['label_encoded'] = label_encoder.fit_transform(label_df['label'])

    texts = []
    audios = []
    labels = []
    uuids = []

    for _, row in label_df.iterrows():
        uuid = row['uuid']
        if uuid in text_data and uuid in audio_features:
            texts.append(text_data[uuid])
            audios.append(audio_features[uuid])
            labels.append(row['label_encoded'])
            uuids.append(uuid)

    # 标准化音频特征
    scaler = StandardScaler()
    audios = scaler.fit_transform(audios)

    print(f"总数据样本数: {len(labels)}")
    print(f"类别分布: {np.bincount(labels)}")

    return texts, audios, labels, uuids, label_encoder, scaler

# ========================== 深度学习模型 ==========================
class BertTextModel(nn.Module):
    """基于BERT的文本分类模型 """
    def __init__(self, model_dir, hidden_dim=768, dropout=0.5):
        super(BertTextModel, self).__init__()
        self.bert = BertModel.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            ignore_mismatched_sizes=True
        )
        # 微调更多层
        for param in list(self.bert.parameters())[:-20]:
            param.requires_grad = False

        # 增加更多层和正则化
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_out)
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)

        return logits, cls_out

class AudioCNNModel(nn.Module):
    """基于CNN的音频分类模型 """
    def __init__(self, input_dim, output_dim=256, dropout=0.5):
        super(AudioCNNModel, self).__init__()
        # 更深的网络结构和更多正则化
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # 动态计算全连接层输入维度
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc_input_dim = self._calculate_fc_input_dim(input_dim)

        # 添加额外的线性层确保输出维度一致
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, output_dim)  # 输出指定维度的特征
        self.bn5 = nn.BatchNorm1d(output_dim)
        self.fc3 = nn.Linear(output_dim, 2)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def _calculate_fc_input_dim(self, input_dim):
        """计算卷积层输出后的维度"""
        x = torch.randn(1, 1, input_dim)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        return x.view(-1).shape[0]

    def forward(self, audio):
        x = audio.unsqueeze(1)  # 增加通道维度

        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.leaky_relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        features = self.leaky_relu(self.bn5(self.fc2(x)))  # 确保特征维度一致
        x = self.dropout(features)
        logits = self.fc3(x)

        return logits, features

class MultimodalFusionModel(nn.Module):
    """多模态融合模型 - 使用注意力机制"""
    def __init__(self, text_model, audio_model, text_feature_dim=768, audio_feature_dim=256, dropout=0.5):
        super(MultimodalFusionModel, self).__init__()
        self.text_model = text_model
        self.audio_model = audio_model

        # 解冻部分层进行微调
        for param in list(self.text_model.parameters())[-10:]:
            param.requires_grad = True
        for param in list(self.audio_model.parameters())[-10:]:
            param.requires_grad = True

        # 注意力机制融合 - 确保输入维度与实际特征维度匹配
        self.text_attention = nn.Linear(text_feature_dim, 1)
        self.audio_attention = nn.Linear(audio_feature_dim, 1)

        # 融合后的全连接层
        self.fusion = nn.Linear(text_feature_dim + audio_feature_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text_inputs, audio):
        input_ids, attention_mask = text_inputs
        text_logits, text_features = self.text_model(input_ids, attention_mask)
        audio_logits, audio_features = self.audio_model(audio)

        # 打印特征维度用于调试
        # print(f"文本特征维度: {text_features.shape}, 音频特征维度: {audio_features.shape}")

        # 计算注意力权重
        text_attn = self.softmax(self.text_attention(text_features))
        audio_attn = self.softmax(self.audio_attention(audio_features))

        # 应用注意力
        text_features_weighted = text_features * text_attn
        audio_features_weighted = audio_features * audio_attn

        # 融合特征
        combined = torch.cat([text_features_weighted, audio_features_weighted], dim=1)

        x = self.leaky_relu(self.bn1(self.fusion(combined)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)

        return logits

# ========================== 训练与评估函数 ==========================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=20, model_name="model", class_weights=None):
    best_val_auc = 0.0
    best_model_weights = None
    history = {
        'train_loss': [], 'train_auc': [],
        'val_loss': [], 'val_auc': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 训练"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)

            if isinstance(model, MultimodalFusionModel):
                outputs = model((input_ids, attention_mask), audio)
            elif isinstance(model, BertTextModel):
                outputs, _ = model(input_ids, attention_mask)
            else:  # AudioCNNModel
                outputs, _ = model(audio)

            # 计算损失
            if class_weights is not None:
                loss = criterion(outputs, labels)
                # 应用类别权重
                weights = class_weights[labels]
                loss = (loss * weights).mean()
            else:
                loss = criterion(outputs, labels)

            train_loss += loss.item() * labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            # 调整梯度裁剪值
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            # 根据调度器类型更新
            if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_auc = roc_auc_score(train_labels, train_preds)
        # 接收evaluate_model返回的5个值
        val_loss, val_auc, _, _, _ = evaluate_model(model, val_loader, criterion, class_weights)

        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"训练损失: {train_loss:.4f}, 训练AUC: {train_auc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证AUC: {val_auc:.4f}\n")

        # 处理ReduceLROnPlateau调度器
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_weights = model.state_dict()

    model.load_state_dict(best_model_weights)
    return model, history, best_val_auc

def evaluate_model(model, dataloader, criterion, class_weights=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []  # 存储预测概率
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)

            if isinstance(model, MultimodalFusionModel):
                outputs = model((input_ids, attention_mask), audio)
            elif isinstance(model, BertTextModel):
                outputs, _ = model(input_ids, attention_mask)
            else:  # AudioCNNModel
                outputs, _ = model(audio)

            # 计算损失
            if class_weights is not None:
                loss = criterion(outputs, labels)
                weights = class_weights[labels]
                loss = (loss * weights).mean()
            else:
                loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs[:, 1])  # 存储正类的概率
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, auc, all_labels, all_preds, all_probs

def plot_metrics(history, model_name, fold=None):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    title = f'{model_name} - 损失曲线'
    if fold is not None:
        title += f' (Fold {fold})'
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_auc'], label='训练AUC')
    plt.plot(history['val_auc'], label='验证AUC')
    title = f'{model_name} - AUC曲线'
    if fold is not None:
        title += f' (Fold {fold})'
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    filename = f'{model_name}_metrics'
    if fold is not None:
        filename += f'_fold{fold}'
    plt.savefig(f'{filename}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, fold=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    title = f'{model_name} - 混淆矩阵'
    if fold is not None:
        title += f' (Fold {fold})'
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    filename = f'{model_name}_confusion_matrix'
    if fold is not None:
        filename += f'_fold{fold}'
    plt.savefig(f'{filename}.png')
    plt.close()

def plot_cv_metrics(cv_results, model_name):
    """绘制交叉验证的性能指标"""
    # 只使用我们实际存储的指标
    metrics = ['auc', 'accuracy']

    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 2, i)
        values = [cv_results[fold][metric] for fold in cv_results]
        plt.bar(range(1, len(values)+1), values)
        plt.axhline(np.mean(values), color='r', linestyle='--', label=f'平均值: {np.mean(values):.4f}')
        plt.title(f'{model_name} - {metric.upper()} 跨折表现')
        plt.xlabel('折数')
        plt.ylabel(metric.upper())
        plt.xticks(range(1, len(values)+1))
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_cv_metrics.png')
    plt.close()

# ========================== 交叉验证训练函数 ==========================
def cross_validate_model(model_creator, texts, audios, labels, tokenizer, 
                         n_splits=5, batch_size=8, num_epochs=20, 
                         model_name="model", class_names=None):
    """
    对模型进行交叉验证

    参数:
        model_creator: 函数，用于创建新的模型实例
        texts: 文本数据列表
        audios: 音频特征数组
        labels: 标签数组
        tokenizer: BERT分词器
        n_splits: 交叉验证折数
        batch_size: 批处理大小
        num_epochs: 训练轮数
        model_name: 模型名称
        class_names: 类别名称列表

    返回:
        cv_results: 包含每折结果的字典
        all_y_true: 所有测试样本的真实标签
        all_y_pred: 所有测试样本的预测标签
        all_y_probs: 所有测试样本的预测概率
    """
    # 初始化分层K折交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    cv_results = {}
    all_y_true = []
    all_y_pred = []
    all_y_probs = []

    # 创建BERT数据集包装器
    class BertDataset(Dataset):
        def __init__(self, texts, audios, labels, tokenizer, max_len=256):
            self.texts = texts
            self.audios = audios
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'audio': torch.tensor(self.audios[idx], dtype=torch.float32),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    # 遍历每一折
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        print(f"\n===== 交叉验证 折数 {fold}/{n_splits} =====")

        # 划分训练集和验证集
        train_texts = [texts[i] for i in train_idx]
        train_audios = audios[train_idx]
        train_labels = [labels[i] for i in train_idx]

        val_texts = [texts[i] for i in val_idx]
        val_audios = audios[val_idx]
        val_labels = [labels[i] for i in val_idx]

        # 创建数据集
        train_dataset = BertDataset(train_texts, train_audios, train_labels, tokenizer)
        val_dataset = BertDataset(val_texts, val_audios, val_labels, tokenizer)

        # 计算类别权重和采样权重
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(len(class_counts) / class_counts).to(device)
        print(f"类别权重: {class_weights}")

        # 创建加权采样器
        sample_weights = [class_weights[label].item() for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型、优化器和损失函数
        model = model_creator().to(device)
        criterion = nn.CrossEntropyLoss()

        # 根据模型类型设置不同的优化器参数
        if model_name.startswith('bert'):
            optimizer = optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8, weight_decay=0.01)
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps
            )
        elif model_name.startswith('audio'):
            optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
        else:  # 多模态模型
            optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8, weight_decay=0.01)
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps
            )

        # 训练模型
        best_model, history, best_val_auc = train_model(
            model, train_loader, val_loader, criterion, 
            optimizer, scheduler, num_epochs, 
            f"{model_name}_fold{fold}", class_weights
        )

        # 评估模型
        _, _, y_true, y_pred, y_probs = evaluate_model(best_model, val_loader, criterion, class_weights)

        # 保存结果
        cv_results[fold] = {
            'auc': best_val_auc,
            'accuracy': accuracy_score(y_true, y_pred),
            'report': classification_report(y_true, y_pred, output_dict=True),
            'history': history
        }

        # 收集所有预测结果
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_probs.extend(y_probs)

        # 打印当前折的评估报告
        print(f"\n{model_name} 折数 {fold} 评估:")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # 绘制当前折的指标图
        plot_metrics(history, model_name, fold)
        plot_confusion_matrix(y_true, y_pred, class_names, model_name, fold)

        # 保存当前折的最佳模型
        torch.save(best_model.state_dict(), f'{model_name}_best_fold{fold}.pth')

    # 计算交叉验证的平均性能
    print(f"\n===== {model_name} 交叉验证结果 =====")
    print(f"平均AUC: {np.mean([cv_results[fold]['auc'] for fold in cv_results]):.4f} ± {np.std([cv_results[fold]['auc'] for fold in cv_results]):.4f}")
    print(f"平均准确率: {np.mean([cv_results[fold]['accuracy'] for fold in cv_results]):.4f} ± {np.std([cv_results[fold]['accuracy'] for fold in cv_results]):.4f}")

    # 打印整体评估报告
    print("\n整体评估报告:")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))

    # 绘制交叉验证指标图
    plot_cv_metrics(cv_results, model_name)
    plot_confusion_matrix(all_y_true, all_y_pred, class_names, f"{model_name}_cv_combined")

    return cv_results, all_y_true, all_y_pred, all_y_probs

# ========================== 主函数 ==========================
def main():
    # 1. 数据路径设置
    data_dir = "./kaggle/input/alzheimer/data"
    tsv_dir = os.path.join(data_dir, "tsv2")
    egemaps_dir = os.path.join(data_dir, "egemaps2")
    train_label_path = os.path.join(data_dir, "2_final_list_train.csv")

    # 2. 加载数据
    print("===== 加载数据 =====")
    train_labels = pd.read_csv(train_label_path)
    text_data = load_text_data(tsv_dir)
    audio_features = load_audio_features(egemaps_dir)

    # 3. 准备所有数据用于交叉验证
    print("\n===== 准备所有数据 =====")
    texts, audios, labels, uuids, label_encoder, scaler = prepare_all_data(
        train_labels, text_data, audio_features
    )

    # 保存编码器和标准化器
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('audio_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 4. 加载BERT模型和分词器
    print("\n===== 手动下载BERT模型 =====")
    bert_cache_dir = download_bert_files()

    print("\n===== 加载BERT分词器 =====")
    bert_tokenizer = BertTokenizer.from_pretrained(
        bert_cache_dir,
        local_files_only=True,
        trust_remote_code=False
    )

    # 模型参数设置
    num_epochs = 30
    batch_size = 8
    n_splits = 5  # 5折交叉验证
    class_names = label_encoder.classes_

    # 5. 交叉验证BERT文本模型
    print("\n===== 交叉验证BERT文本模型 =====")
    def create_bert_model():
        return BertTextModel(bert_cache_dir)

    bert_cv_results, _, _, _ = cross_validate_model(
        create_bert_model, texts, audios, labels, bert_tokenizer,
        n_splits=n_splits, batch_size=batch_size, num_epochs=num_epochs,
        model_name="bert_text_model", class_names=class_names
    )

    # 6. 交叉验证音频CNN模型
    print("\n===== 交叉验证音频CNN模型 =====")
    audio_dim = audios.shape[1]
    def create_audio_model():
        return AudioCNNModel(input_dim=audio_dim, output_dim=256)  # 指定输出维度

    audio_cv_results, _, _, _ = cross_validate_model(
        create_audio_model, texts, audios, labels, bert_tokenizer,
        n_splits=n_splits, batch_size=batch_size, num_epochs=num_epochs,
        model_name="audio_cnn_model", class_names=class_names
    )

    # 7. 交叉验证多模态融合模型
    print("\n===== 交叉验证多模态融合模型 =====")
    def create_fusion_model():
        bert_model = BertTextModel(bert_cache_dir)
        audio_model = AudioCNNModel(input_dim=audio_dim, output_dim=256)  # 确保音频特征维度为256
        return MultimodalFusionModel(bert_model, audio_model, 
                                    text_feature_dim=768, audio_feature_dim=256)  # 明确指定特征维度

    fusion_cv_results, _, _, _ = cross_validate_model(
        create_fusion_model, texts, audios, labels, bert_tokenizer,
        n_splits=n_splits, batch_size=batch_size, num_epochs=num_epochs,
        model_name="multimodal_fusion_model", class_names=class_names
    )

    # 8. 比较不同模型的性能
    print("\n===== 模型性能比较 =====")
    models = {
        "BERT文本模型": bert_cv_results,
        "音频CNN模型": audio_cv_results,
        "多模态融合模型": fusion_cv_results
    }

    for name, results in models.items():
        avg_auc = np.mean([results[fold]['auc'] for fold in results])
        avg_acc = np.mean([results[fold]['accuracy'] for fold in results])
        print(f"{name}:")
        print(f"  平均AUC: {avg_auc:.4f}")
        print(f"  平均准确率: {avg_acc:.4f}\n")

    print("\n所有交叉验证完成，最佳模型已按折保存！")

if __name__ == "__main__":
    main()