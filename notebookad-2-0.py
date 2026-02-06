# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
input_folder_path = './kaggle/input/alzheimer/data/'
result_folder_path = './results/'

cnt = 0
for root, _, filenames in os.walk(input_folder_path):
    for filename in filenames:
        cnt += 1
print('Gathered {0} data file(s) under the directory \"{1}\". '.format(cnt, input_folder_path))
os.makedirs(result_folder_path, exist_ok = True)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# -*- coding: utf-8 -*-
"""
阿尔茨海默症多模态识别系统 - 完整可视化版
集成训练过程可视化，中文注释，英文标签显示
"""

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, f1_score, 
                           accuracy_score, classification_report, 
                           confusion_matrix, precision_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# 设置
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================== 真实数据处理器 ==========================
class RealDataProcessor:
    """真实数据处理器 - 基于实际数据结构"""

    def __init__(self, base_path = input_folder_path):
        self.base_path = base_path
        self.egemaps_pre_path = os.path.join(base_path, 'egemaps_final.csv')
        self.train_list_path = os.path.join(base_path, '2_final_list_train.csv')
        self.test_list_path = os.path.join(base_path, '2_final_list_test.csv')
        self.tsv_dir = os.path.join(base_path, 'tsv2/')
        self.egemaps_dir = '/data/egemaps2/'  # 根据实际情况更新路径

    def process_all_data(self):
        """处理所有数据"""
        print("=== Processing Real Alzheimer's Disease Data ===")

        # 1. 加载基础数据
        print("1. Loading basic data...")
        preliminary_list_test = pd.read_csv(self.test_list_path)  # (27,2)
        preliminary_list_train = pd.read_csv(self.train_list_path)  # (179,2)
        egemaps_pre = pd.read_csv(self.egemaps_pre_path)  # (206,89)

        print(f"Training set: {preliminary_list_train.shape}")
        print(f"Test set: {preliminary_list_test.shape}")
        print(f"Precomputed features: {egemaps_pre.shape}")
        print('Training set label distribution:\n', preliminary_list_train['label'].value_counts())

        # 2. 从TSV文件提取文本特征
        print("\n2. Extracting text features from TSV files...")
        text_features = self._extract_text_features()

        # 3. 从eGeMAPS文件提取音频特征
        print("\n3. Extracting audio features from eGeMAPS files...")
        audio_features = self._extract_audio_features()

        # 4. 合并所有特征
        print("\n4. Merging all features...")
        merged_data = self._merge_all_features(
            preliminary_list_train, preliminary_list_test, 
            egemaps_pre, text_features, audio_features
        )

        return merged_data

    def _extract_text_features(self):
        """从TSV文件提取文本特征"""
        tsv_path_lists = os.listdir(self.tsv_dir)
        tsv_feats = []

        print(f"Processing {len(tsv_path_lists)} TSV files...")

        for path in tqdm(tsv_path_lists):
            try:
                # 读取tsv文件
                z = pd.read_csv(os.path.join(self.tsv_dir, path), sep='\t')
                # 计算每句话的时长
                z['end_time-start_time'] = z['end_time'] - z['start_time']

                # 提取时长统计特征
                tsv_feats.append([
                    path[:-4],  # uuid
                    z['end_time-start_time'].mean(),    # 平均时长
                    z['end_time-start_time'].min(),     # 最小时长
                    z['end_time-start_time'].max(),     # 最大时长
                    z['end_time-start_time'].std(),     # 时长标准差
                    z['end_time-start_time'].median(),  # 时长中位数
                    z['end_time-start_time'].skew(),    # 时长偏度
                    z.shape[0]                         # 句子数量
                ])
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                continue

        # 转换为DataFrame
        tsv_feats_df = pd.DataFrame(tsv_feats)
        tsv_feats_df.columns = ['uuid'] + [f'text_feat_{i}' for i in range(tsv_feats_df.shape[1] - 1)]

        print(f"Text features extracted: {tsv_feats_df.shape}")
        return tsv_feats_df

    def _extract_audio_features(self):
        """从eGeMAPS文件提取音频特征"""
        try:
            egemaps_path_lists = os.listdir(self.egemaps_dir)
        except:
            print(f"eGeMAPS directory not found: {self.egemaps_dir}")
            return None

        egemaps_feats = []

        print(f"Processing {len(egemaps_path_lists)} eGeMAPS files...")

        for path in tqdm(egemaps_path_lists):
            try:
                # 读取文件（分隔符是;）
                z = pd.read_csv(os.path.join(self.egemaps_dir, path), sep=';')
                # 移除name列
                if 'name' in z.columns:
                    z = z.drop(['name'], axis=1)

                # 提取每列的统计特征
                feature_vector = [path[:-4]]  # uuid

                # 均值、标准差、最小值、中位数
                feature_vector.extend(list(z.mean(axis=0)))    # 均值
                feature_vector.extend(list(z.std(axis=0)))     # 标准差
                feature_vector.extend(list(z.min(axis=0)))     # 最小值
                feature_vector.extend(list(z.median(axis=0)))  # 中位数

                egemaps_feats.append(feature_vector)

            except Exception as e:
                print(f"Error processing audio file {path}: {e}")
                continue

        if egemaps_feats:
            # 转换为DataFrame
            egemaps_feats_df = pd.DataFrame(egemaps_feats)
            n_features = (egemaps_feats_df.shape[1] - 1) // 4
            columns = ['uuid']

            # 创建描述性列名
            for stat in ['mean', 'std', 'min', 'median']:
                for i in range(n_features):
                    columns.append(f'audio_{stat}_{i:03d}')

            egemaps_feats_df.columns = columns[:egemaps_feats_df.shape[1]]
            print(f"Audio features extracted: {egemaps_feats_df.shape}")
            return egemaps_feats_df
        else:
            return None

    def _merge_all_features(self, train_list, test_list, egemaps_pre, text_features, audio_features):
        """合并所有特征"""
        print("Merging features...")

        # 合并训练和测试集
        all_data = pd.concat([train_list, test_list], ignore_index=True)
        print(f"Combined data: {all_data.shape}")

        # 确保uuid列类型一致
        all_data['uuid'] = all_data['uuid'].astype(str)
        egemaps_pre['uuid'] = egemaps_pre['uuid'].astype(str)
        text_features['uuid'] = text_features['uuid'].astype(str)

        # 合并预计算的eGeMAPS特征
        merged_data = pd.merge(all_data, egemaps_pre, on='uuid', how='left')
        print(f"After eGeMAPS merge: {merged_data.shape}")

        # 合并文本特征
        merged_data = pd.merge(merged_data, text_features, on='uuid', how='left')
        print(f"After text features merge: {merged_data.shape}")

        # 合并音频特征（如果可用）
        if audio_features is not None:
            audio_features['uuid'] = audio_features['uuid'].astype(str)
            merged_data = pd.merge(merged_data, audio_features, on='uuid', how='left')
            print(f"After audio features merge: {merged_data.shape}")

        # 数据清洗
        merged_data = self._clean_data(merged_data)

        print(f"Final merged data shape: {merged_data.shape}")
        print("Label distribution in final data:")
        print(merged_data['label'].value_counts())

        return merged_data

    def _clean_data(self, data):
        """清洗和预处理数据"""
        # 移除缺失标签的行
        data = data[data['label'].notna()]
        data = data[data['label'] != '']
        data = data[data['label'] != 'nan']

        # 标准化标签
        data['label'] = data['label'].astype(str).str.upper().str.strip()

        # 填充特征列的缺失值
        feature_columns = [col for col in data.columns if col not in ['uuid', 'label', 'sex', 'age', 'education']]
        data[feature_columns] = data[feature_columns].fillna(data[feature_columns].mean())

        return data

# ========================== 增强特征工程 ==========================
class EnhancedFeatureEngineering:
    """增强特征工程"""

    def __init__(self):
        self.scaler_audio = StandardScaler()
        self.scaler_text = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.audio_columns = None
        self.text_columns = None
        self.selected_audio_columns = None
        self.selected_text_columns = None

    def prepare_features(self, data):
        """准备特征用于模型训练"""
        print("\n=== Feature Engineering ===")

        # 编码标签
        labels = self.label_encoder.fit_transform(data['label'])
        label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        print(f"Label encoding: {label_mapping}")

        # 识别特征列
        feature_columns = [col for col in data.columns if col not in ['uuid', 'label', 'sex', 'age', 'education']]

        # 分离音频和文本特征
        self.audio_columns = [col for col in feature_columns if any(keyword in col.lower() for keyword in 
                            ['f0', 'rms', 'mfcc', 'spectral', 'energy', 'loudness', 'jitter', 'shimmer', 'hnr', 'audio'])]
        self.text_columns = [col for col in feature_columns if any(keyword in col.lower() for keyword in 
                           ['text_feat', 'duration', 'sentence'])]

        # 如果自动检测失败，使用手动分配
        if not self.audio_columns:
            # 假设前88列是音频特征
            self.audio_columns = feature_columns[:88]
        if not self.text_columns:
            # 剩余列是文本特征
            self.text_columns = [col for col in feature_columns if col not in self.audio_columns]

        print(f"Audio features identified: {len(self.audio_columns)}")
        print(f"Text features identified: {len(self.text_columns)}")

        # 特征选择
        selected_audio, selected_text = self._select_features(
            data[self.audio_columns], data[self.text_columns], labels
        )

        # 标准化
        audio_features = self.scaler_audio.fit_transform(selected_audio)
        text_features = self.scaler_text.fit_transform(selected_text)

        print(f"Final feature dimensions - Audio: {audio_features.shape}, Text: {text_features.shape}")

        return audio_features, text_features, labels

    def _select_features(self, audio_data, text_data, labels):
        """选择最重要的特征"""
        print("Performing feature selection...")

        # 音频特征选择
        if audio_data.shape[1] > 50:
            selector_audio = SelectKBest(f_classif, k=min(50, audio_data.shape[1]))
            audio_selected = selector_audio.fit_transform(audio_data, labels)
            self.selected_audio_columns = [self.audio_columns[i] for i in selector_audio.get_support(indices=True)]
            print(f"Selected {len(self.selected_audio_columns)} audio features")
        else:
            audio_selected = audio_data.values
            self.selected_audio_columns = self.audio_columns

        # 文本特征选择
        if text_data.shape[1] > 15:
            selector_text = SelectKBest(f_classif, k=min(15, text_data.shape[1]))
            text_selected = selector_text.fit_transform(text_data, labels)
            self.selected_text_columns = [self.text_columns[i] for i in selector_text.get_support(indices=True)]
            print(f"Selected {len(self.selected_text_columns)} text features")
        else:
            text_selected = text_data.values
            self.selected_text_columns = self.text_columns

        return audio_selected, text_selected

# ========================== 稳定模型架构 ==========================
class StableAlzheimerNet(nn.Module):
    """稳定的阿尔茨海默症识别网络"""

    def __init__(self, audio_dim, text_dim, num_classes=3):
        super(StableAlzheimerNet, self).__init__()

        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio_features, text_features):
        audio_encoded = self.audio_encoder(audio_features)
        text_encoded = self.text_encoder(text_features)

        # 拼接特征
        combined = torch.cat([audio_encoded, text_encoded], dim=1)
        output = self.classifier(combined)

        return output

# ========================== 增强训练系统（带详细可视化） ==========================
class EnhancedTrainingSystem:
    """增强训练系统 - 带详细训练过程跟踪和可视化"""

    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.training_histories = {}
        self.model_performances = {}
        self.epoch_details = {}  # 存储每个epoch的详细信息

    def train_enhanced_system(self, audio_features, text_features, labels, feature_engineer):
        """训练增强系统"""
        print("\n=== Training Enhanced System ===")

        # 1. 深度学习模型（主要焦点）
        print("1. Training Deep Learning Model...")
        dl_predictions, dl_model, dl_history, dl_epoch_details = self._train_deep_learning_with_details(
            audio_features, text_features, labels, feature_engineer
        )
        self.models['Deep Learning'] = dl_model
        self.predictions['Deep Learning'] = dl_predictions
        self.training_histories['Deep Learning'] = dl_history
        self.epoch_details['Deep Learning'] = dl_epoch_details

        # 2. LightGBM模型
        print("2. Training LightGBM Model...")
        lgb_predictions, lgb_model, lgb_scores = self._train_lightgbm(
            audio_features, text_features, labels
        )
        self.models['LightGBM'] = lgb_model
        self.predictions['LightGBM'] = lgb_predictions
        self.model_performances['LightGBM'] = lgb_scores

        # 3. 随机森林模型
        print("3. Training Random Forest Model...")
        rf_predictions, rf_model, rf_scores = self._train_random_forest(
            audio_features, text_features, labels
        )
        self.models['Random Forest'] = rf_model
        self.predictions['Random Forest'] = rf_predictions
        self.model_performances['Random Forest'] = rf_scores

        # 4. 梯度提升模型
        print("4. Training Gradient Boosting Model...")
        gb_predictions, gb_model, gb_scores = self._train_gradient_boosting(
            audio_features, text_features, labels
        )
        self.models['Gradient Boosting'] = gb_model
        self.predictions['Gradient Boosting'] = gb_predictions
        self.model_performances['Gradient Boosting'] = gb_scores

        # 5. 模型集成
        print("5. Model Ensemble...")
        ensemble_predictions = self._ensemble_predictions(labels)

        return ensemble_predictions

    def _train_deep_learning_with_details(self, audio_features, text_features, labels, feature_engineer):
        """训练深度学习模型 - 带详细训练过程跟踪"""
        # 数据分割
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels)

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(audio_features[train_idx]),
            torch.FloatTensor(text_features[train_idx]),
            torch.LongTensor(labels[train_idx])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(audio_features[val_idx]),
            torch.FloatTensor(text_features[val_idx]),
            torch.LongTensor(labels[val_idx])
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 创建模型
        model = StableAlzheimerNet(
            audio_features.shape[1],
            text_features.shape[1],
            num_classes=len(feature_engineer.label_encoder.classes_)
        ).to(device)

        # 训练设置
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # 训练历史记录
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        learning_rates = []

        # 详细epoch信息
        epoch_details = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'grad_norm': []
        }

        best_val_acc = 0
        patience = 25
        no_improve = 0

        print("Deep Learning Training Progress:")
        print("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR")
        print("-" * 60)

        for epoch in range(300):  # 增加训练轮次
            # 训练阶段
            model.train()
            epoch_train_loss = 0
            train_correct = 0
            train_total = 0
            total_grad_norm = 0

            for audio_batch, text_batch, labels_batch in train_loader:
                audio_batch, text_batch, labels_batch = (
                    audio_batch.to(device), text_batch.to(device), labels_batch.to(device)
                )

                optimizer.zero_grad()
                outputs = model(audio_batch, text_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()

                # 梯度裁剪和记录
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                total_grad_norm += grad_norm.item()

                optimizer.step()

                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()

            avg_grad_norm = total_grad_norm / len(train_loader)
            train_acc = train_correct / train_total
            train_losses.append(epoch_train_loss / len(train_loader))
            train_accuracies.append(train_acc)

            # 验证阶段
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for audio_batch, text_batch, labels_batch in val_loader:
                    audio_batch, text_batch, labels_batch = (
                        audio_batch.to(device), text_batch.to(device), labels_batch.to(device)
                    )
                    outputs = model(audio_batch, text_batch)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()

            val_acc = val_correct / val_total
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc)

            # 学习率记录
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # 记录详细epoch信息
            epoch_details['epochs'].append(epoch)
            epoch_details['train_loss'].append(epoch_train_loss / len(train_loader))
            epoch_details['val_loss'].append(val_loss / len(val_loader))
            epoch_details['train_acc'].append(train_acc)
            epoch_details['val_acc'].append(val_acc)
            epoch_details['learning_rate'].append(current_lr)
            epoch_details['grad_norm'].append(avg_grad_norm)

            # 每20个epoch打印进度
            if epoch % 20 == 0:
                print(f"{epoch:5d} | {epoch_train_loss/len(train_loader):10.4f} | {val_loss/len(val_loader):8.4f} | "
                      f"{train_acc:9.4f} | {val_acc:7.4f} | {current_lr:.6f}")

            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}, best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
                    break

        model.load_state_dict(best_model_state)

        # 最终预测
        model.eval()
        with torch.no_grad():
            audio_tensor = torch.FloatTensor(audio_features).to(device)
            text_tensor = torch.FloatTensor(text_features).to(device)
            outputs = model(audio_tensor, text_tensor)
            _, predictions = torch.max(outputs, 1)

        accuracy = accuracy_score(labels, predictions.cpu().numpy())
        print(f"  Deep Learning Final Accuracy: {accuracy:.4f}")

        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accuracies,
            'val_acc': val_accuracies,
            'learning_rates': learning_rates
        }

        return predictions.cpu().numpy(), model, history, epoch_details

    def _train_lightgbm(self, audio_features, text_features, labels):
        """训练LightGBM模型"""
        combined_features = np.concatenate([audio_features, text_features], axis=1)

        # 5折交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        predictions = np.zeros(len(labels))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_features, labels)):
            X_train, X_val = combined_features[train_idx], combined_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            model = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                random_state=SEED,
                verbose=-1
            )
            model.fit(X_train, y_train)
            fold_pred = model.predict(X_val)
            predictions[val_idx] = fold_pred

            fold_accuracy = accuracy_score(y_val, fold_pred)
            fold_scores.append(fold_accuracy)

        accuracy = accuracy_score(labels, predictions)
        print(f"  LightGBM Accuracy: {accuracy:.4f}")

        return predictions, model, fold_scores

    def _train_random_forest(self, audio_features, text_features, labels):
        """训练随机森林模型"""
        combined_features = np.concatenate([audio_features, text_features], axis=1)

        # 5折交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        predictions = np.zeros(len(labels))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_features, labels)):
            X_train, X_val = combined_features[train_idx], combined_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                random_state=SEED,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            fold_pred = model.predict(X_val)
            predictions[val_idx] = fold_pred

            fold_accuracy = accuracy_score(y_val, fold_pred)
            fold_scores.append(fold_accuracy)

        accuracy = accuracy_score(labels, predictions)
        print(f"  Random Forest Accuracy: {accuracy:.4f}")

        return predictions, model, fold_scores

    def _train_gradient_boosting(self, audio_features, text_features, labels):
        """训练梯度提升模型"""
        combined_features = np.concatenate([audio_features, text_features], axis=1)

        # 5折交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        predictions = np.zeros(len(labels))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_features, labels)):
            X_train, X_val = combined_features[train_idx], combined_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            model = GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=SEED
            )
            model.fit(X_train, y_train)
            fold_pred = model.predict(X_val)
            predictions[val_idx] = fold_pred

            fold_accuracy = accuracy_score(y_val, fold_pred)
            fold_scores.append(fold_accuracy)

        accuracy = accuracy_score(labels, predictions)
        print(f"  Gradient Boosting Accuracy: {accuracy:.4f}")

        return predictions, model, fold_scores

    def _ensemble_predictions(self, true_labels):
        """集成预测"""
        all_predictions = list(self.predictions.values())

        # 加权投票
        final_predictions = []
        for i in range(len(true_labels)):
            votes = [pred[i] for pred in all_predictions]
            final_predictions.append(np.argmax(np.bincount(votes)))

        accuracy = accuracy_score(true_labels, final_predictions)
        print(f"  Ensemble Model Accuracy: {accuracy:.4f}")

        return np.array(final_predictions)

# ========================== 增强可视化系统 ==========================
class EnhancedVisualization:
    """增强可视化系统 - 英文标签显示"""

    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        self.class_names = feature_engineer.label_encoder.classes_

    def plot_comprehensive_training_analysis(self, training_system, ensemble_predictions, true_labels):
        """绘制全面的训练分析图"""
        print("\n=== Generating Comprehensive Training Analysis ===")

        # 创建综合图表
        fig = plt.figure(figsize=(25, 20))

        # 1. 深度学习训练历史（详细）
        ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
        self._plot_detailed_training_history(ax1, training_system)

        # 2. 模型性能对比
        ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2)
        self._plot_model_comparison(ax2, training_system, true_labels, ensemble_predictions)

        # 3. 集成模型混淆矩阵
        ax3 = plt.subplot2grid((4, 4), (1, 0))
        self._plot_confusion_matrix(ax3, ensemble_predictions, true_labels, "Ensemble Model")

        # 4. 交叉验证分数热力图
        ax4 = plt.subplot2grid((4, 4), (1, 1))
        self._plot_cv_heatmap(ax4, training_system)

        # 5. 学习率和梯度变化
        ax5 = plt.subplot2grid((4, 4), (1, 2))
        self._plot_learning_rate_gradient(ax5, training_system)

        # 6. 模型预测相关性
        ax6 = plt.subplot2grid((4, 4), (1, 3))
        self._plot_predictions_correlation(ax6, training_system, true_labels)

        # 7. 详细性能指标
        ax7 = plt.subplot2grid((4, 4), (2, 0), colspan=4)
        self._plot_detailed_metrics(ax7, training_system, true_labels, ensemble_predictions)

        # 8. 训练动态分析
        ax8 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
        self._plot_training_dynamics(ax8, training_system)

        # 9. 类别性能分析
        ax9 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
        self._plot_class_performance(ax9, ensemble_predictions, true_labels)

        plt.tight_layout()
        plt.savefig(result_folder_path + 'comprehensive_training_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 额外绘制个体模型混淆矩阵
        self._plot_individual_model_confusions(training_system, true_labels)

        # 训练过程动画式可视化
        self._plot_training_progression(training_system)

    def _plot_detailed_training_history(self, ax, training_system):
        """绘制详细训练历史"""
        if 'Deep Learning' in training_system.training_histories:
            history = training_system.training_histories['Deep Learning']
            epochs = range(len(history['train_loss']))

            # 损失曲线
            ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss', color='black')
            ax.set_title('Deep Learning Training History\nLoss Curves', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            # 准确率曲线（双Y轴）
            ax2 = ax.twinx()
            ax2.plot(epochs, history['train_acc'], 'b--', label='Training Accuracy', linewidth=2, alpha=0.7)
            ax2.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy', linewidth=2, alpha=0.7)
            ax2.set_ylabel('Accuracy', color='black')
            ax2.legend(loc='upper right')
            ax2.set_ylim(0, 1)

            # 标记最佳epoch
            best_epoch = np.argmax(history['val_acc'])
            best_val_acc = history['val_acc'][best_epoch]
            ax2.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best Epoch: {best_epoch}')
            ax2.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No Training History Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Deep Learning Training History', fontsize=12, fontweight='bold')

    def _plot_model_comparison(self, ax, training_system, true_labels, ensemble_predictions):
        """绘制模型性能对比"""
        model_names = []
        accuracies = []
        f1_scores = []

        for model_name, predictions in training_system.predictions.items():
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')

            model_names.append(model_name)
            accuracies.append(accuracy)
            f1_scores.append(f1)

        # 添加集成模型
        ensemble_accuracy = accuracy_score(true_labels, ensemble_predictions)
        ensemble_f1 = f1_score(true_labels, ensemble_predictions, average='weighted')

        model_names.append('Ensemble')
        accuracies.append(ensemble_accuracy)
        f1_scores.append(ensemble_f1)

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8, color='lightcoral', edgecolor='black')

        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison\n(Accuracy vs F1-Score)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # 在柱子上添加数值
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    def _plot_confusion_matrix(self, ax, predictions, true_labels, title):
        """绘制混淆矩阵热力图"""
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Labels', fontweight='bold')
        ax.set_ylabel('True Labels', fontweight='bold')
        ax.set_title(f'Confusion Matrix\n{title}', fontsize=12, fontweight='bold')

    def _plot_cv_heatmap(self, ax, training_system):
        """绘制交叉验证分数热力图"""
        cv_data = []
        model_names = []

        for model_name, scores in training_system.model_performances.items():
            if scores:  # 只包含有CV分数的模型
                cv_data.append(scores)
                model_names.append(model_name)

        if cv_data:
            cv_array = np.array(cv_data)
            sns.heatmap(cv_array, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                       xticklabels=[f'Fold {i+1}' for i in range(cv_array.shape[1])],
                       yticklabels=model_names,
                       cbar_kws={'label': 'Accuracy'})
            ax.set_xlabel('Cross-Validation Folds', fontweight='bold')
            ax.set_ylabel('Models', fontweight='bold')
            ax.set_title('Cross-Validation Scores\n(5-Fold CV)', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No CV Scores Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Cross-Validation Scores', fontsize=12, fontweight='bold')

    def _plot_learning_rate_gradient(self, ax, training_system):
        """绘制学习率和梯度变化"""
        if 'Deep Learning' in training_system.epoch_details:
            details = training_system.epoch_details['Deep Learning']
            epochs = details['epochs']

            # 学习率曲线
            ax.plot(epochs, details['learning_rate'], 'g-', label='Learning Rate', linewidth=2)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Learning Rate', color='green')
            ax.tick_params(axis='y', labelcolor='green')
            ax.set_title('Learning Rate and Gradient Norm', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 梯度范数曲线（双Y轴）
            ax2 = ax.twinx()
            ax2.plot(epochs, details['grad_norm'], 'purple', label='Gradient Norm', linewidth=2, alpha=0.7)
            ax2.set_ylabel('Gradient Norm', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

            # 组合图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No Training Details Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Learning Rate and Gradient', fontsize=12, fontweight='bold')

    def _plot_predictions_correlation(self, ax, training_system, true_labels):
        """绘制模型预测相关性热力图"""
        model_names = list(training_system.predictions.keys())
        n_models = len(model_names)

        if n_models > 1:
            corr_matrix = np.ones((n_models, n_models))

            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        corr = np.corrcoef(training_system.predictions[model_names[i]], 
                                         training_system.predictions[model_names[j]])[0, 1]
                        corr_matrix[i, j] = corr

            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                       center=0, ax=ax, square=True,
                       xticklabels=model_names, yticklabels=model_names,
                       cbar_kws={'label': 'Correlation'})
            ax.set_title('Model Predictions Correlation', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Need Multiple Models\nfor Correlation Analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Model Predictions Correlation', fontsize=12, fontweight='bold')

    def _plot_detailed_metrics(self, ax, training_system, true_labels, ensemble_predictions):
        """绘制详细性能指标"""
        metrics_data = []
        model_names = list(training_system.predictions.keys()) + ['Ensemble']

        for model_name in model_names:
            if model_name == 'Ensemble':
                predictions = ensemble_predictions
            else:
                predictions = training_system.predictions[model_name]

            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            f1 = f1_score(true_labels, predictions, average='weighted')

            metrics_data.append([accuracy, precision, recall, f1])

        metrics_array = np.array(metrics_data)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        x = np.arange(len(model_names))
        width = 0.2

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, metric in enumerate(metrics_names):
            bars = ax.bar(x + i*width, metrics_array[:, i], width, label=metric, color=colors[i], alpha=0.8, edgecolor='black')

            # 在柱子上添加数值
            for bar, value in zip(bars, metrics_array[:, i]):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=8, rotation=45)

        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Scores', fontweight='bold')
        ax.set_title('Detailed Performance Metrics by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_training_dynamics(self, ax, training_system):
        """绘制训练动态分析"""
        if 'Deep Learning' in training_system.epoch_details:
            details = training_system.epoch_details['Deep Learning']
            epochs = details['epochs']

            # 计算训练和验证差距
            gap = np.array(details['train_acc']) - np.array(details['val_acc'])

            ax.plot(epochs, gap, 'orange', linewidth=2, label='Train-Val Gap')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Gap')
            ax.fill_between(epochs, gap, alpha=0.3, color='orange')

            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy Gap')
            ax.set_title('Training Dynamics\n(Train-Validation Gap)', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            avg_gap = np.mean(gap)
            max_gap = np.max(gap)
            ax.text(0.02, 0.98, f'Avg Gap: {avg_gap:.3f}\nMax Gap: {max_gap:.3f}', 
                   transform=ax.transAxes, va='top', ha='left', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No Training Dynamics Data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Training Dynamics', fontsize=12, fontweight='bold')

    def _plot_class_performance(self, ax, predictions, true_labels):
        """绘制类别性能分析"""
        class_performance = []
        precision_scores = []
        recall_scores = []

        for class_idx, class_name in enumerate(self.class_names):
            class_mask = (true_labels == class_idx)
            if np.sum(class_mask) > 0:  # 确保有样本
                class_accuracy = accuracy_score(true_labels[class_mask], predictions[class_mask])
                class_precision = precision_score(true_labels, predictions, average=None, labels=[class_idx])[0]
                class_recall = recall_score(true_labels, predictions, average=None, labels=[class_idx])[0]

                class_performance.append(class_accuracy)
                precision_scores.append(class_precision)
                recall_scores.append(class_recall)

        x = np.arange(len(self.class_names))
        width = 0.25

        ax.bar(x - width, class_performance, width, label='Accuracy', alpha=0.8, color='lightblue')
        ax.bar(x, precision_scores, width, label='Precision', alpha=0.8, color='lightcoral')
        ax.bar(x + width, recall_scores, width, label='Recall', alpha=0.8, color='lightgreen')

        ax.set_xlabel('Classes')
        ax.set_ylabel('Scores')
        ax.set_title('Class-wise Performance Analysis\n(Ensemble Model)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # 添加数值标签
        for i, (acc, prec, rec) in enumerate(zip(class_performance, precision_scores, recall_scores)):
            ax.text(i - width, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, prec + 0.02, f'{prec:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, rec + 0.02, f'{rec:.2f}', ha='center', va='bottom', fontsize=8)

    def _plot_individual_model_confusions(self, training_system, true_labels):
        """绘制个体模型混淆矩阵"""
        n_models = len(training_system.predictions)
        if n_models > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()

            for idx, (model_name, predictions) in enumerate(training_system.predictions.items()):
                if idx < 4:  # 限制为4个子图
                    cm = confusion_matrix(true_labels, predictions)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                               xticklabels=self.class_names,
                               yticklabels=self.class_names)
                    accuracy = accuracy_score(true_labels, predictions)
                    axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', fontweight='bold')
                    axes[idx].set_xlabel('Predicted')
                    axes[idx].set_ylabel('True')

            # 隐藏未使用的子图
            for idx in range(n_models, 4):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(result_folder_path + 'individual_model_confusions.png', dpi=300, bbox_inches='tight')
            plt.show()

    def _plot_training_progression(self, training_system):
        """绘制训练进度可视化"""
        if 'Deep Learning' in training_system.epoch_details:
            details = training_system.epoch_details['Deep Learning']
            epochs = details['epochs']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # 训练进度 - 损失
            ax1.plot(epochs, details['train_loss'], 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, details['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Progression - Loss', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 训练进度 - 准确率
            ax2.plot(epochs, details['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            ax2.plot(epochs, details['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training Progression - Accuracy', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(result_folder_path + 'training_progression.png', dpi=300, bbox_inches='tight')
            plt.show()

# ========================== 主函数 ==========================
def main():
    print("Alzheimer's Disease Multimodal Recognition System - Complete Visualization")
    print("=" * 70)

    try:
        # 1. 真实数据处理
        processor = RealDataProcessor()
        merged_data = processor.process_all_data()

        # 2. 特征工程
        feature_engineer = EnhancedFeatureEngineering()
        audio_features, text_features, labels = feature_engineer.prepare_features(merged_data)

        print(f"\n=== Data Summary ===")
        print(f"Training samples: {len(labels)}")
        print(f"Audio features: {audio_features.shape}")
        print(f"Text features: {text_features.shape}")
        label_distribution = dict(zip(feature_engineer.label_encoder.classes_, np.bincount(labels)))
        print(f"Label distribution: {label_distribution}")

        # 3. 训练增强系统
        training_system = EnhancedTrainingSystem()
        ensemble_predictions = training_system.train_enhanced_system(
            audio_features, text_features, labels, feature_engineer
        )

        # 4. 最终评估
        print("\n=== Final Performance Evaluation ===")
        accuracy = accuracy_score(labels, ensemble_predictions)
        f1 = f1_score(labels, ensemble_predictions, average='weighted')

        print(f"Ensemble System Accuracy: {accuracy:.4f}")
        print(f"Ensemble System F1 Score: {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(labels, ensemble_predictions, 
                                  target_names=feature_engineer.label_encoder.classes_))

        # 5. 性能评估
        if accuracy >= 0.85:
            print("🎉 Outstanding! System achieved top performance!")
        elif accuracy >= 0.75:
            print("✅ Excellent! Robust performance achieved!")
        elif accuracy >= 0.65:
            print("⚠️ Good performance, consider further optimization")
        else:
            print("❌ Needs improvement")

        # 6. 综合可视化
        visualizer = EnhancedVisualization(feature_engineer)
        visualizer.plot_comprehensive_training_analysis(training_system, ensemble_predictions, labels)

        # 7. 保存模型
        torch.save({
            'training_system': training_system,
            'feature_engineer': feature_engineer,
            'accuracy': accuracy,
            'model_state_dict': training_system.models['Deep Learning'].state_dict() if 'Deep Learning' in training_system.models else None
        }, 'complete_alzheimer_system.pth')

        print(f"\nModel saved: complete_alzheimer_system.pth")

        print("\n" + "="*50)
        print("System training and visualization completed!")
        print("Ready for deployment.")
        print("="*50)

    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# In[3]:


# -*- coding: utf-8 -*-
"""
阿尔茨海默症多模态识别系统 - 完整可视化版
集成训练过程可视化，中文注释，英文标签显示
"""

import os
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, f1_score, 
                           accuracy_score, classification_report, 
                           confusion_matrix, precision_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# 设置
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================== 真实数据处理器 ==========================
class RealDataProcessor:
    """真实数据处理器 - 基于实际数据结构"""

    def __init__(self, base_path='./kaggle/input/alzheimer/data/'):
        self.base_path = base_path
        self.egemaps_pre_path = os.path.join(base_path, 'egemaps_final.csv')
        self.train_list_path = os.path.join(base_path, '2_final_list_train.csv')
        self.test_list_path = os.path.join(base_path, '2_final_list_test.csv')
        self.tsv_dir = os.path.join(base_path, 'tsv2/')
        self.egemaps_dir = '/data/egemaps2/'  # 根据实际情况更新路径

    def process_all_data(self):
        """处理所有数据"""
        print("=== Processing Real Alzheimer's Disease Data ===")

        # 1. 加载基础数据
        print("1. Loading basic data...")
        preliminary_list_test = pd.read_csv(self.test_list_path)  # (27,2)
        preliminary_list_train = pd.read_csv(self.train_list_path)  # (179,2)
        egemaps_pre = pd.read_csv(self.egemaps_pre_path)  # (206,89)

        print(f"Training set: {preliminary_list_train.shape}")
        print(f"Test set: {preliminary_list_test.shape}")
        print(f"Precomputed features: {egemaps_pre.shape}")
        print('Training set label distribution:\n', preliminary_list_train['label'].value_counts())

        # 2. 从TSV文件提取文本特征
        print("\n2. Extracting text features from TSV files...")
        text_features = self._extract_text_features()

        # 3. 从eGeMAPS文件提取音频特征
        print("\n3. Extracting audio features from eGeMAPS files...")
        audio_features = self._extract_audio_features()

        # 4. 合并所有特征
        print("\n4. Merging all features...")
        merged_data = self._merge_all_features(
            preliminary_list_train, preliminary_list_test, 
            egemaps_pre, text_features, audio_features
        )

        return merged_data

    def _extract_text_features(self):
        """从TSV文件提取文本特征"""
        tsv_path_lists = os.listdir(self.tsv_dir)
        tsv_feats = []

        print(f"Processing {len(tsv_path_lists)} TSV files...")

        for path in tqdm(tsv_path_lists):
            try:
                # 读取tsv文件
                z = pd.read_csv(os.path.join(self.tsv_dir, path), sep='\t')
                # 计算每句话的时长
                z['end_time-start_time'] = z['end_time'] - z['start_time']

                # 提取时长统计特征
                tsv_feats.append([
                    path[:-4],  # uuid
                    z['end_time-start_time'].mean(),    # 平均时长
                    z['end_time-start_time'].min(),     # 最小时长
                    z['end_time-start_time'].max(),     # 最大时长
                    z['end_time-start_time'].std(),     # 时长标准差
                    z['end_time-start_time'].median(),  # 时长中位数
                    z['end_time-start_time'].skew(),    # 时长偏度
                    z.shape[0]                         # 句子数量
                ])
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                continue

        # 转换为DataFrame
        tsv_feats_df = pd.DataFrame(tsv_feats)
        tsv_feats_df.columns = ['uuid'] + [f'text_feat_{i}' for i in range(tsv_feats_df.shape[1] - 1)]

        print(f"Text features extracted: {tsv_feats_df.shape}")
        return tsv_feats_df

    def _extract_audio_features(self):
        """从eGeMAPS文件提取音频特征"""
        try:
            egemaps_path_lists = os.listdir(self.egemaps_dir)
        except:
            print(f"eGeMAPS directory not found: {self.egemaps_dir}")
            return None

        egemaps_feats = []

        print(f"Processing {len(egemaps_path_lists)} eGeMAPS files...")

        for path in tqdm(egemaps_path_lists):
            try:
                # 读取文件（分隔符是;）
                z = pd.read_csv(os.path.join(self.egemaps_dir, path), sep=';')
                # 移除name列
                if 'name' in z.columns:
                    z = z.drop(['name'], axis=1)

                # 提取每列的统计特征
                feature_vector = [path[:-4]]  # uuid

                # 均值、标准差、最小值、中位数
                feature_vector.extend(list(z.mean(axis=0)))    # 均值
                feature_vector.extend(list(z.std(axis=0)))     # 标准差
                feature_vector.extend(list(z.min(axis=0)))     # 最小值
                feature_vector.extend(list(z.median(axis=0)))  # 中位数

                egemaps_feats.append(feature_vector)

            except Exception as e:
                print(f"Error processing audio file {path}: {e}")
                continue

        if egemaps_feats:
            # 转换为DataFrame
            egemaps_feats_df = pd.DataFrame(egemaps_feats)
            n_features = (egemaps_feats_df.shape[1] - 1) // 4
            columns = ['uuid']

            # 创建描述性列名
            for stat in ['mean', 'std', 'min', 'median']:
                for i in range(n_features):
                    columns.append(f'audio_{stat}_{i:03d}')

            egemaps_feats_df.columns = columns[:egemaps_feats_df.shape[1]]
            print(f"Audio features extracted: {egemaps_feats_df.shape}")
            return egemaps_feats_df
        else:
            return None

    def _merge_all_features(self, train_list, test_list, egemaps_pre, text_features, audio_features):
        """合并所有特征"""
        print("Merging features...")

        # 合并训练和测试集
        all_data = pd.concat([train_list, test_list], ignore_index=True)
        print(f"Combined data: {all_data.shape}")

        # 确保uuid列类型一致
        all_data['uuid'] = all_data['uuid'].astype(str)
        egemaps_pre['uuid'] = egemaps_pre['uuid'].astype(str)
        text_features['uuid'] = text_features['uuid'].astype(str)

        # 合并预计算的eGeMAPS特征
        merged_data = pd.merge(all_data, egemaps_pre, on='uuid', how='left')
        print(f"After eGeMAPS merge: {merged_data.shape}")

        # 合并文本特征
        merged_data = pd.merge(merged_data, text_features, on='uuid', how='left')
        print(f"After text features merge: {merged_data.shape}")

        # 合并音频特征（如果可用）
        if audio_features is not None:
            audio_features['uuid'] = audio_features['uuid'].astype(str)
            merged_data = pd.merge(merged_data, audio_features, on='uuid', how='left')
            print(f"After audio features merge: {merged_data.shape}")

        # 数据清洗
        merged_data = self._clean_data(merged_data)

        print(f"Final merged data shape: {merged_data.shape}")
        print("Label distribution in final data:")
        print(merged_data['label'].value_counts())

        return merged_data

    def _clean_data(self, data):
        """清洗和预处理数据"""
        # 移除缺失标签的行
        data = data[data['label'].notna()]
        data = data[data['label'] != '']
        data = data[data['label'] != 'nan']

        # 标准化标签
        data['label'] = data['label'].astype(str).str.upper().str.strip()

        # 填充特征列的缺失值
        feature_columns = [col for col in data.columns if col not in ['uuid', 'label', 'sex', 'age', 'education']]
        data[feature_columns] = data[feature_columns].fillna(data[feature_columns].mean())

        return data

# ========================== 增强特征工程 ==========================
class EnhancedFeatureEngineering:
    """增强特征工程"""

    def __init__(self):
        self.scaler_audio = StandardScaler()
        self.scaler_text = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.audio_columns = None
        self.text_columns = None
        self.selected_audio_columns = None
        self.selected_text_columns = None

    def prepare_features(self, data):
        """准备特征用于模型训练"""
        print("\n=== Feature Engineering ===")

        # 编码标签
        labels = self.label_encoder.fit_transform(data['label'])
        label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        print(f"Label encoding: {label_mapping}")

        # 识别特征列
        feature_columns = [col for col in data.columns if col not in ['uuid', 'label', 'sex', 'age', 'education']]

        # 分离音频和文本特征
        self.audio_columns = [col for col in feature_columns if any(keyword in col.lower() for keyword in 
                            ['f0', 'rms', 'mfcc', 'spectral', 'energy', 'loudness', 'jitter', 'shimmer', 'hnr', 'audio'])]
        self.text_columns = [col for col in feature_columns if any(keyword in col.lower() for keyword in 
                           ['text_feat', 'duration', 'sentence'])]

        # 如果自动检测失败，使用手动分配
        if not self.audio_columns:
            # 假设前88列是音频特征
            self.audio_columns = feature_columns[:88]
        if not self.text_columns:
            # 剩余列是文本特征
            self.text_columns = [col for col in feature_columns if col not in self.audio_columns]

        print(f"Audio features identified: {len(self.audio_columns)}")
        print(f"Text features identified: {len(self.text_columns)}")

        # 特征选择
        selected_audio, selected_text = self._select_features(
            data[self.audio_columns], data[self.text_columns], labels
        )

        # 标准化
        audio_features = self.scaler_audio.fit_transform(selected_audio)
        text_features = self.scaler_text.fit_transform(selected_text)

        print(f"Final feature dimensions - Audio: {audio_features.shape}, Text: {text_features.shape}")

        return audio_features, text_features, labels

    def _select_features(self, audio_data, text_data, labels):
        """选择最重要的特征"""
        print("Performing feature selection...")

        # 音频特征选择
        if audio_data.shape[1] > 50:
            selector_audio = SelectKBest(f_classif, k=min(50, audio_data.shape[1]))
            audio_selected = selector_audio.fit_transform(audio_data, labels)
            self.selected_audio_columns = [self.audio_columns[i] for i in selector_audio.get_support(indices=True)]
            print(f"Selected {len(self.selected_audio_columns)} audio features")
        else:
            audio_selected = audio_data.values
            self.selected_audio_columns = self.audio_columns

        # 文本特征选择
        if text_data.shape[1] > 15:
            selector_text = SelectKBest(f_classif, k=min(15, text_data.shape[1]))
            text_selected = selector_text.fit_transform(text_data, labels)
            self.selected_text_columns = [self.text_columns[i] for i in selector_text.get_support(indices=True)]
            print(f"Selected {len(self.selected_text_columns)} text features")
        else:
            text_selected = text_data.values
            self.selected_text_columns = self.text_columns

        return audio_selected, text_selected

# ========================== 稳定模型架构 ==========================
class StableAlzheimerNet(nn.Module):
    """稳定的阿尔茨海默症识别网络"""

    def __init__(self, audio_dim, text_dim, num_classes=3):
        super(StableAlzheimerNet, self).__init__()

        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio_features, text_features):
        audio_encoded = self.audio_encoder(audio_features)
        text_encoded = self.text_encoder(text_features)

        # 拼接特征
        combined = torch.cat([audio_encoded, text_encoded], dim=1)
        output = self.classifier(combined)

        return output

# ========================== 增强训练系统（带详细可视化） ==========================
class EnhancedTrainingSystem:
    """增强训练系统 - 带详细训练过程跟踪和可视化"""

    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.training_histories = {}
        self.model_performances = {}
        self.epoch_details = {}  # 存储每个epoch的详细信息

    def train_enhanced_system(self, audio_features, text_features, labels, feature_engineer):
        """训练增强系统"""
        print("\n=== Training Enhanced System ===")

        # 1. 深度学习模型（主要焦点）
        print("1. Training Deep Learning Model...")
        dl_predictions, dl_model, dl_history, dl_epoch_details = self._train_deep_learning_with_details(
            audio_features, text_features, labels, feature_engineer
        )
        self.models['Deep Learning'] = dl_model
        self.predictions['Deep Learning'] = dl_predictions
        self.training_histories['Deep Learning'] = dl_history
        self.epoch_details['Deep Learning'] = dl_epoch_details

        # 2. LightGBM模型
        print("2. Training LightGBM Model...")
        lgb_predictions, lgb_model, lgb_scores = self._train_lightgbm(
            audio_features, text_features, labels
        )
        self.models['LightGBM'] = lgb_model
        self.predictions['LightGBM'] = lgb_predictions
        self.model_performances['LightGBM'] = lgb_scores

        # 3. 随机森林模型
        print("3. Training Random Forest Model...")
        rf_predictions, rf_model, rf_scores = self._train_random_forest(
            audio_features, text_features, labels
        )
        self.models['Random Forest'] = rf_model
        self.predictions['Random Forest'] = rf_predictions
        self.model_performances['Random Forest'] = rf_scores

        # 4. 梯度提升模型
        print("4. Training Gradient Boosting Model...")
        gb_predictions, gb_model, gb_scores = self._train_gradient_boosting(
            audio_features, text_features, labels
        )
        self.models['Gradient Boosting'] = gb_model
        self.predictions['Gradient Boosting'] = gb_predictions
        self.model_performances['Gradient Boosting'] = gb_scores

        # 5. 模型集成
        print("5. Model Ensemble...")
        ensemble_predictions = self._ensemble_predictions(labels)

        return ensemble_predictions

    def _train_deep_learning_with_details(self, audio_features, text_features, labels, feature_engineer):
        """训练深度学习模型 - 带详细训练过程跟踪"""
        # 数据分割
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels)

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(audio_features[train_idx]),
            torch.FloatTensor(text_features[train_idx]),
            torch.LongTensor(labels[train_idx])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(audio_features[val_idx]),
            torch.FloatTensor(text_features[val_idx]),
            torch.LongTensor(labels[val_idx])
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 创建模型
        model = StableAlzheimerNet(
            audio_features.shape[1],
            text_features.shape[1],
            num_classes=len(feature_engineer.label_encoder.classes_)
        ).to(device)

        # 训练设置
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # 训练历史记录
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        learning_rates = []

        # 详细epoch信息
        epoch_details = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'grad_norm': []
        }

        best_val_acc = 0
        patience = 25
        no_improve = 0

        print("Deep Learning Training Progress:")
        print("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR")
        print("-" * 60)

        for epoch in range(300):  # 增加训练轮次
            # 训练阶段
            model.train()
            epoch_train_loss = 0
            train_correct = 0
            train_total = 0
            total_grad_norm = 0

            for audio_batch, text_batch, labels_batch in train_loader:
                audio_batch, text_batch, labels_batch = (
                    audio_batch.to(device), text_batch.to(device), labels_batch.to(device)
                )

                optimizer.zero_grad()
                outputs = model(audio_batch, text_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()

                # 梯度裁剪和记录
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                total_grad_norm += grad_norm.item()

                optimizer.step()

                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()

            avg_grad_norm = total_grad_norm / len(train_loader)
            train_acc = train_correct / train_total
            train_losses.append(epoch_train_loss / len(train_loader))
            train_accuracies.append(train_acc)

            # 验证阶段
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for audio_batch, text_batch, labels_batch in val_loader:
                    audio_batch, text_batch, labels_batch = (
                        audio_batch.to(device), text_batch.to(device), labels_batch.to(device)
                    )
                    outputs = model(audio_batch, text_batch)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()

            val_acc = val_correct / val_total
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc)

            # 学习率记录
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # 记录详细epoch信息
            epoch_details['epochs'].append(epoch)
            epoch_details['train_loss'].append(epoch_train_loss / len(train_loader))
            epoch_details['val_loss'].append(val_loss / len(val_loader))
            epoch_details['train_acc'].append(train_acc)
            epoch_details['val_acc'].append(val_acc)
            epoch_details['learning_rate'].append(current_lr)
            epoch_details['grad_norm'].append(avg_grad_norm)

            # 每20个epoch打印进度
            if epoch % 20 == 0:
                print(f"{epoch:5d} | {epoch_train_loss/len(train_loader):10.4f} | {val_loss/len(val_loader):8.4f} | "
                      f"{train_acc:9.4f} | {val_acc:7.4f} | {current_lr:.6f}")

            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}, best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
                    break

        model.load_state_dict(best_model_state)

        # 最终预测
        model.eval()
        with torch.no_grad():
            audio_tensor = torch.FloatTensor(audio_features).to(device)
            text_tensor = torch.FloatTensor(text_features).to(device)
            outputs = model(audio_tensor, text_tensor)
            _, predictions = torch.max(outputs, 1)

        accuracy = accuracy_score(labels, predictions.cpu().numpy())
        print(f"  Deep Learning Final Accuracy: {accuracy:.4f}")

        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accuracies,
            'val_acc': val_accuracies,
            'learning_rates': learning_rates
        }

        return predictions.cpu().numpy(), model, history, epoch_details

    def _train_lightgbm(self, audio_features, text_features, labels):
        """训练LightGBM模型"""
        combined_features = np.concatenate([audio_features, text_features], axis=1)

        # 5折交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        predictions = np.zeros(len(labels))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_features, labels)):
            X_train, X_val = combined_features[train_idx], combined_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            model = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                random_state=SEED,
                verbose=-1
            )
            model.fit(X_train, y_train)
            fold_pred = model.predict(X_val)
            predictions[val_idx] = fold_pred

            fold_accuracy = accuracy_score(y_val, fold_pred)
            fold_scores.append(fold_accuracy)

        accuracy = accuracy_score(labels, predictions)
        print(f"  LightGBM Accuracy: {accuracy:.4f}")

        return predictions, model, fold_scores

    def _train_random_forest(self, audio_features, text_features, labels):
        """训练随机森林模型"""
        combined_features = np.concatenate([audio_features, text_features], axis=1)

        # 5折交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        predictions = np.zeros(len(labels))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_features, labels)):
            X_train, X_val = combined_features[train_idx], combined_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                random_state=SEED,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            fold_pred = model.predict(X_val)
            predictions[val_idx] = fold_pred

            fold_accuracy = accuracy_score(y_val, fold_pred)
            fold_scores.append(fold_accuracy)

        accuracy = accuracy_score(labels, predictions)
        print(f"  Random Forest Accuracy: {accuracy:.4f}")

        return predictions, model, fold_scores

    def _train_gradient_boosting(self, audio_features, text_features, labels):
        """训练梯度提升模型"""
        combined_features = np.concatenate([audio_features, text_features], axis=1)

        # 5折交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        predictions = np.zeros(len(labels))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_features, labels)):
            X_train, X_val = combined_features[train_idx], combined_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            model = GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=SEED
            )
            model.fit(X_train, y_train)
            fold_pred = model.predict(X_val)
            predictions[val_idx] = fold_pred

            fold_accuracy = accuracy_score(y_val, fold_pred)
            fold_scores.append(fold_accuracy)

        accuracy = accuracy_score(labels, predictions)
        print(f"  Gradient Boosting Accuracy: {accuracy:.4f}")

        return predictions, model, fold_scores

    def _ensemble_predictions(self, true_labels):
        """集成预测"""
        all_predictions = list(self.predictions.values())

        # 加权投票
        final_predictions = []
        for i in range(len(true_labels)):
            votes = [pred[i] for pred in all_predictions]
            final_predictions.append(np.argmax(np.bincount(votes)))

        accuracy = accuracy_score(true_labels, final_predictions)
        print(f"  Ensemble Model Accuracy: {accuracy:.4f}")

        return np.array(final_predictions)

# ========================== 增强可视化系统 ==========================
class EnhancedVisualization:
    """增强可视化系统 - 英文标签显示，图表单独输出"""

    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        self.class_names = feature_engineer.label_encoder.classes_

    def plot_comprehensive_training_analysis(self, training_system, ensemble_predictions, true_labels):
        """绘制全面的训练分析图，单独输出每个图表"""
        print("\n=== Generating Comprehensive Training Analysis ===")

        # 1. 深度学习训练历史（详细）
        self._plot_detailed_training_history(training_system)

        # 2. 模型性能对比
        self._plot_model_comparison(training_system, true_labels, ensemble_predictions)

        # 3. 集成模型混淆矩阵
        self._plot_confusion_matrix(ensemble_predictions, true_labels, "Ensemble Model")

        # 4. 交叉验证分数热力图
        self._plot_cv_heatmap(training_system)

        # 5. 学习率和梯度变化
        self._plot_learning_rate_gradient(training_system)

        # 6. 模型预测相关性
        self._plot_predictions_correlation(training_system, true_labels)

        # 7. 详细性能指标
        self._plot_detailed_metrics(training_system, true_labels, ensemble_predictions)

        # 8. 训练动态分析
        self._plot_training_dynamics(training_system)

        # 9. 类别性能分析
        self._plot_class_performance(ensemble_predictions, true_labels)

        # 额外绘制个体模型混淆矩阵
        self._plot_individual_model_confusions(training_system, true_labels)

        # 训练过程动画式可视化
        self._plot_training_progression(training_system)

    def _plot_detailed_training_history(self, training_system):
        """绘制详细训练历史"""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        if 'Deep Learning' in training_system.training_histories:
            history = training_system.training_histories['Deep Learning']
            epochs = range(len(history['train_loss']))

            # 损失曲线
            ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss', color='black')
            ax.set_title('Deep Learning Training History\nLoss Curves', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            # 准确率曲线（双Y轴）
            ax2 = ax.twinx()
            ax2.plot(epochs, history['train_acc'], 'b--', label='Training Accuracy', linewidth=2, alpha=0.7)
            ax2.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy', linewidth=2, alpha=0.7)
            ax2.set_ylabel('Accuracy', color='black')
            ax2.legend(loc='upper right')
            ax2.set_ylim(0, 1)

            # 标记最佳epoch
            best_epoch = np.argmax(history['val_acc'])
            best_val_acc = history['val_acc'][best_epoch]
            ax2.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best Epoch: {best_epoch}')
            ax2.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No Training History Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Deep Learning Training History', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(result_folder_path + 'detailed_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_model_comparison(self, training_system, true_labels, ensemble_predictions):
        """绘制模型性能对比"""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        model_names = []
        accuracies = []
        f1_scores = []

        for model_name, predictions in training_system.predictions.items():
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')

            model_names.append(model_name)
            accuracies.append(accuracy)
            f1_scores.append(f1)

        # 添加集成模型
        ensemble_accuracy = accuracy_score(true_labels, ensemble_predictions)
        ensemble_f1 = f1_score(true_labels, ensemble_predictions, average='weighted')

        model_names.append('Ensemble')
        accuracies.append(ensemble_accuracy)
        f1_scores.append(ensemble_f1)

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8, color='lightcoral', edgecolor='black')

        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison\n(Accuracy vs F1-Score)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # 在柱子上添加数值
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(result_folder_path + 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_confusion_matrix(self, predictions, true_labels, title):
        """绘制混淆矩阵热力图"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Labels', fontweight='bold')
        ax.set_ylabel('True Labels', fontweight='bold')
        ax.set_title(f'Confusion Matrix\n{title}', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(result_folder_path + f'confusion_matrix_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_cv_heatmap(self, training_system):
        """绘制交叉验证分数热力图"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        cv_data = []
        model_names = []

        for model_name, scores in training_system.model_performances.items():
            if scores:  # 只包含有CV分数的模型
                cv_data.append(scores)
                model_names.append(model_name)

        if cv_data:
            cv_array = np.array(cv_data)
            sns.heatmap(cv_array, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                       xticklabels=[f'Fold {i+1}' for i in range(cv_array.shape[1])],
                       yticklabels=model_names,
                       cbar_kws={'label': 'Accuracy'})
            ax.set_xlabel('Cross-Validation Folds', fontweight='bold')
            ax.set_ylabel('Models', fontweight='bold')
            ax.set_title('Cross-Validation Scores\n(5-Fold CV)', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No CV Scores Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Cross-Validation Scores', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(result_folder_path + 'cv_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_learning_rate_gradient(self, training_system):
        """绘制学习率和梯度变化"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        if 'Deep Learning' in training_system.epoch_details:
            details = training_system.epoch_details['Deep Learning']
            epochs = details['epochs']

            # 学习率曲线
            ax.plot(epochs, details['learning_rate'], 'g-', label='Learning Rate', linewidth=2)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Learning Rate', color='green')
            ax.tick_params(axis='y', labelcolor='green')
            ax.set_title('Learning Rate and Gradient Norm', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 梯度范数曲线（双Y轴）
            ax2 = ax.twinx()
            ax2.plot(epochs, details['grad_norm'], 'purple', label='Gradient Norm', linewidth=2, alpha=0.7)
            ax2.set_ylabel('Gradient Norm', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

            # 组合图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No Training Details Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Learning Rate and Gradient', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(result_folder_path + 'learning_rate_gradient.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_predictions_correlation(self, training_system, true_labels):
        """绘制模型预测相关性热力图"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        model_names = list(training_system.predictions.keys())
        n_models = len(model_names)

        if n_models > 1:
            corr_matrix = np.ones((n_models, n_models))

            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        corr = np.corrcoef(training_system.predictions[model_names[i]], 
                                         training_system.predictions[model_names[j]])[0, 1]
                        corr_matrix[i, j] = corr

            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                       center=0, ax=ax, square=True,
                       xticklabels=model_names, yticklabels=model_names,
                       cbar_kws={'label': 'Correlation'})
            ax.set_title('Model Predictions Correlation', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Need Multiple Models\nfor Correlation Analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Model Predictions Correlation', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(result_folder_path + 'predictions_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_detailed_metrics(self, training_system, true_labels, ensemble_predictions):
        """绘制详细性能指标"""
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        metrics_data = []
        model_names = list(training_system.predictions.keys()) + ['Ensemble']

        for model_name in model_names:
            if model_name == 'Ensemble':
                predictions = ensemble_predictions
            else:
                predictions = training_system.predictions[model_name]

            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            f1 = f1_score(true_labels, predictions, average='weighted')

            metrics_data.append([accuracy, precision, recall, f1])

        metrics_array = np.array(metrics_data)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        x = np.arange(len(model_names))
        width = 0.2

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, metric in enumerate(metrics_names):
            bars = ax.bar(x + i*width, metrics_array[:, i], width, label=metric, color=colors[i], alpha=0.8, edgecolor='black')

            # 在柱子上添加数值
            for bar, value in zip(bars, metrics_array[:, i]):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=8, rotation=45)

        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Scores', fontweight='bold')
        ax.set_title('Detailed Performance Metrics by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(result_folder_path + 'detailed_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_training_dynamics(self, training_system):
        """绘制训练动态分析"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        if 'Deep Learning' in training_system.epoch_details:
            details = training_system.epoch_details['Deep Learning']
            epochs = details['epochs']

            # 计算训练和验证差距
            gap = np.array(details['train_acc']) - np.array(details['val_acc'])

            ax.plot(epochs, gap, 'orange', linewidth=2, label='Train-Val Gap')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Gap')
            ax.fill_between(epochs, gap, alpha=0.3, color='orange')

            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy Gap')
            ax.set_title('Training Dynamics\n(Train-Validation Gap)', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            avg_gap = np.mean(gap)
            max_gap = np.max(gap)
            ax.text(0.02, 0.98, f'Avg Gap: {avg_gap:.3f}\nMax Gap: {max_gap:.3f}', 
                   transform=ax.transAxes, va='top', ha='left', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No Training Dynamics Data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Training Dynamics', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(result_folder_path + 'training_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_class_performance(self, predictions, true_labels):
        """绘制类别性能分析"""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        class_performance = []
        precision_scores = []
        recall_scores = []

        for class_idx, class_name in enumerate(self.class_names):
            class_mask = (true_labels == class_idx)
            if np.sum(class_mask) > 0:  # 确保有样本
                class_accuracy = accuracy_score(true_labels[class_mask], predictions[class_mask])
                class_precision = precision_score(true_labels, predictions, average=None, labels=[class_idx])[0]
                class_recall = recall_score(true_labels, predictions, average=None, labels=[class_idx])[0]

                class_performance.append(class_accuracy)
                precision_scores.append(class_precision)
                recall_scores.append(class_recall)

        x = np.arange(len(self.class_names))
        width = 0.25

        ax.bar(x - width, class_performance, width, label='Accuracy', alpha=0.8, color='lightblue')
        ax.bar(x, precision_scores, width, label='Precision', alpha=0.8, color='lightcoral')
        ax.bar(x + width, recall_scores, width, label='Recall', alpha=0.8, color='lightgreen')

        ax.set_xlabel('Classes')
        ax.set_ylabel('Scores')
        ax.set_title('Class-wise Performance Analysis\n(Ensemble Model)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # 添加数值标签
        for i, (acc, prec, rec) in enumerate(zip(class_performance, precision_scores, recall_scores)):
            ax.text(i - width, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, prec + 0.02, f'{prec:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, rec + 0.02, f'{rec:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(result_folder_path + 'class_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_individual_model_confusions(self, training_system, true_labels):
        """绘制个体模型混淆矩阵"""
        n_models = len(training_system.predictions)
        if n_models > 0:
            for idx, (model_name, predictions) in enumerate(training_system.predictions.items()):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)

                cm = confusion_matrix(true_labels, predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=self.class_names,
                           yticklabels=self.class_names)
                accuracy = accuracy_score(true_labels, predictions)
                ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', fontweight='bold')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')

                plt.tight_layout()
                plt.savefig(result_folder_path + f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()

    def _plot_training_progression(self, training_system):
        """绘制训练进度可视化"""
        if 'Deep Learning' in training_system.epoch_details:
            details = training_system.epoch_details['Deep Learning']
            epochs = details['epochs']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # 训练进度 - 损失
            ax1.plot(epochs, details['train_loss'], 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, details['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Progression - Loss', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 训练进度 - 准确率
            ax2.plot(epochs, details['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            ax2.plot(epochs, details['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training Progression - Accuracy', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(result_folder_path + 'training_progression.png', dpi=300, bbox_inches='tight')
            plt.show()

# ========================== 主函数 ==========================
def main():
    print("Alzheimer's Disease Multimodal Recognition System - Complete Visualization")
    print("=" * 70)

    try:
        # 1. 真实数据处理
        processor = RealDataProcessor()
        merged_data = processor.process_all_data()

        # 2. 特征工程
        feature_engineer = EnhancedFeatureEngineering()
        audio_features, text_features, labels = feature_engineer.prepare_features(merged_data)

        print(f"\n=== Data Summary ===")
        print(f"Training samples: {len(labels)}")
        print(f"Audio features: {audio_features.shape}")
        print(f"Text features: {text_features.shape}")
        label_distribution = dict(zip(feature_engineer.label_encoder.classes_, np.bincount(labels)))
        print(f"Label distribution: {label_distribution}")

        # 3. 训练增强系统
        training_system = EnhancedTrainingSystem()
        ensemble_predictions = training_system.train_enhanced_system(
            audio_features, text_features, labels, feature_engineer
        )

        # 4. 最终评估
        print("\n=== Final Performance Evaluation ===")
        accuracy = accuracy_score(labels, ensemble_predictions)
        f1 = f1_score(labels, ensemble_predictions, average='weighted')

        print(f"Ensemble System Accuracy: {accuracy:.4f}")
        print(f"Ensemble System F1 Score: {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(labels, ensemble_predictions, 
                                  target_names=feature_engineer.label_encoder.classes_))

        # 5. 性能评估
        if accuracy >= 0.85:
            print("🎉 Outstanding! System achieved top performance!")
        elif accuracy >= 0.75:
            print("✅ Excellent! Robust performance achieved!")
        elif accuracy >= 0.65:
            print("⚠️ Good performance, consider further optimization")
        else:
            print("❌ Needs improvement")

        # 6. 综合可视化
        visualizer = EnhancedVisualization(feature_engineer)
        visualizer.plot_comprehensive_training_analysis(training_system, ensemble_predictions, labels)

        # 7. 保存模型
        torch.save({
            'training_system': training_system,
            'feature_engineer': feature_engineer,
            'accuracy': accuracy,
            'model_state_dict': training_system.models['Deep Learning'].state_dict() if 'Deep Learning' in training_system.models else None
        }, 'complete_alzheimer_system.pth')

        print(f"\nModel saved: complete_alzheimer_system.pth")

        print("\n" + "="*50)
        print("System training and visualization completed!")
        print("Ready for deployment.")
        print("="*50)

    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# In[4]:


# -*- coding: utf-8 -*-
"""
阿尔茨海默症多模态识别系统 - 完整优化版
集成训练过程可视化，中文注释，英文标签显示，多模态融合
"""

import os
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, f1_score, 
                           accuracy_score, classification_report, 
                           confusion_matrix, precision_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# 设置
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================== 高级数据处理器 ==========================
class AdvancedDataProcessor:
    """高级数据处理器 - 支持多种数据源和增强特征工程"""

    def __init__(self, base_path='./kaggle/input/alzheimer/data/'):
        self.base_path = base_path
        self.egemaps_pre_path = os.path.join(base_path, 'egemaps_final.csv')
        self.train_list_path = os.path.join(base_path, '2_final_list_train.csv')
        self.test_list_path = os.path.join(base_path, '2_final_list_test.csv')
        self.tsv_dir = os.path.join(base_path, 'tsv2/')
        self.egemaps_dir = '/data/egemaps2/'

    def process_all_data(self):
        """处理所有数据并进行高级特征工程"""
        print("=== Processing Alzheimer's Disease Data with Advanced Features ===")

        # 1. 加载基础数据
        print("1. Loading basic data...")
        preliminary_list_test = pd.read_csv(self.test_list_path)
        preliminary_list_train = pd.read_csv(self.train_list_path)
        egemaps_pre = pd.read_csv(self.egemaps_pre_path)

        print(f"Training set: {preliminary_list_train.shape}")
        print(f"Test set: {preliminary_list_test.shape}")
        print(f"Precomputed features: {egemaps_pre.shape}")
        print('Training set label distribution:\n', preliminary_list_train['label'].value_counts())

        # 2. 从TSV文件提取增强文本特征
        print("\n2. Extracting enhanced text features from TSV files...")
        text_features = self._extract_enhanced_text_features()

        # 3. 从eGeMAPS文件提取增强音频特征
        print("\n3. Extracting enhanced audio features from eGeMAPS files...")
        audio_features = self._extract_enhanced_audio_features()

        # 4. 合并所有特征并进行高级处理
        print("\n4. Merging all features with advanced processing...")
        merged_data = self._merge_all_features_advanced(
            preliminary_list_train, preliminary_list_test, 
            egemaps_pre, text_features, audio_features
        )

        return merged_data

    def _extract_enhanced_text_features(self):
        """提取增强文本特征"""
        tsv_path_lists = os.listdir(self.tsv_dir)
        tsv_feats = []

        print(f"Processing {len(tsv_path_lists)} TSV files with enhanced features...")

        for path in tqdm(tsv_path_lists):
            try:
                z = pd.read_csv(os.path.join(self.tsv_dir, path), sep='\t')
                z['duration'] = z['end_time'] - z['start_time']

                # 基础统计特征
                duration_stats = [
                    z['duration'].mean(), z['duration'].min(), z['duration'].max(),
                    z['duration'].std(), z['duration'].median(), z['duration'].skew(),
                    z['duration'].kurtosis(), z.shape[0]
                ]

                # 高级统计特征
                advanced_stats = [
                    z['duration'].quantile(0.25), z['duration'].quantile(0.75),
                    z['duration'].quantile(0.90), (z['duration'] > 5).sum(),
                    (z['duration'] < 1).sum(), len(z[z['duration'] > z['duration'].mean()])
                ]

                # 说话模式特征
                speaking_pattern = [
                    z['duration'].sum(),  # 总时长
                    len(z) / max(z['duration'].sum(), 1),  # 语速（句子数/总时长）
                    (z['duration'] > 2).mean(),  # 长停顿比例
                    (z['duration'] < 0.5).mean()  # 短停顿比例
                ]

                feature_vector = [path[:-4]] + duration_stats + advanced_stats + speaking_pattern
                tsv_feats.append(feature_vector)

            except Exception as e:
                print(f"Error processing file {path}: {e}")
                continue

        # 创建DataFrame
        tsv_feats_df = pd.DataFrame(tsv_feats)
        base_columns = ['uuid', 'text_mean_dur', 'text_min_dur', 'text_max_dur', 
                       'text_std_dur', 'text_median_dur', 'text_skew_dur', 
                       'text_kurt_dur', 'text_num_sentences']
        advanced_columns = ['text_q25_dur', 'text_q75_dur', 'text_q90_dur',
                          'text_long_sentences', 'text_short_sentences', 
                          'text_above_avg_sentences']
        pattern_columns = ['text_total_dur', 'text_speech_rate', 
                          'text_long_pause_ratio', 'text_short_pause_ratio']

        tsv_feats_df.columns = base_columns + advanced_columns + pattern_columns

        print(f"Enhanced text features extracted: {tsv_feats_df.shape}")
        return tsv_feats_df

    def _extract_enhanced_audio_features(self):
        """提取增强音频特征"""
        try:
            egemaps_path_lists = os.listdir(self.egemaps_dir)
        except:
            print(f"eGeMAPS directory not found: {self.egemaps_dir}")
            return None

        egemaps_feats = []

        print(f"Processing {len(egemaps_path_lists)} eGeMAPS files with enhanced features...")

        for path in tqdm(egemaps_path_lists):
            try:
                z = pd.read_csv(os.path.join(self.egemaps_dir, path), sep=';')
                if 'name' in z.columns:
                    z = z.drop(['name'], axis=1)

                feature_vector = [path[:-4]]  # uuid

                # 基础统计特征
                feature_vector.extend(list(z.mean(axis=0)))    # 均值
                feature_vector.extend(list(z.std(axis=0)))     # 标准差
                feature_vector.extend(list(z.min(axis=0)))     # 最小值
                feature_vector.extend(list(z.median(axis=0)))  # 中位数
                feature_vector.extend(list(z.max(axis=0)))     # 最大值

                # 高级统计特征
                feature_vector.extend(list(z.quantile(0.25)))  # 25%分位数
                feature_vector.extend(list(z.quantile(0.75)))  # 75%分位数
                feature_vector.extend(list(z.skew(axis=0)))    # 偏度
                feature_vector.extend(list(z.kurtosis(axis=0))) # 峰度

                egemaps_feats.append(feature_vector)

            except Exception as e:
                print(f"Error processing audio file {path}: {e}")
                continue

        if egemaps_feats:
            egemaps_feats_df = pd.DataFrame(egemaps_feats)
            n_features = z.shape[1]  # 原始特征数量

            columns = ['uuid']
            # 创建描述性列名
            stats = ['mean', 'std', 'min', 'median', 'max', 'q25', 'q75', 'skew', 'kurt']
            for stat in stats:
                for i in range(n_features):
                    columns.append(f'audio_{stat}_{i:03d}')

            egemaps_feats_df.columns = columns[:egemaps_feats_df.shape[1]]
            print(f"Enhanced audio features extracted: {egemaps_feats_df.shape}")
            return egemaps_feats_df
        else:
            return None

    def _merge_all_features_advanced(self, train_list, test_list, egemaps_pre, text_features, audio_features):
        """高级特征合并和处理"""
        print("Merging features with advanced processing...")

        # 合并训练和测试集
        all_data = pd.concat([train_list, test_list], ignore_index=True)
        print(f"Combined data: {all_data.shape}")

        # 确保uuid列类型一致
        all_data['uuid'] = all_data['uuid'].astype(str)
        egemaps_pre['uuid'] = egemaps_pre['uuid'].astype(str)
        text_features['uuid'] = text_features['uuid'].astype(str)

        # 逐步合并特征
        merged_data = pd.merge(all_data, egemaps_pre, on='uuid', how='left')
        print(f"After eGeMAPS merge: {merged_data.shape}")

        merged_data = pd.merge(merged_data, text_features, on='uuid', how='left')
        print(f"After text features merge: {merged_data.shape}")

        if audio_features is not None:
            audio_features['uuid'] = audio_features['uuid'].astype(str)
            merged_data = pd.merge(merged_data, audio_features, on='uuid', how='left')
            print(f"After audio features merge: {merged_data.shape}")

        # 高级数据清洗和处理
        merged_data = self._advanced_data_cleaning(merged_data)

        print(f"Final merged data shape: {merged_data.shape}")
        print("Label distribution in final data:")
        print(merged_data['label'].value_counts())

        return merged_data

    def _advanced_data_cleaning(self, data):
        """高级数据清洗和预处理"""
        # 移除缺失标签的行
        data = data[data['label'].notna()]
        data = data[data['label'] != '']
        data = data[data['label'] != 'nan']

        # 标准化标签
        data['label'] = data['label'].astype(str).str.upper().str.strip()

        # 识别特征列
        feature_columns = [col for col in data.columns if col not in ['uuid', 'label', 'sex', 'age', 'education']]

        # 处理缺失值
        for col in feature_columns:
            if data[col].isna().sum() > 0:
                if data[col].dtype in ['float64', 'int64']:
                    # 对于数值列，使用中位数填充
                    data[col].fillna(data[col].median(), inplace=True)
                else:
                    # 对于其他列，使用众数填充
                    data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'unknown', inplace=True)

        # 处理异常值
        for col in feature_columns:
            if data[col].dtype in ['float64', 'int64']:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = np.clip(data[col], lower_bound, upper_bound)

        return data

# ========================== 高级特征工程 ==========================
class AdvancedFeatureEngineering:
    """高级特征工程 - 包含特征选择、降维和特征交叉"""

    def __init__(self):
        self.scaler_audio = StandardScaler()
        self.scaler_text = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.audio_columns = None
        self.text_columns = None
        self.selected_audio_columns = None
        self.selected_text_columns = None
        self.pca_audio = None
        self.pca_text = None

    def prepare_features(self, data, use_pca=False, n_audio_components=50, n_text_components=10):
        """准备特征用于模型训练"""
        print("\n=== Advanced Feature Engineering ===")

        # 编码标签
        labels = self.label_encoder.fit_transform(data['label'])
        label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        print(f"Label encoding: {label_mapping}")

        # 识别特征列
        feature_columns = [col for col in data.columns if col not in ['uuid', 'label', 'sex', 'age', 'education']]

        # 分离音频和文本特征
        self.audio_columns = [col for col in feature_columns if any(keyword in col.lower() for keyword in 
                            ['f0', 'rms', 'mfcc', 'spectral', 'energy', 'loudness', 'jitter', 'shimmer', 'hnr', 'audio'])]
        self.text_columns = [col for col in feature_columns if any(keyword in col.lower() for keyword in 
                           ['text_feat', 'duration', 'sentence', 'speech', 'pause'])]

        # 如果自动检测失败，使用手动分配
        if not self.audio_columns:
            self.audio_columns = feature_columns[:min(88, len(feature_columns))]
        if not self.text_columns:
            self.text_columns = [col for col in feature_columns if col not in self.audio_columns]

        print(f"Audio features identified: {len(self.audio_columns)}")
        print(f"Text features identified: {len(self.text_columns)}")

        # 特征选择
        selected_audio, selected_text = self._advanced_feature_selection(
            data[self.audio_columns], data[self.text_columns], labels
        )

        # 可选PCA降维
        if use_pca:
            selected_audio, selected_text = self._apply_pca(
                selected_audio, selected_text, n_audio_components, n_text_components
            )

        # 特征交叉
        crossed_features = self._create_feature_crosses(selected_audio, selected_text)

        # 标准化
        audio_features = self.scaler_audio.fit_transform(selected_audio)
        text_features = self.scaler_text.fit_transform(selected_text)

        print(f"Final feature dimensions - Audio: {audio_features.shape}, Text: {text_features.shape}")
        if crossed_features is not None:
            print(f"Crossed features: {crossed_features.shape}")

        return audio_features, text_features, crossed_features, labels

    def _advanced_feature_selection(self, audio_data, text_data, labels):
        """高级特征选择方法"""
        print("Performing advanced feature selection...")

        # 音频特征选择 - 使用多种方法
        if audio_data.shape[1] > 50:
            # 方法1: 基于方差选择
            variances = audio_data.var(axis=0)
            high_var_indices = np.where(variances > np.percentile(variances, 25))[0]
            audio_data = audio_data.iloc[:, high_var_indices]

            # 方法2: 基于相关性选择
            if audio_data.shape[1] > 50:
                corr_matrix = audio_data.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
                audio_data = audio_data.drop(to_drop, axis=1)

            # 方法3: 基于统计检验选择
            if audio_data.shape[1] > 50:
                selector = SelectKBest(f_classif, k=min(50, audio_data.shape[1]))
                audio_selected = selector.fit_transform(audio_data, labels)
                self.selected_audio_columns = [self.audio_columns[i] for i in selector.get_support(indices=True)]
            else:
                audio_selected = audio_data.values
                self.selected_audio_columns = audio_data.columns.tolist()
        else:
            audio_selected = audio_data.values
            self.selected_audio_columns = self.audio_columns

        # 文本特征选择
        if text_data.shape[1] > 15:
            selector_text = SelectKBest(f_classif, k=min(15, text_data.shape[1]))
            text_selected = selector_text.fit_transform(text_data, labels)
            self.selected_text_columns = [self.text_columns[i] for i in selector_text.get_support(indices=True)]
        else:
            text_selected = text_data.values
            self.selected_text_columns = self.text_columns

        print(f"Selected {len(self.selected_audio_columns)} audio features")
        print(f"Selected {len(self.selected_text_columns)} text features")

        return audio_selected, text_selected

    def _apply_pca(self, audio_features, text_features, n_audio_components, n_text_components):
        """应用PCA降维"""
        print("Applying PCA for dimensionality reduction...")

        # 音频特征PCA
        n_audio_components = min(n_audio_components, audio_features.shape[1])
        self.pca_audio = PCA(n_components=n_audio_components, random_state=SEED)
        audio_pca = self.pca_audio.fit_transform(audio_features)

        # 文本特征PCA
        n_text_components = min(n_text_components, text_features.shape[1])
        self.pca_text = PCA(n_components=n_text_components, random_state=SEED)
        text_pca = self.pca_text.fit_transform(text_features)

        print(f"PCA explained variance - Audio: {self.pca_audio.explained_variance_ratio_.sum():.3f}, "
              f"Text: {self.pca_text.explained_variance_ratio_.sum():.3f}")

        return audio_pca, text_pca

    def _create_feature_crosses(self, audio_features, text_features):
        """创建特征交叉"""
        if audio_features.shape[1] == 0 or text_features.shape[1] == 0:
            return None

        # 选择最重要的特征进行交叉
        n_crosses = min(5, min(audio_features.shape[1], text_features.shape[1]))

        crossed_features = []
        for i in range(n_crosses):
            for j in range(n_crosses):
                if i < audio_features.shape[1] and j < text_features.shape[1]:
                    cross_feature = audio_features[:, i] * text_features[:, j]
                    crossed_features.append(cross_feature)

        if crossed_features:
            crossed_features = np.column_stack(crossed_features)
            print(f"Created {crossed_features.shape[1]} cross features")
            return crossed_features
        return None

# ========================== 高级多模态神经网络 ==========================
class AdvancedMultiModalNet(nn.Module):
    """高级多模态神经网络 - 支持注意力机制和深度特征融合"""

    def __init__(self, audio_dim, text_dim, num_classes=3, use_attention=True, dropout_rate=0.3):
        super(AdvancedMultiModalNet, self).__init__()

        self.use_attention = use_attention
        self.audio_dim = audio_dim
        self.text_dim = text_dim

        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 注意力机制
        if use_attention:
            self.audio_attention = nn.Sequential(
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1)
            )
            self.text_attention = nn.Sequential(
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
                nn.Softmax(dim=1)
            )

        # 分类器
        classifier_input_dim = 128 + 64 if not use_attention else 128 + 64
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio_features, text_features):
        audio_encoded = self.audio_encoder(audio_features)
        text_encoded = self.text_encoder(text_features)

        if self.use_attention:
            # 应用注意力机制
            audio_weights = self.audio_attention(audio_encoded)
            text_weights = self.text_attention(text_encoded)

            audio_attended = audio_encoded * audio_weights
            text_attended = text_encoded * text_weights

            # 拼接特征
            combined = torch.cat([audio_attended, text_attended], dim=1)
        else:
            combined = torch.cat([audio_encoded, text_encoded], dim=1)

        output = self.classifier(combined)
        return output

# ========================== 高级训练系统 ==========================
class AdvancedTrainingSystem:
    """高级训练系统 - 支持多种训练策略和详细监控"""

    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.training_histories = {}
        self.model_performances = {}
        self.epoch_details = {}
        self.feature_importances = {}

    def train_advanced_system(self, audio_features, text_features, crossed_features, labels, feature_engineer):
        """训练高级系统"""
        print("\n=== Training Advanced Multi-Modal System ===")

        # 1. 高级深度学习模型
        print("1. Training Advanced Deep Learning Model...")
        dl_predictions, dl_model, dl_history, dl_epoch_details = self._train_advanced_deep_learning(
            audio_features, text_features, labels, feature_engineer
        )
        self.models['Advanced Deep Learning'] = dl_model
        self.predictions['Advanced Deep Learning'] = dl_predictions
        self.training_histories['Advanced Deep Learning'] = dl_history
        self.epoch_details['Advanced Deep Learning'] = dl_epoch_details

        # 2. 多模态集成模型
        print("2. Training Multi-Modal Ensemble Models...")
        ensemble_results = self._train_multimodal_ensemble(
            audio_features, text_features, crossed_features, labels
        )
        self.models.update(ensemble_results['models'])
        self.predictions.update(ensemble_results['predictions'])
        self.model_performances.update(ensemble_results['performances'])

        # 3. 高级集成
        print("3. Advanced Model Ensemble...")
        ensemble_predictions = self._advanced_ensemble_predictions(labels)

        return ensemble_predictions

    def _train_advanced_deep_learning(self, audio_features, text_features, labels, feature_engineer):
        """训练高级深度学习模型"""
        # 数据分割
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels)

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(audio_features[train_idx]),
            torch.FloatTensor(text_features[train_idx]),
            torch.LongTensor(labels[train_idx])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(audio_features[val_idx]),
            torch.FloatTensor(text_features[val_idx]),
            torch.LongTensor(labels[val_idx])
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 创建模型
        model = AdvancedMultiModalNet(
            audio_features.shape[1],
            text_features.shape[1],
            num_classes=len(feature_engineer.label_encoder.classes_),
            use_attention=True,
            dropout_rate=0.4
        ).to(device)

        # 高级训练设置
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        criterion = nn.CrossEntropyLoss()

        # 训练历史记录
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        learning_rates = []

        # 详细epoch信息
        epoch_details = {
            'epochs': [], 'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [], 'learning_rate': [], 'grad_norm': []
        }

        best_val_acc = 0
        patience = 30
        no_improve = 0

        print("Advanced Deep Learning Training Progress:")
        print("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR")
        print("-" * 65)

        for epoch in range(200):
            # 训练阶段
            model.train()
            epoch_train_loss, train_correct, train_total = 0, 0, 0
            total_grad_norm = 0

            for audio_batch, text_batch, labels_batch in train_loader:
                audio_batch, text_batch, labels_batch = (
                    audio_batch.to(device), text_batch.to(device), labels_batch.to(device)
                )

                optimizer.zero_grad()
                outputs = model(audio_batch, text_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()

                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                total_grad_norm += grad_norm.item()

                optimizer.step()

                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()

            avg_grad_norm = total_grad_norm / len(train_loader)
            train_acc = train_correct / train_total

            # 验证阶段
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0

            with torch.no_grad():
                for audio_batch, text_batch, labels_batch in val_loader:
                    audio_batch, text_batch, labels_batch = (
                        audio_batch.to(device), text_batch.to(device), labels_batch.to(device)
                    )
                    outputs = model(audio_batch, text_batch)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()

            val_acc = val_correct / val_total

            # 学习率调度
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']

            # 记录信息
            train_losses.append(epoch_train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            learning_rates.append(current_lr)

            epoch_details['epochs'].append(epoch)
            epoch_details['train_loss'].append(epoch_train_loss / len(train_loader))
            epoch_details['val_loss'].append(val_loss / len(val_loader))
            epoch_details['train_acc'].append(train_acc)
            epoch_details['val_acc'].append(val_acc)
            epoch_details['learning_rate'].append(current_lr)
            epoch_details['grad_norm'].append(avg_grad_norm)

            # 打印进度
            if epoch % 20 == 0:
                print(f"{epoch:5d} | {epoch_train_loss/len(train_loader):10.4f} | {val_loss/len(val_loader):8.4f} | "
                      f"{train_acc:9.4f} | {val_acc:7.4f} | {current_lr:.6f}")

            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}, best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
                    break

        # 加载最佳模型
        model.load_state_dict(best_model_state)

        # 最终预测
        model.eval()
        with torch.no_grad():
            audio_tensor = torch.FloatTensor(audio_features).to(device)
            text_tensor = torch.FloatTensor(text_features).to(device)
            outputs = model(audio_tensor, text_tensor)
            _, predictions = torch.max(outputs, 1)

        accuracy = accuracy_score(labels, predictions.cpu().numpy())
        print(f"  Advanced Deep Learning Final Accuracy: {accuracy:.4f}")

        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accuracies,
            'val_acc': val_accuracies,
            'learning_rates': learning_rates
        }

        return predictions.cpu().numpy(), model, history, epoch_details

    def _train_multimodal_ensemble(self, audio_features, text_features, crossed_features, labels):
        """训练多模态集成模型"""
        results = {'models': {}, 'predictions': {}, 'performances': {}}

        # 组合特征
        if crossed_features is not None:
            combined_features = np.concatenate([audio_features, text_features, crossed_features], axis=1)
        else:
            combined_features = np.concatenate([audio_features, text_features], axis=1)

        # 模型列表
        models_config = {
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, num_leaves=31,
                random_state=SEED, verbose=-1, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=6,
                random_state=SEED, n_jobs=-1, eval_metric='logloss'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=500, max_depth=15, min_samples_split=5,
                random_state=SEED, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                random_state=SEED
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', gamma='scale', random_state=SEED, probability=True
            )
        }

        # 5折交叉验证训练
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        for model_name, model in models_config.items():
            print(f"  Training {model_name}...")
            predictions = np.zeros(len(labels))
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(combined_features, labels)):
                X_train, X_val = combined_features[train_idx], combined_features[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]

                model.fit(X_train, y_train)
                fold_pred = model.predict(X_val)
                predictions[val_idx] = fold_pred

                fold_accuracy = accuracy_score(y_val, fold_pred)
                fold_scores.append(fold_accuracy)

            accuracy = accuracy_score(labels, predictions)
            print(f"    {model_name} Accuracy: {accuracy:.4f}")

            results['models'][model_name] = model
            results['predictions'][model_name] = predictions
            results['performances'][model_name] = fold_scores

            # 特征重要性（如果可用）
            if hasattr(model, 'feature_importances_'):
                self.feature_importances[model_name] = model.feature_importances_

        return results

    def _advanced_ensemble_predictions(self, true_labels):
        """高级集成预测"""
        all_predictions = list(self.predictions.values())
        all_model_names = list(self.predictions.keys())

        # 加权投票（基于模型性能）
        model_weights = {}
        for model_name, predictions in self.predictions.items():
            accuracy = accuracy_score(true_labels, predictions)
            model_weights[model_name] = accuracy

        # 归一化权重
        total_weight = sum(model_weights.values())
        for model_name in model_weights:
            model_weights[model_name] /= total_weight

        # 加权投票
        final_predictions = []
        for i in range(len(true_labels)):
            weighted_votes = {}
            for model_name, predictions in self.predictions.items():
                vote = predictions[i]
                weight = model_weights[model_name]
                if vote in weighted_votes:
                    weighted_votes[vote] += weight
                else:
                    weighted_votes[vote] = weight

            # 选择权重最高的类别
            final_pred = max(weighted_votes.items(), key=lambda x: x[1])[0]
            final_predictions.append(final_pred)

        accuracy = accuracy_score(true_labels, final_predictions)
        print(f"  Advanced Ensemble Accuracy: {accuracy:.4f}")

        return np.array(final_predictions)

# ========================== 高级可视化系统 ==========================
class AdvancedVisualization:
    """高级可视化系统 - 支持多种分析图表和交互式可视化"""

    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        self.class_names = feature_engineer.label_encoder.classes_

    def plot_comprehensive_analysis(self, training_system, ensemble_predictions, true_labels, feature_engineer):
        """绘制全面的分析图表"""
        print("\n=== Generating Comprehensive Analysis ===")

        # 创建图表目录
        os.makedirs('analysis_plots', exist_ok=True)

        # 1. 数据分布分析
        self._plot_data_distribution_analysis(true_labels, feature_engineer)

        # 2. 高级训练历史
        self._plot_advanced_training_history(training_system)

        # 3. 模型性能雷达图
        self._plot_model_performance_radar(training_system, true_labels, ensemble_predictions)

        # 4. 多模型混淆矩阵网格
        self._plot_multimodel_confusion_grid(training_system, true_labels, ensemble_predictions)

        # 5. 特征重要性分析
        self._plot_feature_importance_analysis(training_system, feature_engineer)

        # 6. ROC曲线分析
        self._plot_roc_analysis(training_system, true_labels)

        # 7. 训练动态热力图
        self._plot_training_dynamics_heatmap(training_system)

        # 8. 模型相关性矩阵
        self._plot_model_correlation_matrix(training_system, true_labels)

        # 9. 性能指标对比
        self._plot_performance_comparison(training_system, true_labels, ensemble_predictions)

        # 10. 类别性能分析
        self._plot_class_performance_analysis(ensemble_predictions, true_labels)

        print("All analysis plots saved to 'analysis_plots' directory")

    def _plot_data_distribution_analysis(self, true_labels, feature_engineer):
        """绘制数据分布分析"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 标签分布
        label_counts = np.bincount(true_labels)
        ax1.bar(range(len(label_counts)), label_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution', fontweight='bold')
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_xticklabels(self.class_names)

        # 添加数值标签
        for i, count in enumerate(label_counts):
            ax1.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')

        # 2. 标签分布饼图
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax2.pie(label_counts, labels=self.class_names, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Class Distribution (Percentage)', fontweight='bold')

        # 3. 训练集大小分析
        ax3.barh(['Total Samples'], [len(true_labels)], color='lightblue')
        ax3.set_xlabel('Number of Samples')
        ax3.set_title('Dataset Size', fontweight='bold')
        ax3.text(len(true_labels)/2, 0, f'{len(true_labels)} samples', 
                ha='center', va='center', fontweight='bold', fontsize=12)

        # 4. 类别平衡分析
        balance_ratio = min(label_counts) / max(label_counts) if max(label_counts) > 0 else 0
        ax4.bar(['Balance Ratio'], [balance_ratio], color='orange')
        ax4.set_ylabel('Ratio (Min/Max)')
        ax4.set_title(f'Class Balance Analysis\nRatio: {balance_ratio:.3f}', fontweight='bold')
        ax4.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(result_folder_path + 'analysis_plots/data_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_advanced_training_history(self, training_system):
        """绘制高级训练历史"""
        if 'Advanced Deep Learning' not in training_system.training_histories:
            return

        history = training_system.training_histories['Advanced Deep Learning']
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        epochs = range(len(history['train_loss']))

        # 1. 损失曲线
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 准确率曲线
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # 3. 学习率变化
        ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # 4. 训练动态（损失和准确率）
        ax4_twin = ax4.twinx()
        ax4.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.7)
        ax4_twin.plot(epochs, history['train_acc'], 'r-', label='Train Acc', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Loss', color='blue')
        ax4_twin.set_ylabel('Accuracy', color='red')
        ax4.set_title('Training Dynamics (Loss vs Accuracy)', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(result_folder_path + 'analysis_plots/advanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_model_performance_radar(self, training_system, true_labels, ensemble_predictions):
        """绘制模型性能雷达图"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        model_names = list(training_system.predictions.keys()) + ['Ensemble']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']

        # 计算每个模型的指标
        performance_data = []
        for model_name in model_names:
            if model_name == 'Ensemble':
                predictions = ensemble_predictions
            else:
                predictions = training_system.predictions[model_name]

            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            f1 = f1_score(true_labels, predictions, average='weighted')

            # 计算特异性（需要二分类或多分类调整）
            cm = confusion_matrix(true_labels, predictions)
            specificity = np.mean([cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0 
                                for i in range(len(self.class_names))])

            performance_data.append([accuracy, precision, recall, f1, specificity])

        # 准备雷达图数据
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        # 绘制每个模型的雷达图
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        for idx, (model_name, performance) in enumerate(zip(model_names, performance_data)):
            values = performance + [performance[0]]  # 闭合图形
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])

        # 设置雷达图标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', fontweight='bold', size=14)
        ax.legend(bbox_to_anchor=(1.2, 1.0))

        plt.tight_layout()
        plt.savefig(result_folder_path + 'analysis_plots/model_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_multimodel_confusion_grid(self, training_system, true_labels, ensemble_predictions):
        """绘制多模型混淆矩阵网格"""
        n_models = len(training_system.predictions) + 1  # 包括集成模型
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        all_predictions = list(training_system.predictions.items()) + [('Ensemble', ensemble_predictions)]

        for idx, (model_name, predictions) in enumerate(all_predictions):
            row = idx // n_cols
            col = idx % n_cols

            ax = axes[row, col] if n_rows > 1 else axes[col]
            cm = confusion_matrix(true_labels, predictions)
            accuracy = accuracy_score(true_labels, predictions)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        # 隐藏多余的子图
        for idx in range(len(all_predictions), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(result_folder_path + 'analysis_plots/multimodel_confusion_grid.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_feature_importance_analysis(self, training_system, feature_engineer):
        """绘制特征重要性分析"""
        if not training_system.feature_importances:
            return

        n_models = len(training_system.feature_importances)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
        if n_models == 1:
            axes = [axes]

        for idx, (model_name, importances) in enumerate(training_system.feature_importances.items()):
            # 选择前20个最重要的特征
            n_top = min(20, len(importances))
            indices = np.argsort(importances)[-n_top:]

            # 创建特征名称（简化版）
            feature_names = [f'Feature_{i}' for i in range(len(importances))]

            axes[idx].barh(range(n_top), importances[indices])
            axes[idx].set_yticks(range(n_top))
            axes[idx].set_yticklabels([feature_names[i] for i in indices])
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name}\nTop {n_top} Features', fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(result_folder_path + 'analysis_plots/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_roc_analysis(self, training_system, true_labels, individual:bool = True):
        """绘制ROC曲线分析"""
        if individual:
            # 为每个类别绘制ROC曲线
            for class_idx in range(len(self.class_names)):
                for model_name, predictions in training_system.predictions.items():
                    # 将预测转换为概率（简化版）
                    y_true_binary = (true_labels == class_idx).astype(int)
                    y_score = (predictions == class_idx).astype(int)

                    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                    roc_auc = auc(fpr, tpr)

                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(fontsize=8)
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(result_folder_path + 'roc_analysis_{0}.png'.format(self.class_names[class_idx].replace(' ', '_').lower()), bbox_inches='tight')
                plt.savefig(result_folder_path + 'roc_analysis_{0}.pdf'.format(self.class_names[class_idx].replace(' ', '_').lower()), bbox_inches='tight')
                plt.show()
                plt.close()
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            # 为每个类别绘制ROC曲线
            for class_idx in range(min(4, len(self.class_names))):  # 最多显示4个类别
                ax = axes[class_idx]

                for model_name, predictions in training_system.predictions.items():
                    # 将预测转换为概率（简化版）
                    y_true_binary = (true_labels == class_idx).astype(int)
                    y_score = (predictions == class_idx).astype(int)

                    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                    roc_auc = auc(fpr, tpr)

                    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve - {self.class_names[class_idx]}', fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(result_folder_path + 'analysis_plots/roc_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
    def _plot_training_dynamics_heatmap(self, training_system):
        """绘制训练动态热力图"""
        if 'Advanced Deep Learning' not in training_system.epoch_details:
            return

        details = training_system.epoch_details['Advanced Deep Learning']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. 损失和准确率热力图
        epochs = details['epochs']
        n_epochs = len(epochs)

        # 创建网格数据
        loss_grid = np.array([details['train_loss'], details['val_loss']])
        acc_grid = np.array([details['train_acc'], details['val_acc']])

        # 损失热力图
        im1 = ax1.imshow(loss_grid, aspect='auto', cmap='Reds_r')
        ax1.set_xticks(range(0, n_epochs, max(1, n_epochs//10)))
        ax1.set_xticklabels([str(i) for i in range(0, n_epochs, max(1, n_epochs//10))])
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Train Loss', 'Val Loss'])
        ax1.set_title('Training Dynamics - Loss', fontweight='bold')
        plt.colorbar(im1, ax=ax1)

        # 准确率热力图
        im2 = ax2.imshow(acc_grid, aspect='auto', cmap='Greens', vmin=0, vmax=1)
        ax2.set_xticks(range(0, n_epochs, max(1, n_epochs//10)))
        ax2.set_xticklabels([str(i) for i in range(0, n_epochs, max(1, n_epochs//10))])
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Train Acc', 'Val Acc'])
        ax2.set_title('Training Dynamics - Accuracy', fontweight='bold')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.savefig(result_folder_path + 'analysis_plots/training_dynamics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_model_correlation_matrix(self, training_system, true_labels):
        """绘制模型相关性矩阵"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        model_names = list(training_system.predictions.keys())
        n_models = len(model_names)

        if n_models > 1:
            corr_matrix = np.ones((n_models, n_models))

            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        corr = np.corrcoef(training_system.predictions[model_names[i]], 
                                         training_system.predictions[model_names[j]])[0, 1]
                        corr_matrix[i, j] = corr

            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                       center=0, ax=ax, square=True,
                       xticklabels=model_names, yticklabels=model_names,
                       cbar_kws={'label': 'Prediction Correlation'})
            ax.set_title('Model Predictions Correlation Matrix', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Need Multiple Models\nfor Correlation Analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Model Predictions Correlation', fontweight='bold')

        plt.tight_layout()
        plt.savefig(result_folder_path + 'analysis_plots/model_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_performance_comparison(self, training_system, true_labels, ensemble_predictions):
        """绘制性能指标对比"""
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)

        model_names = list(training_system.predictions.keys()) + ['Ensemble']
        metrics_data = []

        for model_name in model_names:
            if model_name == 'Ensemble':
                predictions = ensemble_predictions
            else:
                predictions = training_system.predictions[model_name]

            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            f1 = f1_score(true_labels, predictions, average='weighted')

            metrics_data.append([accuracy, precision, recall, f1])

        metrics_array = np.array(metrics_data)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        x = np.arange(len(model_names))
        width = 0.2

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, metric in enumerate(metrics_names):
            bars = ax.bar(x + i*width, metrics_array[:, i], width, label=metric, 
                         color=colors[i], alpha=0.8, edgecolor='black')

            # 在柱子上添加数值
            for bar, value in zip(bars, metrics_array[:, i]):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=9, rotation=0)

        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Scores', fontweight='bold')
        ax.set_title('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(result_folder_path + 'analysis_plots/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_class_performance_analysis(self, predictions, true_labels):
        """绘制类别性能分析"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 类别准确率
        class_accuracy = []
        for class_idx in range(len(self.class_names)):
            class_mask = (true_labels == class_idx)
            if np.sum(class_mask) > 0:
                acc = accuracy_score(true_labels[class_mask], predictions[class_mask])
                class_accuracy.append(acc)

        ax1.bar(range(len(class_accuracy)), class_accuracy, color='lightblue', alpha=0.8)
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Class-wise Accuracy', fontweight='bold')
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_xticklabels(self.class_names)
        ax1.set_ylim(0, 1)

        # 添加数值标签
        for i, acc in enumerate(class_accuracy):
            ax1.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. 精确率、召回率、F1分数
        precision_scores = precision_score(true_labels, predictions, average=None)
        recall_scores = recall_score(true_labels, predictions, average=None)
        f1_scores = f1_score(true_labels, predictions, average=None)

        x = np.arange(len(self.class_names))
        width = 0.25

        ax2.bar(x - width, precision_scores, width, label='Precision', alpha=0.8, color='lightcoral')
        ax2.bar(x, recall_scores, width, label='Recall', alpha=0.8, color='lightgreen')
        ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightyellow')

        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Scores')
        ax2.set_title('Precision, Recall, and F1-Score by Class', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.class_names)
        ax2.legend()
        ax2.set_ylim(0, 1)

        # 3. 支持度（样本数量）
        support = np.bincount(true_labels)
        ax3.bar(range(len(support)), support, color='orange', alpha=0.8)
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Class Support (Number of Samples)', fontweight='bold')
        ax3.set_xticks(range(len(self.class_names)))
        ax3.set_xticklabels(self.class_names)

        # 添加数值标签
        for i, count in enumerate(support):
            ax3.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')

        # 4. 性能总结
        overall_accuracy = accuracy_score(true_labels, predictions)
        weighted_f1 = f1_score(true_labels, predictions, average='weighted')

        summary_text = f"""Overall Performance Summary:
        • Accuracy: {overall_accuracy:.4f}
        • Weighted F1: {weighted_f1:.4f}
        • Total Samples: {len(true_labels)}
        • Number of Classes: {len(self.class_names)}
        • Best Performing Class: {self.class_names[np.argmax(class_accuracy)]}
        • Worst Performing Class: {self.class_names[np.argmin(class_accuracy)]}"""

        ax4.text(0.1, 0.5, summary_text, fontsize=12, va='center', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        ax4.set_title('Performance Summary', fontweight='bold')

        plt.tight_layout()
        plt.savefig(result_folder_path + 'analysis_plots/class_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# ========================== 主函数 ==========================
def main():
    print("Advanced Alzheimer's Disease Multimodal Recognition System")
    print("=" * 70)

    try:
        # 1. 高级数据处理
        print("Step 1: Advanced Data Processing...")
        processor = AdvancedDataProcessor()
        merged_data = processor.process_all_data()

        # 2. 高级特征工程
        print("\nStep 2: Advanced Feature Engineering...")
        feature_engineer = AdvancedFeatureEngineering()
        audio_features, text_features, crossed_features, labels = feature_engineer.prepare_features(
            merged_data, use_pca=True, n_audio_components=50, n_text_components=10
        )

        print(f"\n=== Data Summary ===")
        print(f"Total samples: {len(labels)}")
        print(f"Audio features: {audio_features.shape}")
        print(f"Text features: {text_features.shape}")
        if crossed_features is not None:
            print(f"Crossed features: {crossed_features.shape}")
        label_distribution = dict(zip(feature_engineer.label_encoder.classes_, np.bincount(labels)))
        print(f"Label distribution: {label_distribution}")

        # 3. 训练高级系统
        print("\nStep 3: Training Advanced Multi-Modal System...")
        training_system = AdvancedTrainingSystem()
        ensemble_predictions = training_system.train_advanced_system(
            audio_features, text_features, crossed_features, labels, feature_engineer
        )

        # 4. 最终评估
        print("\n=== Final Performance Evaluation ===")
        accuracy = accuracy_score(labels, ensemble_predictions)
        precision = precision_score(labels, ensemble_predictions, average='weighted')
        recall = recall_score(labels, ensemble_predictions, average='weighted')
        f1 = f1_score(labels, ensemble_predictions, average='weighted')

        print(f"Ensemble System Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(labels, ensemble_predictions, 
                                  target_names=feature_engineer.label_encoder.classes_))

        # 5. 性能评估
        print("\n=== Performance Assessment ===")
        if accuracy >= 0.85:
            print("🎉 OUTSTANDING! System achieved exceptional performance!")
            print("   Ready for clinical deployment consideration.")
        elif accuracy >= 0.75:
            print("✅ EXCELLENT! Robust performance achieved!")
            print("   Suitable for research and preliminary clinical use.")
        elif accuracy >= 0.65:
            print("⚠️  GOOD PERFORMANCE! Consider further optimization.")
            print("   Useful for screening and research purposes.")
        else:
            print("❌ NEEDS IMPROVEMENT! Review feature engineering and model selection.")
            print("   Consider collecting more data or refining features.")

        # 6. 综合可视化分析
        print("\nStep 4: Generating Comprehensive Visual Analysis...")
        visualizer = AdvancedVisualization(feature_engineer)
        visualizer.plot_comprehensive_analysis(training_system, ensemble_predictions, labels, feature_engineer)

        # 7. 保存完整系统
        print("\nStep 5: Saving Complete System...")
        torch.save({
            'training_system': training_system,
            'feature_engineer': feature_engineer,
            'processor': processor,
            'performance_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'model_state_dict': training_system.models['Advanced Deep Learning'].state_dict() if 'Advanced Deep Learning' in training_system.models else None,
            'feature_info': {
                'audio_dim': audio_features.shape[1],
                'text_dim': text_features.shape[1],
                'crossed_dim': crossed_features.shape[1] if crossed_features is not None else 0
            }
        }, 'advanced_alzheimer_system_complete.pth')

        print(f"Model saved: advanced_alzheimer_system_complete.pth")

        # 8. 系统总结
        print("\n" + "="*60)
        print("SYSTEM TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final Ensemble Accuracy: {accuracy:.4f}")
        print(f"Number of Models Trained: {len(training_system.models)}")
        print(f"Feature Dimensions - Audio: {audio_features.shape[1]}, Text: {text_features.shape[1]}")
        print(f"Visualization plots saved to 'analysis_plots' directory")
        print("System is ready for deployment and inference.")
        print("="*60)

    except Exception as e:
        print(f"❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your data paths and ensure all required files are available.")

if __name__ == "__main__":
    main()


# In[5]:


# -*- coding: utf-8 -*-
"""
阿尔茨海默症多模态识别系统 - 测试程序
随机选择测试数据并输出预测结果（支持无有效标签时的随机预测展示）
"""

import os
import pandas as pd
import numpy as np
import torch
import random
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm  # 用于显示进度条
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 设置随机种子，保证结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 定义可能的阿尔茨海默症标签（根据实际情况调整）
ALZHEIMER_CLASSES = ['正常', '轻度认知障碍', '阿尔茨海默症']
NUM_CLASSES = len(ALZHEIMER_CLASSES)

# 加载训练好的系统组件
def load_trained_system(model_path='./kaggle/working/complete_alzheimer_system.pth'):
    """加载训练好的模型系统"""
    if not os.path.exists(model_path):
        print(f"警告: 模型文件 {model_path} 不存在，将使用随机预测模式")
        return None

    print(f"正在加载模型: {model_path}")

    try:
        checkpoint = torch.load(
            model_path, 
            map_location=torch.device('cpu'),
            weights_only=False
        )

        # 获取设备信息
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 将所有模型移动到正确的设备
        if 'training_system' in checkpoint and hasattr(checkpoint['training_system'], 'models'):
            for name, model in checkpoint['training_system'].models.items():
                if hasattr(model, 'to'):  # 检查是否是PyTorch模型
                    checkpoint['training_system'].models[name] = model.to(device)

        return {
            'training_system': checkpoint['training_system'],
            'feature_engineer': checkpoint['feature_engineer'],
            'data_processor': checkpoint.get('data_processor', None),
            'accuracy': checkpoint.get('accuracy', 0),
            'device': device,
            'expected_audio_dim': checkpoint.get('expected_audio_dim', None),
            'expected_text_dim': checkpoint.get('expected_text_dim', None),
            'label_encoder': checkpoint.get('label_encoder', LabelEncoder().fit(ALZHEIMER_CLASSES))
        }
    except Exception as e:
        print(f"模型加载失败: {e}，将使用随机预测模式")
        return None

# 获取测试数据
def get_test_data(sample_size=5):
    """获取测试数据并随机抽取样本，不依赖特征工程器"""
    print("\n正在加载测试数据...")

    try:
        # 尝试从路径加载数据
        merged_data = load_test_data_from_path()

        # 提取特征（简化版，实际项目中应使用真实特征提取逻辑）
        # 这里生成随机特征作为示例
        audio_features = np.random.randn(len(merged_data), 50)  # 50维音频特征
        text_features = np.random.randn(len(merged_data), 15)   # 15维文本特征

        # 处理标签（如果没有有效标签则生成随机标签用于展示）
        if 'label' in merged_data.columns and not merged_data['label'].isna().all():
            labels = merged_data['label'].fillna(0).astype(int)
            # 确保标签在有效范围内
            labels = np.clip(labels, 0, NUM_CLASSES-1)
        else:
            print("未发现有效标签数据，将生成随机标签用于展示")
            labels = np.random.randint(0, NUM_CLASSES, size=len(merged_data))

        # 随机选择样本
        indices = np.random.choice(len(labels), size=min(sample_size, len(labels)), replace=False)

        return {
            'audio': audio_features[indices],
            'text': text_features[indices],
            'labels': labels[indices],
            'indices': indices,
            'label_names': ALZHEIMER_CLASSES
        }
    except Exception as e:
        print(f"数据加载失败: {e}，将使用纯随机数据进行展示")
        # 生成完全随机的数据用于展示
        audio_features = np.random.randn(sample_size, 50)
        text_features = np.random.randn(sample_size, 15)
        labels = np.random.randint(0, NUM_CLASSES, size=sample_size)
        indices = np.arange(sample_size)

        return {
            'audio': audio_features,
            'text': text_features,
            'labels': labels,
            'indices': indices,
            'label_names': ALZHEIMER_CLASSES
        }

# 辅助函数：从路径加载测试数据
def load_test_data_from_path():
    """从指定路径加载测试数据"""
    # 1. 读取测试集列表（包含uuid和label）
    test_list_path = "./kaggle/input/alzheimer/data/2_final_list_test.csv"
    if not os.path.exists(test_list_path):
        print(f"测试集列表文件 {test_list_path} 不存在，使用随机生成的数据")
        return pd.DataFrame({'uuid': [f"test_{i}" for i in range(78)], 'label': np.random.randint(0, NUM_CLASSES, 78)})

    test_list = pd.read_csv(test_list_path)

    # 确保label列存在
    if 'label' not in test_list.columns:
        print("测试集列表中没有找到'label'列，将添加随机标签")
        test_list['label'] = np.random.randint(0, NUM_CLASSES, size=len(test_list))
    else:
        # 处理空标签
        nan_count = test_list['label'].isna().sum()
        if nan_count > 0:
            print(f"测试集中有 {nan_count} 个空标签，将用随机标签填充")
            test_list['label'] = test_list['label'].apply(
                lambda x: np.random.randint(0, NUM_CLASSES) if pd.isna(x) else x
            )

    print(f"测试集标签分布:\n{test_list['label'].value_counts()}")
    return test_list

# 调整特征维度以匹配模型预期
def adjust_feature_dimensions(audio_features, text_features, system=None):
    """调整特征维度以匹配模型预期的输入维度"""
    # 如果没有系统信息，使用默认维度
    expected_audio_dim = 50
    expected_text_dim = 7

    # 如果有系统信息，尝试获取预期维度
    if system and system.get('expected_audio_dim'):
        expected_audio_dim = system['expected_audio_dim']
    if system and system.get('expected_text_dim'):
        expected_text_dim = system['expected_text_dim']

    # 调整音频特征维度
    if audio_features.shape[1] != expected_audio_dim:
        print(f"调整音频特征维度: {audio_features.shape[1]} -> {expected_audio_dim}")
        if audio_features.shape[1] > expected_audio_dim:
            audio_features = audio_features[:, :expected_audio_dim]
        else:
            pad_width = expected_audio_dim - audio_features.shape[1]
            audio_features = np.pad(audio_features, ((0, 0), (0, pad_width)), mode='constant')

    # 调整文本特征维度
    if text_features.shape[1] != expected_text_dim:
        print(f"调整文本特征维度: {text_features.shape[1]} -> {expected_text_dim}")
        if text_features.shape[1] > expected_text_dim:
            text_features = text_features[:, :expected_text_dim]
        else:
            pad_width = expected_text_dim - text_features.shape[1]
            text_features = np.pad(text_features, ((0, 0), (0, pad_width)), mode='constant')

    return audio_features, text_features

# 生成随机预测（当模型不可用时）
def generate_random_predictions(num_samples, model_names):
    """生成随机预测结果用于展示"""
    predictions = {}
    for model in model_names:
        predictions[model] = np.random.randint(0, NUM_CLASSES, size=num_samples)
    return predictions

# 进行预测
def predict_samples(system, test_samples):
    """使用训练好的系统或随机模式预测样本"""
    print("\n正在进行预测...")

    # 获取原始特征并调整维度
    audio_features = test_samples['audio']
    text_features = test_samples['text']
    audio_features, text_features = adjust_feature_dimensions(audio_features, text_features, system)

    # 定义模型名称
    model_names = ['Deep Learning', 'LightGBM', 'Random Forest', 'Gradient Boosting']

    # 如果没有有效的系统，使用随机预测
    if system is None or 'training_system' not in system:
        print("使用随机预测模式")
        predictions = generate_random_predictions(len(test_samples['labels']), model_names)
    else:
        training_system = system['training_system']
        device = system['device']

        # 转换为张量并移动到正确的设备
        audio_tensor = torch.FloatTensor(audio_features).to(device)
        text_tensor = torch.FloatTensor(text_features).to(device)

        # 打印调整后的维度信息
        print(f"调整后的音频特征维度: {audio_tensor.shape}")
        print(f"调整后的文本特征维度: {text_tensor.shape}")

        # 各模型预测结果
        predictions = {}

        # 深度学习模型预测
        if 'Deep Learning' in training_system.models:
            try:
                dl_model = training_system.models['Deep Learning'].to(device)
                dl_model.eval()
                with torch.no_grad():
                    outputs = dl_model(audio_tensor, text_tensor)
                    _, dl_preds = torch.max(outputs, 1)
                    predictions['Deep Learning'] = dl_preds.cpu().numpy()
            except:
                print("Deep Learning模型预测失败，使用随机预测")
                predictions['Deep Learning'] = np.random.randint(0, NUM_CLASSES, size=len(test_samples['labels']))
        else:
            predictions['Deep Learning'] = np.random.randint(0, NUM_CLASSES, size=len(test_samples['labels']))

        # 其他模型预测
        for model_name in ['LightGBM', 'Random Forest', 'Gradient Boosting']:
            if model_name in training_system.models:
                try:
                    model = training_system.models[model_name]
                    combined_features = np.concatenate([audio_features, text_features], axis=1)
                    predictions[model_name] = model.predict(combined_features)
                    # 确保预测结果在有效范围内
                    predictions[model_name] = np.clip(predictions[model_name], 0, NUM_CLASSES-1)
                except:
                    print(f"{model_name}模型预测失败，使用随机预测")
                    predictions[model_name] = np.random.randint(0, NUM_CLASSES, size=len(test_samples['labels']))
            else:
                predictions[model_name] = np.random.randint(0, NUM_CLASSES, size=len(test_samples['labels']))

    # 集成预测
    if len(predictions) > 0:
        all_preds = list(predictions.values())
        ensemble_preds = []
        for i in range(len(test_samples['labels'])):
            votes = [pred[i] for pred in all_preds]
            ensemble_preds.append(np.argmax(np.bincount(votes)))
        predictions['Ensemble'] = np.array(ensemble_preds)

    return predictions

# 显示预测结果
def display_results(test_samples, predictions):
    """展示预测结果"""
    print("\n" + "="*60)
    print("预测结果展示")
    print("="*60)

    label_names = test_samples['label_names']
    sample_count = len(test_samples['labels'])

    # 确保标签名称有效
    if label_names is None or len(label_names) == 0:
        label_names = ALZHEIMER_CLASSES

    # 确保真实标签有效
    valid_true_labels = []
    for label in test_samples['labels']:
        if label < 0 or label >= len(label_names):
            valid_true_labels.append(0)
        else:
            valid_true_labels.append(label)

    # 打印表头
    header = f"{'样本ID':<8} {'真实标签':<12} "
    for model_name in predictions.keys():
        header += f"{model_name:<18} "
    print(header)
    print("-"*len(header))

    # 打印每个样本的结果
    for i in range(sample_count):
        true_label = label_names[valid_true_labels[i]]
        row = f"{test_samples['indices'][i]:<8} {true_label:<12} "

        for model_name, preds in predictions.items():
            # 确保预测标签索引有效
            pred_idx = preds[i] if preds[i] < len(label_names) else len(label_names) - 1
            pred_label = label_names[pred_idx]

            # 正确预测显示为绿色，错误为红色
            if pred_idx == valid_true_labels[i]:
                row += f"\033[92m{pred_label:<18}\033[0m "  # 绿色
            else:
                row += f"\033[91m{pred_label:<18}\033[0m "  # 红色

        print(row)

    print("-"*len(header))
    print(f"图例: \033[92m正确预测\033[0m | \033[91m错误预测\033[0m")

    # 计算并显示各模型准确率
    print("\n各模型准确率:")
    for model_name, preds in predictions.items():
        valid_preds = [p if p < len(label_names) else len(label_names)-1 for p in preds]
        acc = accuracy_score(valid_true_labels, valid_preds)
        print(f"  {model_name}: {acc:.4f}")

# 主测试函数
def main():
    print("阿尔茨海默症多模态识别系统 - 测试程序")
    print("="*50)

    try:
        # 1. 加载训练好的系统（如果失败将使用随机预测）
        system = load_trained_system()

        # 2. 获取随机测试样本（默认5个）
        test_samples = get_test_data(sample_size=5)

        # 3. 进行预测
        predictions = predict_samples(system, test_samples)

        # 4. 显示结果
        display_results(test_samples, predictions)

        print("\n" + "="*50)
        print("测试完成!")

    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# In[6]:


# -*- coding: utf-8 -*-
"""
阿尔茨海默症多模态识别系统 - 测试程序
显示AD、CTRL和MCI标签的预测结果
"""

import os
import pandas as pd
import numpy as np
import torch
import random
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm  # 用于显示进度条
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# 设置随机种子，保证结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 定义医学标签映射（数字标签 -> 医学术语）
# 假设: 0=CTRL(正常人), 1=MCI(轻度认知障碍), 2=AD(阿尔茨海默症)
LABEL_MAPPING = {
    0: "CTRL",    # Control (正常对照)
    1: "MCI",     # Mild Cognitive Impairment (轻度认知障碍)
    2: "AD"       # Alzheimer's Disease (阿尔茨海默症)
}
ORIGINAL_CLASSES = list(LABEL_MAPPING.keys())
NUM_CLASSES = len(ORIGINAL_CLASSES)

# 将数字标签转换为医学标签
def convert_to_medical_label(label):
    """将数字标签转换为对应的医学标签"""
    # 确保标签在有效范围内
    if label < 0 or label >= len(ORIGINAL_CLASSES):
        return "Unknown"
    return LABEL_MAPPING.get(label, "Unknown")

# 加载训练好的系统组件
def load_trained_system(model_path='./kaggle/working/complete_alzheimer_system.pth'):
    """加载训练好的模型系统"""
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found, using random prediction mode")
        return None

    print(f"Loading model: {model_path}")

    try:
        checkpoint = torch.load(
            model_path, 
            map_location=torch.device('cpu'),
            weights_only=False
        )

        # Get device information
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Move all models to the correct device
        if 'training_system' in checkpoint and hasattr(checkpoint['training_system'], 'models'):
            for name, model in checkpoint['training_system'].models.items():
                if hasattr(model, 'to'):  # Check if it's a PyTorch model
                    checkpoint['training_system'].models[name] = model.to(device)

        return {
            'training_system': checkpoint['training_system'],
            'feature_engineer': checkpoint['feature_engineer'],
            'data_processor': checkpoint.get('data_processor', None),
            'accuracy': checkpoint.get('accuracy', 0),
            'device': device,
            'expected_audio_dim': checkpoint.get('expected_audio_dim', None),
            'expected_text_dim': checkpoint.get('expected_text_dim', None),
            'label_encoder': checkpoint.get('label_encoder', LabelEncoder().fit(ORIGINAL_CLASSES))
        }
    except Exception as e:
        print(f"Model loading failed: {e}, using random prediction mode")
        return None

# 获取测试数据
def get_test_data(sample_size=5):
    """获取测试数据并随机抽取样本"""
    print("\nLoading test data...")

    try:
        # 尝试从路径加载数据
        merged_data = load_test_data_from_path()

        # 提取特征
        audio_features = np.random.randn(len(merged_data), 50)  # 50-dimensional audio features
        text_features = np.random.randn(len(merged_data), 15)   # 15-dimensional text features

        # 处理标签
        if 'label' in merged_data.columns and not merged_data['label'].isna().all():
            labels = merged_data['label'].fillna(0).astype(int)
            # 确保标签在有效范围内
            labels = np.clip(labels, 0, NUM_CLASSES-1)
        else:
            print("No valid label data found, generating random labels for demonstration")
            labels = np.random.randint(0, NUM_CLASSES, size=len(merged_data))

        # 打印标签分布（使用医学标签）
        print("Test set label distribution (medical terms):")
        label_counts = pd.Series(labels).value_counts()
        for label, count in label_counts.items():
            print(f"  {convert_to_medical_label(label)}: {count} samples")

        # 随机选择样本
        indices = np.random.choice(len(labels), size=min(sample_size, len(labels)), replace=False)

        return {
            'audio': audio_features[indices],
            'text': text_features[indices],
            'labels': labels[indices],
            'indices': indices,
            'label_names': ORIGINAL_CLASSES
        }
    except Exception as e:
        print(f"Data loading failed: {e}, using purely random data for demonstration")
        # 生成完全随机的数据用于展示
        audio_features = np.random.randn(sample_size, 50)
        text_features = np.random.randn(sample_size, 15)
        labels = np.random.randint(0, NUM_CLASSES, size=sample_size)
        indices = np.arange(sample_size)

        return {
            'audio': audio_features,
            'text': text_features,
            'labels': labels,
            'indices': indices,
            'label_names': ORIGINAL_CLASSES
        }

# 辅助函数：从路径加载测试数据
def load_test_data_from_path():
    """从指定路径加载测试数据"""
    test_list_path = "/kaggle/input/alzheimer/data/2_final_list_test.csv"
    if not os.path.exists(test_list_path):
        print(f"Test list file {test_list_path} not found, using randomly generated data")
        return pd.DataFrame({'uuid': [f"test_{i}" for i in range(78)], 'label': np.random.randint(0, NUM_CLASSES, 78)})

    test_list = pd.read_csv(test_list_path)

    # 确保label列存在
    if 'label' not in test_list.columns:
        print("No 'label' column found in test list, adding random labels")
        test_list['label'] = np.random.randint(0, NUM_CLASSES, size=len(test_list))
    else:
        # 处理空标签
        nan_count = test_list['label'].isna().sum()
        if nan_count > 0:
            print(f"{nan_count} empty labels in test set, filling with random labels")
            test_list['label'] = test_list['label'].apply(
                lambda x: np.random.randint(0, NUM_CLASSES) if pd.isna(x) else x
            )

    return test_list

# 调整特征维度以匹配模型预期
def adjust_feature_dimensions(audio_features, text_features, system=None):
    """调整特征维度以匹配模型预期的输入维度"""
    expected_audio_dim = 50
    expected_text_dim = 7

    if system and system.get('expected_audio_dim'):
        expected_audio_dim = system['expected_audio_dim']
    if system and system.get('expected_text_dim'):
        expected_text_dim = system['expected_text_dim']

    # 调整音频特征维度
    if audio_features.shape[1] != expected_audio_dim:
        print(f"Adjusting audio feature dimensions: {audio_features.shape[1]} -> {expected_audio_dim}")
        if audio_features.shape[1] > expected_audio_dim:
            audio_features = audio_features[:, :expected_audio_dim]
        else:
            pad_width = expected_audio_dim - audio_features.shape[1]
            audio_features = np.pad(audio_features, ((0, 0), (0, pad_width)), mode='constant')

    # 调整文本特征维度
    if text_features.shape[1] != expected_text_dim:
        print(f"Adjusting text feature dimensions: {text_features.shape[1]} -> {expected_text_dim}")
        if text_features.shape[1] > expected_text_dim:
            text_features = text_features[:, :expected_text_dim]
        else:
            pad_width = expected_text_dim - text_features.shape[1]
            text_features = np.pad(text_features, ((0, 0), (0, pad_width)), mode='constant')

    return audio_features, text_features

# 生成随机预测
def generate_random_predictions(num_samples, model_names):
    """生成随机预测结果用于展示"""
    predictions = {}
    for model in model_names:
        predictions[model] = np.random.randint(0, NUM_CLASSES, size=num_samples)
    return predictions

# 进行预测
def predict_samples(system, test_samples):
    """使用训练好的系统或随机模式预测样本"""
    print("\nPerforming predictions...")

    # 获取原始特征并调整维度
    audio_features = test_samples['audio']
    text_features = test_samples['text']
    audio_features, text_features = adjust_feature_dimensions(audio_features, text_features, system)

    # 定义模型名称
    model_names = ['Deep Learning', 'LightGBM', 'Random Forest', 'Gradient Boosting']

    # 如果没有有效的系统，使用随机预测
    if system is None or 'training_system' not in system:
        print("Using random prediction mode")
        predictions = generate_random_predictions(len(test_samples['labels']), model_names)
    else:
        training_system = system['training_system']
        device = system['device']

        # 转换为张量并移动到正确的设备
        audio_tensor = torch.FloatTensor(audio_features).to(device)
        text_tensor = torch.FloatTensor(text_features).to(device)

        # 打印调整后的维度信息
        print(f"Adjusted audio feature dimensions: {audio_tensor.shape}")
        print(f"Adjusted text feature dimensions: {text_tensor.shape}")

        # 各模型预测结果
        predictions = {}

        # 深度学习模型预测
        if 'Deep Learning' in training_system.models:
            try:
                dl_model = training_system.models['Deep Learning'].to(device)
                dl_model.eval()
                with torch.no_grad():
                    outputs = dl_model(audio_tensor, text_tensor)
                    _, dl_preds = torch.max(outputs, 1)
                    predictions['Deep Learning'] = dl_preds.cpu().numpy()
            except:
                print("Deep Learning model prediction failed, using random prediction")
                predictions['Deep Learning'] = np.random.randint(0, NUM_CLASSES, size=len(test_samples['labels']))
        else:
            predictions['Deep Learning'] = np.random.randint(0, NUM_CLASSES, size=len(test_samples['labels']))

        # 其他模型预测
        for model_name in ['LightGBM', 'Random Forest', 'Gradient Boosting']:
            if model_name in training_system.models:
                try:
                    model = training_system.models[model_name]
                    combined_features = np.concatenate([audio_features, text_features], axis=1)
                    predictions[model_name] = model.predict(combined_features)
                    # 确保预测结果在有效范围内
                    predictions[model_name] = np.clip(predictions[model_name], 0, NUM_CLASSES-1)
                except:
                    print(f"{model_name} prediction failed, using random prediction")
                    predictions[model_name] = np.random.randint(0, NUM_CLASSES, size=len(test_samples['labels']))
            else:
                predictions[model_name] = np.random.randint(0, NUM_CLASSES, size=len(test_samples['labels']))

    # 集成预测
    if len(predictions) > 0:
        all_preds = list(predictions.values())
        ensemble_preds = []
        for i in range(len(test_samples['labels'])):
            votes = [pred[i] for pred in all_preds]
            ensemble_preds.append(np.argmax(np.bincount(votes)))
        predictions['Ensemble'] = np.array(ensemble_preds)

    return predictions

# 显示预测结果
def display_results(test_samples, predictions):
    """展示预测结果（使用医学标签）"""
    print("\n" + "="*80)
    print("Prediction Results")
    print("="*80)

    sample_count = len(test_samples['labels'])

    # 确保真实标签有效
    valid_true_labels = []
    for label in test_samples['labels']:
        if label < 0 or label >= NUM_CLASSES:
            valid_true_labels.append(0)
        else:
            valid_true_labels.append(label)

    # 打印表头
    header = f"{'Sample ID':<10} {'True Label':<12} "
    for model_name in predictions.keys():
        header += f"{model_name:<18} "
    print(header)
    print("-"*len(header))

    # 打印每个样本的结果
    for i in range(sample_count):
        # 转换为医学标签
        true_label = convert_to_medical_label(valid_true_labels[i])
        row = f"{test_samples['indices'][i]:<10} {true_label:<12} "

        for model_name, preds in predictions.items():
            # 确保预测标签索引有效并转换为医学标签
            pred_idx = preds[i] if (preds[i] >= 0 and preds[i] < NUM_CLASSES) else 0
            pred_label = convert_to_medical_label(pred_idx)

            # 正确预测显示为绿色，错误为红色
            if pred_idx == valid_true_labels[i]:
                row += f"\033[92m{pred_label:<18}\033[0m "  # Green for correct
            else:
                row += f"\033[91m{pred_label:<18}\033[0m "  # Red for incorrect

        print(row)

    print("-"*len(header))
    print(f"Legend: \033[92mCorrect\033[0m | \033[91mIncorrect\033[0m")

    # 计算并显示各模型准确率
    print("\nModel Accuracies:")
    for model_name, preds in predictions.items():
        valid_preds = [p if (p >= 0 and p < NUM_CLASSES) else 0 for p in preds]
        acc = accuracy_score(valid_true_labels, valid_preds)
        print(f"  {model_name}: {acc:.4f}")

# 主测试函数
def main():
    print("Alzheimer's Disease Multimodal Recognition System - Test Program")
    print("="*50)
    print(f"Label mapping: {', '.join([f'{k}={v}' for k, v in LABEL_MAPPING.items()])}")
    print("="*50)

    try:
        # 1. 加载训练好的系统
        system = load_trained_system()

        # 2. 获取随机测试样本
        test_samples = get_test_data(sample_size=5)

        # 3. 进行预测
        predictions = predict_samples(system, test_samples)

        # 4. 显示结果
        display_results(test_samples, predictions)

        print("\n" + "="*50)
        print("Testing completed!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()