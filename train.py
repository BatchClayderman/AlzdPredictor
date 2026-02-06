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

# -*- coding: utf-8 -*-
"""
阿尔茨海默症多模态识别系统 - 完整可视化版
集成训练过程可视化，中文注释，英文标签显示
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (recall_score, f1_score,
                           accuracy_score, classification_report, 
                           confusion_matrix, precision_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
        
        patience = 90 # 早停检查
        maximum_epoch_count = 100 # 最大训练轮次
        no_improve = 0
        best_val_acc = 0

        print("Deep Learning Training Progress:")
        print("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR")
        print("-" * 60)
        
        for epoch in range(maximum_epoch_count):
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
        model = None
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
        model = None
        combined_features = np.concatenate([audio_features, text_features], axis=1)
        
        # 5折交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        predictions = np.zeros(len(labels))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_features, labels)):
            X_train, X_val = combined_features[train_idx], combined_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            model = RandomForestClassifier(
                class_weight='balanced',
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
        model = None
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
    
    def plot_comprehensive_training_analysis(self, training_system, ensemble_predictions, true_labels, individual:bool = True):
        """绘制全面的训练分析图"""
        print("\n=== Generating Comprehensive Training Analysis ===")
        if individual:
            # 1. 深度学习训练历史（详细）
            self._plot_detailed_training_history(None, training_system)

            # 2. 模型性能对比
            self._plot_model_comparison(None, training_system, true_labels, ensemble_predictions)
            plt.tight_layout()
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0201.png', bbox_inches = 'tight')
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0201.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

            # 3. 集成模型混淆矩阵
            self._plot_confusion_matrix(None, ensemble_predictions, true_labels, "Ensemble Model")
            plt.tight_layout()
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0301.png', bbox_inches = 'tight')
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0301.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

            # 4. 交叉验证分数热力图
            self._plot_cv_heatmap(None, training_system)
            plt.tight_layout()
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0401.png', bbox_inches='tight')
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0401.pdf', bbox_inches = 'tight')
            plt.show()
            plt.close()

            # 5. 学习率和梯度变化
            self._plot_learning_rate_gradient(None, training_system)

            # 6. 模型预测相关性
            self._plot_predictions_correlation(None, training_system, true_labels)
            plt.tight_layout()
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0601.png', bbox_inches='tight')
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0601.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

            # 7. 详细性能指标
            self._plot_detailed_metrics(None, training_system, true_labels, ensemble_predictions)
            plt.tight_layout()
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0701.png', bbox_inches='tight')
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0701.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

            # 8. 训练动态分析
            self._plot_training_dynamics(None, training_system)
            plt.tight_layout()
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0801.png', bbox_inches = 'tight')
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0801.pdf', bbox_inches = 'tight')
            plt.show()
            plt.close()

            # 9. 类别性能分析
            self._plot_class_performance(None, ensemble_predictions, true_labels)
            plt.tight_layout()
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0901.png', bbox_inches = 'tight')
            plt.savefig(result_folder_path + 'comprehensive_training_analysis_0901.pdf', bbox_inches = 'tight')
            plt.show()
            plt.close()

            # 额外绘制个体模型混淆矩阵
            self._plot_individual_model_confusions(training_system, true_labels)

            # 训练过程动画式可视化
            self._plot_training_progression(training_system)
        else:
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
            plt.savefig(result_folder_path + 'comprehensive_training_analysis.png', bbox_inches = 'tight')
            plt.savefig(result_folder_path + 'comprehensive_training_analysis.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

            # 额外绘制个体模型混淆矩阵
            self._plot_individual_model_confusions(training_system, true_labels)

            # 训练过程动画式可视化
            self._plot_training_progression(training_system)
        plt.close()
    
    def _plot_detailed_training_history(self, ax, training_system):
        """绘制详细训练历史"""
        if ax is None or ax is plt:
            if 'Deep Learning' in training_system.training_histories:
                history = training_system.training_histories['Deep Learning']
                epochs = range(len(history['train_loss']))

                # 损失曲线
                plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8, marker = 'o', markevery = len(history['train_loss']) // 10)
                plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8, marker = 's', markevery = len(history['val_loss']) // 10)
                plt.xlabel('Epoch', fontweight='bold')
                plt.ylabel('Loss', fontweight='bold')
                plt.legend(loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(result_folder_path + 'comprehensive_training_analysis_0101.png', bbox_inches='tight')
                plt.savefig(result_folder_path + 'comprehensive_training_analysis_0101.pdf', bbox_inches='tight')
                plt.show()
                plt.close()

                # 准确率曲线（双Y轴）
                plt.plot(epochs, history['train_acc'], 'b--', label='Training Accuracy', linewidth=2, alpha=0.7, marker = 'o', markevery = len(history['train_acc']) // 10)
                plt.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy', linewidth=2, alpha=0.7, marker = 's', markevery = len(history['val_acc']) // 10)
                plt.xlabel('Epoch', fontweight='bold')
                plt.ylabel('Accuracy', fontweight='bold')
                plt.legend(loc='upper right')
                plt.ylim(0, 1)

                # 标记最佳epoch
                best_epoch = np.argmax(history['val_acc'])
                plt.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best Epoch: {best_epoch}')
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(result_folder_path + 'comprehensive_training_analysis_0102.png', bbox_inches='tight')
                plt.savefig(result_folder_path + 'comprehensive_training_analysis_0102.pdf', bbox_inches='tight')
                plt.show()
                plt.close()
            else:
                plt.text(0.5, 0.5, 'No Training History Available',
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.show()
                plt.close()
        else:
            if 'Deep Learning' in training_system.training_histories:
                history = training_system.training_histories['Deep Learning']
                epochs = range(len(history['train_loss']))

                # 损失曲线
                ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
                ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
                ax.set_xlabel('Epoch', fontweight='bold')
                ax.set_ylabel('Loss', fontweight='bold')
                # ax.set_title('Deep Learning Training History\nLoss Curves', fontsize=12, fontweight='bold')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)

                # 准确率曲线（双Y轴）
                ax2 = ax.twinx()
                ax2.plot(epochs, history['train_acc'], 'b--', label='Training Accuracy', linewidth=2, alpha=0.7)
                ax2.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy', linewidth=2, alpha=0.7)
                ax2.set_xlabel('Epoch', fontweight='bold')
                ax2.set_ylabel('Accuracy', fontweight='bold')
                ax2.legend(loc='upper right')
                ax2.set_ylim(0, 1)

                # 标记最佳epoch
                best_epoch = np.argmax(history['val_acc'])
                ax2.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best Epoch: {best_epoch}')
                ax2.legend(loc='upper right')
            else:
                ax.text(0.5, 0.5, 'No Training History Available',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)

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

        if ax is None or ax is plt:
            bars1 = plt.bar(x - width / 2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue',
                           edgecolor='black')
            bars2 = plt.bar(x + width / 2, f1_scores, width, label='F1-Score', alpha=0.8, color='lightcoral',
                           edgecolor='black')
            plt.xlabel('Models', fontweight='bold')
            plt.ylabel('Scores', fontweight='bold')
            # plt.title('Model Performance Comparison\n(Accuracy vs F1-Score)', fontsize=14, fontweight='bold')
            plt.xticks(ticks=x,  # 对应ax.set_xticks(x)：x轴刻度位置
                       labels=model_names,  # 对应ax.set_xticklabels(model_names)：刻度标签
                       rotation=45,  # 标签旋转45度，和原代码一致
                       ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

            # 在柱子上添加数值
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            bars1 = ax.bar(x - width / 2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue',
                            edgecolor='black')
            bars2 = ax.bar(x + width / 2, f1_scores, width, label='F1-Score', alpha=0.8, color='lightcoral',
                            edgecolor='black')
            ax.set_xlabel('Models')
            ax.set_ylabel('Scores')
            # ax.set_title('Model Performance Comparison\n(Accuracy vs F1-Score)', fontsize=14, fontweight='bold')
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
                   yticklabels=self.class_names)
        if ax is None or ax is plt:
            plt.xlabel('Predicted Labels', fontweight='bold')
            plt.ylabel('True Labels', fontweight='bold')
            # plt.title(f'Confusion Matrix\n{title}', fontsize=12, fontweight='bold')
        else:
            ax.set_xlabel('Predicted Labels', fontweight='bold')
            ax.set_ylabel('True Labels', fontweight='bold')
            # ax.set_title(f'Confusion Matrix\n{title}', fontsize=12, fontweight='bold')
    
    def _plot_cv_heatmap(self, ax, training_system):
        """绘制交叉验证分数热力图"""
        cv_data = []
        model_names = []
        
        for model_name, scores in training_system.model_performances.items():
            if scores:  # 只包含有CV分数的模型
                cv_data.append(scores)
                model_names.append(model_name)

        if ax is None or ax is plt:
            if cv_data:
                cv_array = np.array(cv_data)
                sns.heatmap(cv_array, annot=True, fmt='.3f', cmap='YlOrRd', ax=None,
                           xticklabels=[f'Fold {i + 1}' for i in range(cv_array.shape[1])],
                           yticklabels=model_names)
                plt.xlabel('Cross-Validation Folds', fontweight='bold')
                plt.ylabel('Models', fontweight='bold')
            else:
                plt.text(0.5, 0.5, 'No CV Scores Available',
                       ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        else:
            if cv_data:
                cv_array = np.array(cv_data)
                sns.heatmap(cv_array, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                           xticklabels=[f'Fold {i+1}' for i in range(cv_array.shape[1])],
                           yticklabels=model_names)
                ax.set_xlabel('Cross-Validation Folds', fontweight='bold')
                ax.set_ylabel('Models', fontweight='bold')
                # ax.set_title('Cross-Validation Scores\n(5-Fold CV)', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No CV Scores Available',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                # ax.set_title('Cross-Validation Scores', fontsize=12, fontweight='bold')
    
    def _plot_learning_rate_gradient(self, ax, training_system):
        """绘制学习率和梯度变化"""
        if 'Deep Learning' in training_system.epoch_details:
            details = training_system.epoch_details['Deep Learning']
            epochs = details['epochs']

            if ax is None or ax is plt:
                # 学习率曲线
                plt.plot(epochs, details['learning_rate'], 'g-', label='Learning Rate', linewidth=2)
                plt.xlabel('Epoch', fontweight='bold')
                plt.ylabel('Learning Rate', fontweight='bold')
                plt.tick_params(axis='y')
                # plt.set_title('Learning Rate and Gradient Norm', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(result_folder_path + 'comprehensive_training_analysis_0501.png', bbox_inches='tight')
                plt.savefig(result_folder_path + 'comprehensive_training_analysis_0501.pdf', bbox_inches='tight')
                plt.show()
                plt.close()

                # 梯度范数曲线（双Y轴）
                plt.plot(epochs, details['grad_norm'], label='Gradient Norm', linewidth=2, alpha=0.7)
                plt.xlabel('Epoch', fontweight='bold')
                plt.ylabel('Gradient Norm', fontweight='bold')
                plt.tick_params(axis='y')

                # 组合图例
                plt.tight_layout()
                plt.savefig(result_folder_path + 'comprehensive_training_analysis_0502.png', bbox_inches='tight')
                plt.savefig(result_folder_path + 'comprehensive_training_analysis_0502.pdf', bbox_inches='tight')
                plt.show()
                plt.close()
            else:
                # 学习率曲线
                ax.plot(epochs, details['learning_rate'], 'g-', label='Learning Rate', linewidth=2)
                ax.set_xlabel('Epoch', fontweight='bold')
                ax.set_ylabel('Learning Rate', fontweight='bold')
                ax.tick_params(axis='y')
                # ax.set_title('Learning Rate and Gradient Norm', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # 梯度范数曲线（双Y轴）
                ax2 = ax.twinx()
                ax2.plot(epochs, details['grad_norm'], label='Gradient Norm', linewidth=2, alpha=0.7)
                plt.xlabel('Epoch', fontweight='bold')
                ax2.set_ylabel('Gradient Norm', fontweight='bold')
                ax2.tick_params(axis='y')

                # 组合图例
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            if ax is None or ax is plt:
                plt.text(0.5, 0.5, 'No Training Details Available',
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                # plt.title('Learning Rate and Gradient', fontsize=12, fontweight='bold')
                plt.show()
                plt.close()
            else:
                ax.text(0.5, 0.5, 'No Training Details Available',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                # ax.set_title('Learning Rate and Gradient', fontsize=12, fontweight='bold')
    
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
            if ax is None or ax is plt:
                sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                       center=0, ax=None, square=True,
                       xticklabels=model_names, yticklabels=model_names)
                plt.xlabel('Predicted Labels', fontweight='bold')
                plt.ylabel('True Labels', fontweight='bold')
                # plt.title('Model Predictions Correlation', fontsize=12, fontweight='bold')
            else:
                sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                        center=0, ax=ax, square=True,
                        xticklabels=model_names, yticklabels=model_names)
                ax.set_xlabel('Predicted Labels', fontweight='bold')
                ax.set_ylabel('True Labels', fontweight='bold')
                # ax.set_title('Model Predictions Correlation', fontsize=12, fontweight='bold')
        else:
            if ax is None or ax is plt:
                plt.text(0.5, 0.5, 'Need Multiple Models\nfor Correlation Analysis',
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                # plt.title('Model Predictions Correlation', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Need Multiple Models\nfor Correlation Analysis',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
                # ax.set_title('Model Predictions Correlation', fontsize=12, fontweight='bold')
    
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

        if ax is None or ax is plt:
            for i, metric in enumerate(metrics_names):
                bars = plt.bar(x + i*width, metrics_array[:, i], width, label=metric, color=colors[i], alpha=0.8, edgecolor='black')

                # 在柱子上添加数值
                for bar, value in zip(bars, metrics_array[:, i]):
                    plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f'{value:.3f}',
                           ha='center', va='bottom', fontsize=8, rotation=45)

            plt.xlabel('Models', fontweight='bold')
            plt.ylabel('Scores', fontweight='bold')
            # plt.title('Detailed Performance Metrics by Model', fontsize=14, fontweight='bold')
            plt.xticks(x + width*1.5, model_names, rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
        else:
            for i, metric in enumerate(metrics_names):
                bars = ax.bar(x + i * width, metrics_array[:, i], width, label=metric, color=colors[i], alpha=0.8,
                              edgecolor='black')

                # 在柱子上添加数值
                for bar, value in zip(bars, metrics_array[:, i]):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.3f}',
                            ha='center', va='bottom', fontsize=8, rotation=45)

            ax.set_xlabel('Models', fontweight='bold')
            ax.set_ylabel('Scores', fontweight='bold')
            # ax.set_title('Detailed Performance Metrics by Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width * 1.5)
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

            if ax is None or ax is plt:
                plt.plot(epochs, gap, 'orange', linewidth=2, label='Train-Val Gap')
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Gap')
                plt.fill_between(epochs, gap, alpha=0.3, color='orange')

                plt.xlabel('Epoch', fontweight='bold')
                plt.ylabel('Accuracy Gap', fontweight='bold')
                # plt.title('Training Dynamics\n(Train-Validation Gap)', fontsize=12, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # 添加统计信息
                avg_gap = np.mean(gap)
                max_gap = np.max(gap)
                plt.text(0.02, 0.98, f'Avg Gap: {avg_gap:.3f}\nMax Gap: {max_gap:.3f}',
                        transform=plt.gca().transAxes, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.plot(epochs, gap, 'orange', linewidth=2, label='Train-Val Gap')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Gap')
                ax.fill_between(epochs, gap, alpha=0.3, color='orange')

                ax.set_xlabel('Epoch', fontweight='bold')
                ax.set_ylabel('Accuracy Gap', fontweight='bold')
                # ax.set_title('Training Dynamics\n(Train-Validation Gap)', fontsize=12, fontweight='bold')
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
            # ax.set_title('Training Dynamics', fontsize=12, fontweight='bold')
    
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

        if ax is None or ax is plt:
            plt.bar(x - width, class_performance, width, label='Accuracy', alpha=0.8, color='lightblue')
            plt.bar(x, precision_scores, width, label='Precision', alpha=0.8, color='lightcoral')
            plt.bar(x + width, recall_scores, width, label='Recall', alpha=0.8, color='lightgreen')

            plt.xlabel('Classes', fontweight='bold')
            plt.ylabel('Scores', fontweight='bold')
            # plt.title('Class-wise Performance Analysis\n(Ensemble Model)', fontsize=12, fontweight='bold')
            plt.xticks(x, self.class_names)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

            # 添加数值标签
            for i, (acc, prec, rec) in enumerate(zip(class_performance, precision_scores, recall_scores)):
                plt.text(i - width, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
                plt.text(i, prec + 0.02, f'{prec:.2f}', ha='center', va='bottom', fontsize=8)
                plt.text(i + width, rec + 0.02, f'{rec:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.bar(x - width, class_performance, width, label='Accuracy', alpha=0.8, color='lightblue')
            ax.bar(x, precision_scores, width, label='Precision', alpha=0.8, color='lightcoral')
            ax.bar(x + width, recall_scores, width, label='Recall', alpha=0.8, color='lightgreen')

            ax.set_xlabel('Classes', fontweight='bold')
            ax.set_ylabel('Scores', fontweight='bold')
            # ax.set_title('Class-wise Performance Analysis\n(Ensemble Model)', fontsize=12, fontweight='bold')
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
    
    def _plot_individual_model_confusions(self, training_system, true_labels, individual:bool = True):
        """绘制个体模型混淆矩阵"""
        n_models = len(training_system.predictions)
        if n_models > 0:
            os.system('rm individual_model_confusion_*.*')
            if individual:
                name_width = len(str(n_models))
                for idx, (model_name, predictions) in enumerate(training_system.predictions.items()):
                    cm = confusion_matrix(true_labels, predictions)
                    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
                                xticklabels = self.class_names,
                                yticklabels = self.class_names)
                    accuracy = accuracy_score(true_labels, predictions)
                    plt.xlabel('Predicted Labels', fontweight='bold')
                    plt.ylabel('True Labels', fontweight='bold')
                    plt.tight_layout()
                    if model_name and 0 < accuracy < 1:
                        plt.savefig(result_folder_path + 'individual_model_confusion_{0}_{1:.4f}.png'.format(model_name.replace(' ', '_').lower(), accuracy))
                        plt.savefig(result_folder_path + 'individual_model_confusion_{0}_{1:.4f}.pdf'.format(model_name.replace(' ', '_').lower(), accuracy))
                    else:
                        plt.savefig(result_folder_path + 'individual_model_confusion_{{0:0>{0}}}.png'.format(name_width).format(idx + 1), bbox_inches = 'tight')
                        plt.savefig(result_folder_path + 'individual_model_confusion_{{0:0>{0}}}.pdf'.format(name_width).format(idx + 1), bbox_inches = 'tight')
                    plt.show()
            else:
                fig, axes = plt.subplots(2, 2, figsize = (15, 12))
                axes = axes.ravel()

                for idx, (model_name, predictions) in enumerate(training_system.predictions.items()):
                    if idx < 4:  # 限制为4个子图
                        cm = confusion_matrix(true_labels, predictions)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                                   xticklabels=self.class_names,
                                   yticklabels=self.class_names)
                        # axes[idx].set_title('{0}\nAccuracy: {1:.4f}'.format(model_name, accuracy_score(true_labels, predictions)), fontweight='bold')
                        axes[idx].set_xlabel('Predicted Labels', fontweight='bold')
                        axes[idx].set_ylabel('True Labels', fontweight='bold')
                    else:
                        break

                # 隐藏未使用的子图
                for idx in range(n_models, 4):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                plt.savefig(result_folder_path + 'individual_model_confusions.png', bbox_inches = 'tight')
                plt.savefig(result_folder_path + 'individual_model_confusions.pdf', bbox_inches = 'tight')
                plt.show()
            plt.close()
    
    def _plot_training_progression(self, training_system, individual:bool = True):
        """绘制训练进度可视化"""
        if 'Deep Learning' in training_system.epoch_details:
            details = training_system.epoch_details['Deep Learning']
            epochs = details['epochs']

            if individual:
                # 训练进度 - 损失
                plt.plot(epochs, details['train_loss'], 'b-', label = 'Training Loss', linewidth = 2, marker = 'o', markevery = len(details['train_loss']) // 10)
                plt.plot(epochs, details['val_loss'], 'r-', label = 'Validation Loss', linewidth = 2, marker = 's', markevery = len(details['val_loss']) // 10)
                plt.xlabel('Epoch', fontweight='bold')
                plt.ylabel('Loss', fontweight='bold')
                # plt.title('Training Progression - Loss', fontweight = 'bold')
                plt.legend()
                plt.tight_layout()
                plt.savefig(result_folder_path + 'training_progression_loss.png', bbox_inches = 'tight')
                plt.savefig(result_folder_path + 'training_progression_loss.pdf', bbox_inches = 'tight')
                plt.show()
                plt.close()

                # 训练进度 - 准确率
                plt.plot(epochs, details['train_acc'], 'b-', label = 'Training Accuracy', linewidth = 2, marker = 'o', markevery = len(details['train_acc']) // 10)
                plt.plot(epochs, details['val_acc'], 'r-', label = 'Validation Accuracy', linewidth = 2, marker = 's', markevery = len(details['val_acc']) // 10)
                plt.xlabel('Epoch', fontweight='bold')
                plt.ylabel('Accuracy', fontweight='bold')
                # plt.title('Training Progression - Accuracy', fontweight = 'bold')
                plt.legend()
                plt.grid(True, alpha = 0.3)
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(result_folder_path + 'training_progression_accuracy.png', bbox_inches = 'tight')
                plt.savefig(result_folder_path + 'training_progression_accuracy.pdf', bbox_inches = 'tight')
                plt.show()
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                # 训练进度 - 损失
                ax1.plot(epochs, details['train_loss'], 'b-', label='Training Loss', linewidth=2)
                ax1.plot(epochs, details['val_loss'], 'r-', label='Validation Loss', linewidth=2)
                ax1.set_xlabel('Epoch', fontweight='bold')
                ax1.set_ylabel('Loss', fontweight='bold')
                # ax1.set_title('Training Progression - Loss', fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # 训练进度 - 准确率
                ax2.plot(epochs, details['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
                ax2.plot(epochs, details['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
                ax2.set_xlabel('Epoch', fontweight='bold')
                ax2.set_ylabel('Accuracy', fontweight='bold')
                # ax2.set_title('Training Progression - Accuracy', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)

                plt.tight_layout()
                plt.savefig(result_folder_path + 'training_progressions.png', bbox_inches = 'tight')
                plt.savefig(result_folder_path + 'training_progressions.pdf', bbox_inches = 'tight')
                plt.show()
            plt.close()

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
        print("Comprehensive training analysis visualization saved as \"{0}comprehensive_training_analysis.pdf\". ".format(result_folder_path))
        # 7. 保存模型
        torch.save({
            'training_system': training_system,
            'feature_engineer': feature_engineer,
            'accuracy': accuracy,
            'model_state_dict': training_system.models['Deep Learning'].state_dict() if 'Deep Learning' in training_system.models else None
        }, './alzdModel.pth')
        
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