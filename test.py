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
def load_trained_system(model_path='./alzdModel.pth'):
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
    test_list_path = "/kaggle/input/alzheimer/data/2_final_list_test.csv"
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