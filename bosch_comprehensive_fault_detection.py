#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bosch Production Line - 종합적인 불량 검출 솔루션
LGES DL AutoEncoder-based Fault Detection Solution 참고

다양한 접근법 구현:
1. 지도학습 (Supervised Learning) - Binary Classification
2. 비지도학습 (Unsupervised Learning) - Clustering, Anomaly Detection  
3. AutoEncoder 기반 이상 탐지
4. Isolation Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import time
from datetime import datetime

# 머신러닝 라이브러리
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           matthews_corrcoef, roc_auc_score, f1_score)
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# 불균형 데이터 처리
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    print("Warning: imbalanced-learn이 설치되지 않음. 기본 샘플링만 사용합니다.")

# LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM이 설치되지 않음. 다른 모델을 사용합니다.")

# PyTorch (AutoEncoder용)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("PyTorch 사용 가능")
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch가 설치되지 않음. 통계적 방법을 사용합니다.")

warnings.filterwarnings('ignore')

class BoschFaultDetectionSuite:
    """
    Bosch 생산라인 불량 검출을 위한 종합적인 솔루션
    """
    
    def __init__(self, sampling_strategy='hybrid', random_state=42):
        self.sampling_strategy = sampling_strategy  # 'under', 'over', 'hybrid'
        self.random_state = random_state
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self, data_path, sample_size=100000):
        """
        데이터 로드 및 전처리
        """
        print("=" * 80)
        print("🔧 데이터 로드 및 전처리")
        print("=" * 80)
        
        # 1. 기본 데이터 로드
        print(f"📂 데이터 로딩: {sample_size:,}개 샘플")
        df = pd.read_csv(data_path, nrows=sample_size)
        print(f"   로드 완료: {df.shape[0]:,} x {df.shape[1]:,}")
        
        # 2. 기본 정보 확인
        normal_count = (df['Response'] == 0).sum()
        fault_count = (df['Response'] == 1).sum()
        fault_rate = fault_count / len(df)
        
        print(f"\n📊 클래스 분포:")
        print(f"   정상 제품: {normal_count:,}개 ({(1-fault_rate)*100:.2f}%)")
        print(f"   불량 제품: {fault_count:,}개 ({fault_rate*100:.2f}%)")
        print(f"   불균형 비율: {int((1-fault_rate)/fault_rate)}:1")
        
        # 3. 특징 선택 및 정리
        print(f"\n🔍 특징 전처리:")
        
        # 기본 특징 분리
        feature_cols = [col for col in df.columns if col not in ['Id', 'Response']]
        X_raw = df[feature_cols].copy()
        y = df['Response'].copy()
        
        print(f"   원본 특징 수: {len(feature_cols):,}개")
        
        # 4. 결측값이 많은 특징 제거 (95% 이상 결측)
        missing_threshold = 0.95
        missing_ratio = X_raw.isnull().sum() / len(X_raw)
        valid_features = missing_ratio[missing_ratio < missing_threshold].index.tolist()
        
        X_filtered = X_raw[valid_features].copy()
        print(f"   결측값 필터링 후: {len(valid_features):,}개")
        
        # 5. 분산이 0인 특징 제거
        X_temp = X_filtered.fillna(0)
        variances = X_temp.var()
        non_zero_var_features = variances[variances > 1e-8].index.tolist()
        
        X_clean = X_filtered[non_zero_var_features].copy()
        print(f"   분산 필터링 후: {len(non_zero_var_features):,}개")
        
        # 6. 결측값 처리
        print(f"\n🛠️ 결측값 처리:")
        missing_before = X_clean.isnull().sum().sum()
        X_clean = X_clean.fillna(X_clean.median())
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        missing_after = X_clean.isnull().sum().sum()
        print(f"   처리 전: {missing_before:,}개 → 처리 후: {missing_after:,}개")
        
        # 7. 기본 집계 특징 생성
        print(f"\n🔧 특징 공학:")
        feature_df = pd.DataFrame()
        
        # 기본 통계 특징
        feature_df['count_non_null'] = X_clean.count(axis=1)
        feature_df['count_zeros'] = (X_clean == 0).sum(axis=1)
        feature_df['missing_ratio'] = X_clean.isnull().sum(axis=1) / len(X_clean.columns)
        feature_df['mean'] = X_clean.mean(axis=1, skipna=True)
        feature_df['std'] = X_clean.std(axis=1, skipna=True)
        feature_df['min'] = X_clean.min(axis=1, skipna=True)
        feature_df['max'] = X_clean.max(axis=1, skipna=True)
        feature_df['range'] = feature_df['max'] - feature_df['min']
        feature_df['median'] = X_clean.median(axis=1, skipna=True)
        
        # 스테이션별 집계 (상위 5개 스테이션)
        station_groups = {}
        for col in X_clean.columns:
            parts = col.split('_')
            if len(parts) >= 2:
                station = parts[0] + '_' + parts[1]
                if station not in station_groups:
                    station_groups[station] = []
                station_groups[station].append(col)
        
        top_stations = sorted(station_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        
        for station, cols in top_stations:
            if len(cols) > 1:
                station_data = X_clean[cols]
                feature_df[f'{station}_mean'] = station_data.mean(axis=1, skipna=True)
                feature_df[f'{station}_count'] = station_data.count(axis=1)
        
        # NaN 값 처리
        feature_df = feature_df.fillna(0)
        
        print(f"   생성된 집계 특징: {len(feature_df.columns)}개")
        
        # 8. 스케일링
        print(f"\n특징 스케일링:")\n        self.scaler = RobustScaler()\n        X_scaled = self.scaler.fit_transform(feature_df)\n        \n        print(f"   스케일링 완료: RobustScaler 사용")\n        \n        # 최종 데이터 정보 저장\n        self.X = X_scaled\n        self.y = y.values\n        self.feature_names = feature_df.columns.tolist()\n        \n        print(f"\n✅ 전처리 완료:")\n        print(f"   최종 데이터 크기: {self.X.shape[0]:,} x {self.X.shape[1]:,}")\n        print(f"   불량률: {self.y.mean():.4%}")\n        \n        return self.X, self.y
    \n    def apply_sampling(self, X, y):\n        \"\"\"\n        불균형 데이터 샘플링 적용\n        \"\"\"\n        print(f\"\\n⚖️ 불균형 데이터 처리 ({self.sampling_strategy}):\")\n        \n        original_counts = np.bincount(y)\n        print(f\"   원본 분포: 정상 {original_counts[0]:,}, 불량 {original_counts[1]:,}\")\n        \n        if self.sampling_strategy == 'under':\n            # 언더샘플링: 다수 클래스를 소수 클래스 수준으로 줄임\n            if IMBALANCED_LEARN_AVAILABLE:\n                sampler = RandomUnderSampler(random_state=self.random_state)\n                X_resampled, y_resampled = sampler.fit_resample(X, y)\n            else:\n                # 수동 언더샘플링\n                minority_count = original_counts[1]\n                normal_indices = np.where(y == 0)[0]\n                fault_indices = np.where(y == 1)[0]\n                \n                selected_normal = np.random.choice(normal_indices, minority_count, replace=False)\n                selected_indices = np.concatenate([selected_normal, fault_indices])\n                \n                X_resampled = X[selected_indices]\n                y_resampled = y[selected_indices]\n        \n        elif self.sampling_strategy == 'over':\n            # 오버샘플링: SMOTE 사용\n            if IMBALANCED_LEARN_AVAILABLE:\n                sampler = SMOTE(random_state=self.random_state)\n                X_resampled, y_resampled = sampler.fit_resample(X, y)\n            else:\n                # 간단한 복제 기반 오버샘플링\n                majority_count = original_counts[0]\n                minority_count = original_counts[1]\n                \n                fault_indices = np.where(y == 1)[0]\n                n_copies = majority_count // minority_count\n                \n                fault_X = X[fault_indices]\n                fault_y = y[fault_indices]\n                \n                # 복제\n                replicated_X = np.tile(fault_X, (n_copies, 1))\n                replicated_y = np.tile(fault_y, n_copies)\n                \n                X_resampled = np.vstack([X, replicated_X])\n                y_resampled = np.concatenate([y, replicated_y])\n        \n        elif self.sampling_strategy == 'hybrid':\n            # 하이브리드: 적절한 균형 맞추기\n            target_size = 20000  # 각 클래스당 목표 크기\n            \n            if IMBALANCED_LEARN_AVAILABLE:\n                # 먼저 언더샘플링\n                under_sampler = RandomUnderSampler(\n                    sampling_strategy={0: target_size * 3, 1: original_counts[1]},\n                    random_state=self.random_state\n                )\n                X_temp, y_temp = under_sampler.fit_resample(X, y)\n                \n                # 그다음 오버샘플링\n                over_sampler = SMOTE(\n                    sampling_strategy={1: target_size},\n                    random_state=self.random_state\n                )\n                X_resampled, y_resampled = over_sampler.fit_resample(X_temp, y_temp)\n            else:\n                # 수동 하이브리드\n                normal_indices = np.where(y == 0)[0]\n                fault_indices = np.where(y == 1)[0]\n                \n                # 정상 데이터 언더샘플링\n                selected_normal = np.random.choice(normal_indices, target_size, replace=False)\n                \n                # 불량 데이터 오버샘플링 (복제)\n                n_copies = target_size // len(fault_indices)\n                selected_fault = np.tile(fault_indices, n_copies)\n                remaining = target_size - len(selected_fault)\n                if remaining > 0:\n                    additional_fault = np.random.choice(fault_indices, remaining, replace=True)\n                    selected_fault = np.concatenate([selected_fault, additional_fault])\n                \n                selected_indices = np.concatenate([selected_normal, selected_fault])\n                X_resampled = X[selected_indices]\n                y_resampled = y[selected_indices]\n        \n        else:\n            # 샘플링 안함\n            X_resampled, y_resampled = X, y\n        \n        new_counts = np.bincount(y_resampled)\n        print(f\"   샘플링 후: 정상 {new_counts[0]:,}, 불량 {new_counts[1]:,}\")\n        print(f\"   새로운 균형: {new_counts[0]/new_counts[1]:.1f}:1\")\n        \n        return X_resampled, y_resampled\n    \n    def train_supervised_models(self, X, y):\n        \"\"\"\n        지도학습 모델 학습 (Binary Classification)\n        \"\"\"\n        print(f\"\\n🤖 지도학습 모델 학습:\")\n        print(\"-\" * 50)\n        \n        # 데이터 분할\n        X_train, X_test, y_train, y_test = train_test_split(\n            X, y, test_size=0.2, random_state=self.random_state, stratify=y\n        )\n        \n        print(f\"   학습 데이터: {X_train.shape[0]:,}개\")\n        print(f\"   테스트 데이터: {X_test.shape[0]:,}개\")\n        \n        # 불균형 처리\n        X_train_balanced, y_train_balanced = self.apply_sampling(X_train, y_train)\n        \n        models_to_train = []\n        \n        # LightGBM\n        if LIGHTGBM_AVAILABLE:\n            lgbm = LGBMClassifier(\n                n_estimators=100,\n                learning_rate=0.1,\n                max_depth=6,\n                random_state=self.random_state,\n                class_weight='balanced',\n                verbosity=-1\n            )\n            models_to_train.append(('LightGBM', lgbm))\n        \n        # Random Forest (sklearn 기본)\n        from sklearn.ensemble import RandomForestClassifier\n        rf = RandomForestClassifier(\n            n_estimators=100,\n            max_depth=8,\n            random_state=self.random_state,\n            class_weight='balanced',\n            n_jobs=-1\n        )\n        models_to_train.append(('RandomForest', rf))\n        \n        # 모델 학습 및 평가\n        for name, model in models_to_train:\n            print(f\"\\n   🔧 {name} 학습 중...\")\n            \n            start_time = time.time()\n            model.fit(X_train_balanced, y_train_balanced)\n            train_time = time.time() - start_time\n            \n            # 예측\n            y_pred = model.predict(X_test)\n            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None\n            \n            # 평가\n            mcc = matthews_corrcoef(y_test, y_pred)\n            f1 = f1_score(y_test, y_pred)\n            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0\n            \n            self.models[name] = model\n            self.results[name] = {\n                'MCC': mcc,\n                'F1': f1,\n                'AUC': auc,\n                'train_time': train_time,\n                'predictions': y_pred,\n                'probabilities': y_pred_proba\n            }\n            \n            print(f\"      MCC: {mcc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}\")\n            print(f\"      학습 시간: {train_time:.2f}초\")\n        \n        return X_test, y_test\n    \n    def train_isolation_forest(self, X, y):\n        \"\"\"\n        Isolation Forest 이상 탐지\n        \"\"\"\n        print(f\"\\n🌲 Isolation Forest 이상 탐지:\")\n        print(\"-\" * 50)\n        \n        # 데이터 분할\n        X_train, X_test, y_train, y_test = train_test_split(\n            X, y, test_size=0.2, random_state=self.random_state, stratify=y\n        )\n        \n        # 정상 데이터만 사용하여 학습\n        X_train_normal = X_train[y_train == 0]\n        print(f\"   정상 데이터로 학습: {len(X_train_normal):,}개\")\n        \n        # 불량률에 기반한 contamination 설정\n        contamination = y.mean()\n        print(f\"   예상 contamination: {contamination:.4%}\")\n        \n        # Isolation Forest 모델\n        iforest = IsolationForest(\n            n_estimators=100,\n            max_samples='auto',\n            contamination=contamination,\n            random_state=self.random_state,\n            n_jobs=-1\n        )\n        \n        start_time = time.time()\n        iforest.fit(X_train_normal)  # 정상 데이터만으로 학습\n        train_time = time.time() - start_time\n        \n        # 예측 (1: normal, -1: abnormal)\n        y_pred_raw = iforest.predict(X_test)\n        y_pred = (y_pred_raw == -1).astype(int)  # -1을 1로, 1을 0으로 변환\n        \n        # 이상 점수\n        anomaly_scores = iforest.decision_function(X_test)\n        \n        # 평가\n        mcc = matthews_corrcoef(y_test, y_pred)\n        f1 = f1_score(y_test, y_pred)\n        auc = roc_auc_score(y_test, -anomaly_scores)  # 음수로 변환 (낮을수록 이상)\n        \n        self.models['IsolationForest'] = iforest\n        self.results['IsolationForest'] = {\n            'MCC': mcc,\n            'F1': f1,\n            'AUC': auc,\n            'train_time': train_time,\n            'predictions': y_pred,\n            'anomaly_scores': anomaly_scores\n        }\n        \n        print(f\"   결과: MCC: {mcc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}\")\n        print(f\"   학습 시간: {train_time:.2f}초\")\n        \n        return X_test, y_test\n    \n    def train_autoencoder(self, X, y):\n        \"\"\"\n        AutoEncoder 기반 이상 탐지\n        \"\"\"\n        if not TORCH_AVAILABLE:\n            print(f\"\\n❌ PyTorch가 설치되지 않아 AutoEncoder를 사용할 수 없습니다.\")\n            return None, None\n        \n        print(f\"\\n🧠 AutoEncoder 이상 탐지:\")\n        print(\"-\" * 50)\n        \n        # 데이터 분할\n        X_train, X_test, y_train, y_test = train_test_split(\n            X, y, test_size=0.2, random_state=self.random_state, stratify=y\n        )\n        \n        # 정상 데이터만 사용\n        X_train_normal = X_train[y_train == 0]\n        print(f\"   정상 데이터로 학습: {len(X_train_normal):,}개\")\n        \n        # AutoEncoder 모델 정의\n        class AutoEncoder(nn.Module):\n            def __init__(self, input_dim, encoding_dim=64):\n                super(AutoEncoder, self).__init__()\n                \n                # 인코더\n                self.encoder = nn.Sequential(\n                    nn.Linear(input_dim, input_dim // 2),\n                    nn.ReLU(),\n                    nn.Dropout(0.2),\n                    nn.Linear(input_dim // 2, input_dim // 4),\n                    nn.ReLU(),\n                    nn.Dropout(0.2),\n                    nn.Linear(input_dim // 4, encoding_dim),\n                    nn.ReLU()\n                )\n                \n                # 디코더\n                self.decoder = nn.Sequential(\n                    nn.Linear(encoding_dim, input_dim // 4),\n                    nn.ReLU(),\n                    nn.Dropout(0.2),\n                    nn.Linear(input_dim // 4, input_dim // 2),\n                    nn.ReLU(),\n                    nn.Dropout(0.2),\n                    nn.Linear(input_dim // 2, input_dim)\n                )\n            \n            def forward(self, x):\n                encoded = self.encoder(x)\n                decoded = self.decoder(encoded)\n                return decoded\n        \n        # 모델 생성\n        input_dim = X_train.shape[1]\n        model = AutoEncoder(input_dim, encoding_dim=32)\n        \n        # 디바이스 설정\n        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n        model = model.to(device)\n        \n        print(f\"   디바이스: {device}\")\n        print(f\"   입력 차원: {input_dim}\")\n        \n        # 학습 설정\n        criterion = nn.MSELoss()\n        optimizer = optim.Adam(model.parameters(), lr=0.001)\n        \n        # 데이터 로더\n        train_dataset = TensorDataset(\n            torch.FloatTensor(X_train_normal),\n            torch.FloatTensor(X_train_normal)  # AutoEncoder는 입력=출력\n        )\n        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)\n        \n        # 학습\n        model.train()\n        epochs = 50\n        print(f\"   에포크: {epochs}\")\n        \n        start_time = time.time()\n        \n        for epoch in range(epochs):\n            epoch_loss = 0\n            for batch_x, batch_y in train_loader:\n                batch_x = batch_x.to(device)\n                batch_y = batch_y.to(device)\n                \n                optimizer.zero_grad()\n                outputs = model(batch_x)\n                loss = criterion(outputs, batch_y)\n                loss.backward()\n                optimizer.step()\n                \n                epoch_loss += loss.item()\n            \n            if (epoch + 1) % 10 == 0:\n                print(f\"   에포크 {epoch+1}/{epochs}, 손실: {epoch_loss/len(train_loader):.6f}\")\n        \n        train_time = time.time() - start_time\n        \n        # 임계값 계산 (정상 데이터의 재구성 오차)\n        model.eval()\n        with torch.no_grad():\n            X_train_tensor = torch.FloatTensor(X_train_normal).to(device)\n            train_reconstructed = model(X_train_tensor).cpu().numpy()\n            train_errors = np.mean(np.square(X_train_normal - train_reconstructed), axis=1)\n            threshold = np.percentile(train_errors, 95)  # 95퍼센타일 임계값\n        \n        print(f\"   임계값: {threshold:.6f}\")\n        \n        # 테스트 데이터로 예측\n        with torch.no_grad():\n            X_test_tensor = torch.FloatTensor(X_test).to(device)\n            test_reconstructed = model(X_test_tensor).cpu().numpy()\n            test_errors = np.mean(np.square(X_test - test_reconstructed), axis=1)\n        \n        # 이상 탐지\n        y_pred = (test_errors > threshold).astype(int)\n        \n        # 평가\n        mcc = matthews_corrcoef(y_test, y_pred)\n        f1 = f1_score(y_test, y_pred)\n        auc = roc_auc_score(y_test, test_errors)\n        \n        self.models['AutoEncoder'] = model\n        self.results['AutoEncoder'] = {\n            'MCC': mcc,\n            'F1': f1,\n            'AUC': auc,\n            'train_time': train_time,\n            'predictions': y_pred,\n            'reconstruction_errors': test_errors,\n            'threshold': threshold\n        }\n        \n        print(f\"   결과: MCC: {mcc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}\")\n        print(f\"   학습 시간: {train_time:.2f}초\")\n        \n        return X_test, y_test\n    \n    def print_summary(self):\n        \"\"\"\n        전체 결과 요약\n        \"\"\"\n        print(f\"\\n\" + \"=\" * 80)\n        print(\"📋 모델 성능 요약\")\n        print(\"=\" * 80)\n        \n        if not self.results:\n            print(\"학습된 모델이 없습니다.\")\n            return\n        \n        # 결과 정리\n        summary_df = pd.DataFrame()\n        for name, result in self.results.items():\n            summary_df = pd.concat([summary_df, pd.DataFrame({\n                'Model': [name],\n                'MCC': [result['MCC']],\n                'F1-Score': [result['F1']],\n                'AUC': [result['AUC']],\n                'Train Time (s)': [result['train_time']]\n            })], ignore_index=True)\n        \n        # 정렬 (MCC 기준)\n        summary_df = summary_df.sort_values('MCC', ascending=False)\n        \n        print(summary_df.to_string(index=False, float_format='%.4f'))\n        \n        # 최고 성능 모델\n        best_model = summary_df.iloc[0]['Model']\n        best_mcc = summary_df.iloc[0]['MCC']\n        \n        print(f\"\\n🏆 최고 성능: {best_model} (MCC: {best_mcc:.4f})\")\n        \n        # 상세 분석 (최고 성능 모델)\n        if best_model in self.results:\n            result = self.results[best_model]\n            if 'predictions' in result:\n                predictions = result['predictions']\n                print(f\"\\n📊 {best_model} 상세 결과:\")\n                print(f\"   예측된 불량품 수: {predictions.sum():,}개\")\n                print(f\"   불량 예측률: {predictions.mean():.2%}\")\n\ndef main():\n    \"\"\"\n    메인 실행 함수\n    \"\"\"\n    print(\"🏭 Bosch Production Line - 종합 불량 검출 시스템\")\n    print(\"=\" * 80)\n    \n    try:\n        # 데이터 경로\n        data_path = \"C:/Users/ASUS/bosch/data/train_numeric.csv\"\n        \n        # 불량 검출 시스템 초기화\n        detector = BoschFaultDetectionSuite(sampling_strategy='hybrid')\n        \n        # 1. 데이터 로드 및 전처리\n        X, y = detector.load_and_preprocess_data(data_path, sample_size=100000)\n        \n        # 2. 지도학습 모델 학습\n        print(f\"\\n\" + \"=\" * 80)\n        print(\"🎯 지도학습 접근법 (Binary Classification)\")\n        print(\"=\" * 80)\n        X_test, y_test = detector.train_supervised_models(X, y)\n        \n        # 3. Isolation Forest\n        print(f\"\\n\" + \"=\" * 80)\n        print(\"🌲 비지도학습 접근법 (Anomaly Detection)\")\n        print(\"=\" * 80)\n        detector.train_isolation_forest(X, y)\n        \n        # 4. AutoEncoder\n        print(f\"\\n\" + \"=\" * 80)\n        print(\"🧠 딥러닝 접근법 (AutoEncoder)\")\n        print(\"=\" * 80)\n        detector.train_autoencoder(X, y)\n        \n        # 5. 결과 요약\n        detector.print_summary()\n        \n        print(f\"\\n✅ 모든 모델 학습 완료!\")\n        print(f\"\\n💡 권장사항:\")\n        print(f\"   1. 더 많은 데이터로 재학습\")\n        print(f\"   2. 하이퍼파라미터 튜닝 (Optuna 등 활용)\")\n        print(f\"   3. 앙상블 모델 구성\")\n        print(f\"   4. 날짜/범주형 데이터 추가 활용\")\n        \n    except Exception as e:\n        print(f\"❌ 오류 발생: {str(e)}\")\n        import traceback\n        traceback.print_exc()\n\nif __name__ == \"__main__\":\n    main()