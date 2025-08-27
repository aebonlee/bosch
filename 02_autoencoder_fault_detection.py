#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bosch Production Line - AutoEncoder 기반 이상 탐지 예제
LGES DL AutoEncoder Fault Detection Solution 참고

이 예제는 다음과 같은 접근법을 사용합니다:
1. 정상 제품 데이터로 AutoEncoder 학습
2. 재구성 오차를 통한 이상 탐지
3. 임계값 기반 불량품 분류
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras 임포트 (조건부)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
    print("TensorFlow 사용 가능")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow가 설치되지 않음. 대안 구현을 사용합니다.")

warnings.filterwarnings('ignore')

class BoschAutoEncoderFaultDetector:
    """
    Bosch 생산라인 데이터를 위한 AutoEncoder 기반 이상 탐지기
    """
    
    def __init__(self, encoding_dim=32, learning_rate=0.001):
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.scaler = None
        self.threshold = None
        self.history = None
        
    def prepare_data(self, data_path, sample_size=50000):
        """
        데이터 로드 및 전처리
        
        Args:
            data_path (str): 데이터 파일 경로
            sample_size (int): 메모리 절약을 위한 샘플 크기
        """
        print("=" * 60)
        print("🔧 데이터 준비 및 전처리")
        print("=" * 60)
        
        # 데이터 로드
        print(f"데이터 로딩 중... (샘플 크기: {sample_size:,})")
        df = pd.read_csv(data_path, nrows=sample_size)
        print(f"로드 완료: {df.shape[0]:,} x {df.shape[1]:,}")
        
        # 기본 정보
        normal_count = (df['Response'] == 0).sum()
        fault_count = (df['Response'] == 1).sum()
        fault_rate = fault_count / len(df)
        
        print(f"\n📊 데이터 분포:")
        print(f"   정상 제품: {normal_count:,}개 ({(1-fault_rate)*100:.2f}%)")
        print(f"   불량 제품: {fault_count:,}개 ({fault_rate*100:.2f}%)")
        
        # 특징 선택 및 정리
        print(f"\n🔍 특징 선택 및 정리:")
        
        # ID와 Response 제외
        feature_cols = [col for col in df.columns if col not in ['Id', 'Response']]
        X = df[feature_cols].copy()
        y = df['Response'].copy()
        
        print(f"   원본 특징 수: {len(feature_cols):,}개")
        
        # 결측값이 너무 많은 특징 제거 (95% 이상)
        missing_threshold = 0.95
        missing_ratio = X.isnull().sum() / len(X)
        valid_features = missing_ratio[missing_ratio < missing_threshold].index.tolist()
        
        X_filtered = X[valid_features].copy()
        print(f"   결측값 필터링 후: {len(valid_features):,}개")
        
        # 분산이 0인 특징 제거
        X_filled = X_filtered.fillna(0)  # 임시로 0으로 채움
        variances = X_filled.var()
        non_zero_var_features = variances[variances > 1e-8].index.tolist()
        
        X_final = X_filtered[non_zero_var_features].copy()
        print(f"   분산 필터링 후: {len(non_zero_var_features):,}개")
        
        # 결측값 처리 (중앙값으로 대체)
        print(f"\n🛠️ 결측값 처리:")
        missing_before = X_final.isnull().sum().sum()
        
        # 중앙값으로 결측값 채우기
        X_final = X_final.fillna(X_final.median())
        
        missing_after = X_final.isnull().sum().sum()
        print(f"   처리 전 결측값: {missing_before:,}개")
        print(f"   처리 후 결측값: {missing_after:,}개")
        
        # 무한값 처리
        X_final = X_final.replace([np.inf, -np.inf], np.nan)
        X_final = X_final.fillna(X_final.median())
        
        # 최종 데이터 정보
        print(f"\n✅ 전처리 완료:")
        print(f"   최종 특징 수: {X_final.shape[1]:,}개")
        print(f"   샘플 수: {X_final.shape[0]:,}개")
        
        # 특징 스케일링 (RobustScaler 사용 - 이상값에 강함)
        print(f"\n📏 특징 스케일링 (RobustScaler):")
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_final)
        
        print(f"   스케일링 완료: 평균≈0, 중앙값 기준 정규화")
        
        # 정상 데이터와 불량 데이터 분리
        normal_indices = y == 0
        fault_indices = y == 1
        
        X_normal = X_scaled[normal_indices]
        X_fault = X_scaled[fault_indices]
        
        print(f"\n🎯 데이터 분리:")
        print(f"   정상 데이터: {X_normal.shape[0]:,} x {X_normal.shape[1]:,}")
        print(f"   불량 데이터: {X_fault.shape[0]:,} x {X_fault.shape[1]:,}")
        
        return X_normal, X_fault, X_scaled, y
    
    def build_autoencoder(self, input_dim):
        """
        AutoEncoder 모델 구축
        
        Args:
            input_dim (int): 입력 차원
        """
        if not TF_AVAILABLE:
            print("❌ TensorFlow가 없어서 AutoEncoder를 구축할 수 없습니다.")
            return None
        
        print(f"\n🏗️ AutoEncoder 모델 구축:")
        print(f"   입력 차원: {input_dim}")
        print(f"   인코딩 차원: {self.encoding_dim}")
        
        # 입력층
        input_layer = layers.Input(shape=(input_dim,))
        
        # 인코더
        # 점진적으로 차원 축소
        encoder_layers = input_layer
        layer_dims = [int(input_dim * 0.75), int(input_dim * 0.5), 
                     int(input_dim * 0.25), self.encoding_dim]
        
        print(f"   인코더 구조:")
        for i, dim in enumerate(layer_dims):
            encoder_layers = layers.Dense(dim, activation='relu', 
                                        name=f'encoder_{i+1}')(encoder_layers)
            encoder_layers = layers.Dropout(0.2)(encoder_layers)
            print(f"     레이어 {i+1}: {dim}개 뉴런")
        
        # 인코더 모델
        self.encoder = Model(input_layer, encoder_layers, name='encoder')
        
        # 디코더
        # 점진적으로 차원 복원
        decoder_input = layers.Input(shape=(self.encoding_dim,))
        decoder_layers = decoder_input
        
        decode_dims = layer_dims[:-1][::-1] + [input_dim]  # 역순 + 원본 차원
        
        print(f"   디코더 구조:")
        for i, dim in enumerate(decode_dims):
            activation = 'relu' if i < len(decode_dims) - 1 else 'linear'
            decoder_layers = layers.Dense(dim, activation=activation,
                                        name=f'decoder_{i+1}')(decoder_layers)
            if i < len(decode_dims) - 1:  # 마지막 층에는 Dropout 적용 안함
                decoder_layers = layers.Dropout(0.2)(decoder_layers)
            print(f"     레이어 {i+1}: {dim}개 뉴런 ({activation})")
        
        # 디코더 모델
        self.decoder = Model(decoder_input, decoder_layers, name='decoder')
        
        # 전체 AutoEncoder
        encoded = self.encoder(input_layer)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(input_layer, decoded, name='autoencoder')
        
        # 컴파일
        optimizer = Adam(learning_rate=self.learning_rate)
        self.autoencoder.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print(f"\n📋 모델 요약:")
        print(f"   총 파라미터: {self.autoencoder.count_params():,}개")
        
        return self.autoencoder
    
    def train_autoencoder(self, X_normal, validation_split=0.2, epochs=100, batch_size=256):
        """
        정상 데이터로 AutoEncoder 학습
        
        Args:
            X_normal (np.array): 정상 데이터
            validation_split (float): 검증 데이터 비율
            epochs (int): 학습 에포크
            batch_size (int): 배치 크기
        """
        if not TF_AVAILABLE or self.autoencoder is None:
            print("❌ AutoEncoder 모델이 준비되지 않았습니다.")
            return None
        
        print(f"\n🚀 AutoEncoder 학습 시작:")
        print(f"   학습 데이터: {X_normal.shape[0]:,}개")
        print(f"   검증 분할: {validation_split*100:.0f}%")
        print(f"   에포크: {epochs}")
        print(f"   배치 크기: {batch_size}")
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 학습 (정상 데이터로만)
        # AutoEncoder는 입력을 출력으로 재구성하도록 학습
        self.history = self.autoencoder.fit(
            X_normal, X_normal,  # 입력과 출력이 같음
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✅ 학습 완료!")
        
        # 최종 손실값
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        print(f"   최종 학습 손실: {final_loss:.6f}")
        print(f"   최종 검증 손실: {final_val_loss:.6f}")
        
        return self.history
    
    def calculate_threshold(self, X_normal, percentile=95):
        """
        이상 탐지 임계값 계산
        
        Args:
            X_normal (np.array): 정상 데이터
            percentile (float): 임계값 백분위수
        """
        if not TF_AVAILABLE or self.autoencoder is None:
            print("❌ 학습된 AutoEncoder 모델이 없습니다.")
            return None
        
        print(f"\n🎯 이상 탐지 임계값 계산:")
        
        # 정상 데이터의 재구성 오차 계산
        normal_predictions = self.autoencoder.predict(X_normal, verbose=0)
        normal_errors = np.mean(np.square(X_normal - normal_predictions), axis=1)
        
        # 임계값 설정 (백분위수 기준)
        self.threshold = np.percentile(normal_errors, percentile)
        
        print(f"   정상 데이터 재구성 오차:")
        print(f"     평균: {normal_errors.mean():.6f}")
        print(f"     표준편차: {normal_errors.std():.6f}")
        print(f"     최솟값: {normal_errors.min():.6f}")
        print(f"     최댓값: {normal_errors.max():.6f}")
        print(f"   임계값 ({percentile}%): {self.threshold:.6f}")
        
        return self.threshold, normal_errors
    
    def detect_anomalies(self, X_test, y_test=None):
        """
        이상 탐지 수행
        
        Args:
            X_test (np.array): 테스트 데이터
            y_test (np.array): 실제 레이블 (평가용)
        """
        if not TF_AVAILABLE or self.autoencoder is None or self.threshold is None:
            print("❌ 모델이나 임계값이 준비되지 않았습니다.")
            return None
        
        print(f"\n🔍 이상 탐지 수행:")
        print(f"   테스트 데이터: {X_test.shape[0]:,}개")
        
        # 재구성 오차 계산
        test_predictions = self.autoencoder.predict(X_test, verbose=0)
        test_errors = np.mean(np.square(X_test - test_predictions), axis=1)
        
        # 임계값 기준 이상 탐지
        anomaly_predictions = (test_errors > self.threshold).astype(int)
        
        print(f"   재구성 오차 통계:")
        print(f"     평균: {test_errors.mean():.6f}")
        print(f"     표준편차: {test_errors.std():.6f}")
        print(f"     임계값 초과: {anomaly_predictions.sum():,}개 ({anomaly_predictions.mean()*100:.2f}%)")
        
        # 실제 레이블이 있으면 평가 수행
        if y_test is not None:
            print(f"\n📊 성능 평가:")
            
            # 분류 보고서
            print("   분류 보고서:")
            report = classification_report(y_test, anomaly_predictions, 
                                         target_names=['정상', '불량'], 
                                         digits=4)
            print(report)
            
            # 혼동 행렬
            cm = confusion_matrix(y_test, anomaly_predictions)
            print(f"   혼동 행렬:")
            print(f"             예측")
            print(f"        정상    불량")
            print(f"실제 정상 {cm[0,0]:5d}  {cm[0,1]:5d}")
            print(f"    불량 {cm[1,0]:5d}  {cm[1,1]:5d}")
            
            # AUC 계산
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, test_errors)
                print(f"   AUC (재구성 오차 기준): {auc:.4f}")
            
            # 매튜스 상관계수 (MCC)
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(y_test, anomaly_predictions)
            print(f"   Matthews 상관계수: {mcc:.4f}")
            
        return anomaly_predictions, test_errors

def create_simple_example():
    """
    TensorFlow가 없을 때 사용할 간단한 예제
    """
    print("=" * 60)
    print("📋 간단한 이상 탐지 예제 (통계적 방법)")
    print("=" * 60)
    
    data_path = "C:/Users/ASUS/bosch/data/train_numeric.csv"
    
    # 데이터 로드
    print("데이터 로딩 중...")
    df = pd.read_csv(data_path, nrows=10000)
    
    # 간단한 특징 생성
    feature_cols = [col for col in df.columns if col not in ['Id', 'Response']]
    X = df[feature_cols].fillna(0)
    y = df['Response']
    
    # 기본 통계 특징
    X_simple = pd.DataFrame({
        'mean': X.mean(axis=1),
        'std': X.std(axis=1),
        'min': X.min(axis=1),
        'max': X.max(axis=1),
        'non_zero_count': (X != 0).sum(axis=1),
        'zero_ratio': (X == 0).sum(axis=1) / len(feature_cols)
    })
    
    # 정상/불량 분리
    X_normal = X_simple[y == 0]
    X_fault = X_simple[y == 1]
    
    print(f"정상 데이터: {len(X_normal)}개")
    print(f"불량 데이터: {len(X_fault)}개")
    
    # 간단한 이상 탐지 (Isolation Forest 대신 통계적 방법)
    # 각 특징의 정상 범위 계산 (평균 ± 3*표준편차)
    print("\n통계적 이상 탐지:")
    
    anomaly_scores = []
    for _, row in X_simple.iterrows():
        score = 0
        for col in X_simple.columns:
            normal_mean = X_normal[col].mean()
            normal_std = X_normal[col].std()
            
            # z-score 계산
            if normal_std > 0:
                z_score = abs((row[col] - normal_mean) / normal_std)
                if z_score > 3:  # 3-sigma 규칙
                    score += z_score
        
        anomaly_scores.append(score)
    
    # 임계값 기준 분류
    threshold = np.percentile(anomaly_scores, 95)
    predictions = np.array(anomaly_scores) > threshold
    
    print(f"임계값: {threshold:.4f}")
    print(f"이상 예측: {predictions.sum()}개")
    
    # 간단한 평가
    from sklearn.metrics import classification_report
    print("\n성능 평가:")
    print(classification_report(y, predictions.astype(int), 
                              target_names=['정상', '불량'], 
                              digits=4))

def main():
    """
    메인 실행 함수
    """
    print("🏭 Bosch AutoEncoder 기반 이상 탐지 시작")
    
    try:
        # 데이터 경로
        data_path = "C:/Users/ASUS/bosch/data/train_numeric.csv"
        
        if TF_AVAILABLE:
            # TensorFlow 사용 가능한 경우
            detector = BoschAutoEncoderFaultDetector(
                encoding_dim=64,
                learning_rate=0.001
            )
            
            # 데이터 준비
            X_normal, X_fault, X_all, y_all = detector.prepare_data(
                data_path, sample_size=100000
            )
            
            # AutoEncoder 구축
            detector.build_autoencoder(X_normal.shape[1])
            
            # 학습 (정상 데이터만 사용)
            detector.train_autoencoder(
                X_normal, 
                validation_split=0.2,
                epochs=50,
                batch_size=512
            )
            
            # 임계값 계산
            threshold, normal_errors = detector.calculate_threshold(
                X_normal, percentile=95
            )
            
            # 전체 데이터로 이상 탐지
            predictions, test_errors = detector.detect_anomalies(X_all, y_all)
            
            print(f"\n🎉 AutoEncoder 기반 이상 탐지 완료!")
            
        else:
            # TensorFlow가 없는 경우 간단한 예제 실행
            create_simple_example()
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print("\n간단한 예제로 대체 실행:")
        create_simple_example()

if __name__ == "__main__":
    main()