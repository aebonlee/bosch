#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 Bosch 불량 검출 데모
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report

warnings.filterwarnings('ignore')

def main():
    print("Bosch Production Line - 불량 검출 데모")
    print("=" * 50)
    
    try:
        # 데이터 로드
        data_path = "C:/Users/ASUS/bosch/data/train_numeric.csv"
        print(f"데이터 로딩 중... (샘플: 10,000개)")
        
        df = pd.read_csv(data_path, nrows=10000)
        print(f"로드 완료: {df.shape[0]:,} x {df.shape[1]:,}")
        
        # 기본 정보
        normal_count = (df['Response'] == 0).sum()
        fault_count = (df['Response'] == 1).sum()
        fault_rate = fault_count / len(df)
        
        print(f"\n데이터 분포:")
        print(f"  정상: {normal_count:,}개 ({(1-fault_rate)*100:.1f}%)")
        print(f"  불량: {fault_count:,}개 ({fault_rate*100:.1f}%)")
        
        # 간단한 특징 생성
        print(f"\n특징 생성 중...")
        feature_cols = [col for col in df.columns if col not in ['Id', 'Response']]
        X_raw = df[feature_cols]
        y = df['Response']
        
        # 결측값이 많은 특징 제거 (90% 이상)
        missing_ratio = X_raw.isnull().sum() / len(X_raw)
        valid_features = missing_ratio[missing_ratio < 0.9].index.tolist()
        X_filtered = X_raw[valid_features]
        
        print(f"  원본 특징: {len(feature_cols)}개")
        print(f"  필터링 후: {len(valid_features)}개")
        
        # 결측값 처리
        X_clean = X_filtered.fillna(X_filtered.median())
        
        # 집계 특징 생성
        feature_df = pd.DataFrame()
        feature_df['count_non_null'] = X_clean.count(axis=1)
        feature_df['count_zeros'] = (X_clean == 0).sum(axis=1)
        feature_df['mean'] = X_clean.mean(axis=1, skipna=True)
        feature_df['std'] = X_clean.std(axis=1, skipna=True)
        feature_df['min'] = X_clean.min(axis=1, skipna=True)
        feature_df['max'] = X_clean.max(axis=1, skipna=True)
        feature_df['range'] = feature_df['max'] - feature_df['min']
        
        # NaN 처리
        feature_df = feature_df.fillna(0)
        
        print(f"  최종 특징: {len(feature_df.columns)}개")
        
        # 스케일링
        scaler = RobustScaler()
        X = scaler.fit_transform(feature_df)
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"  학습 데이터: {X_train.shape[0]:,}개")
        print(f"  테스트 데이터: {X_test.shape[0]:,}개")
        
        # 모델 1: Random Forest (지도학습)
        print(f"\n1. Random Forest 학습:")
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            random_state=42,
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        mcc_rf = matthews_corrcoef(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf)
        
        print(f"  MCC: {mcc_rf:.4f}")
        print(f"  F1-Score: {f1_rf:.4f}")
        
        # 모델 2: Isolation Forest (이상 탐지)
        print(f"\n2. Isolation Forest 학습:")
        
        # 정상 데이터만으로 학습
        X_train_normal = X_train[y_train == 0]
        
        iforest = IsolationForest(
            n_estimators=50,
            contamination=fault_rate,
            random_state=42
        )
        
        iforest.fit(X_train_normal)
        y_pred_if_raw = iforest.predict(X_test)
        y_pred_if = (y_pred_if_raw == -1).astype(int)  # -1을 1로 변환
        
        mcc_if = matthews_corrcoef(y_test, y_pred_if)
        f1_if = f1_score(y_test, y_pred_if)
        
        print(f"  MCC: {mcc_if:.4f}")
        print(f"  F1-Score: {f1_if:.4f}")
        
        # 결과 비교
        print(f"\n결과 비교:")
        print(f"=" * 30)
        print(f"{'모델':<15} {'MCC':<8} {'F1':<8}")
        print(f"-" * 30)
        print(f"{'Random Forest':<15} {mcc_rf:<8.4f} {f1_rf:<8.4f}")
        print(f"{'IsolationForest':<15} {mcc_if:<8.4f} {f1_if:<8.4f}")
        
        # 최고 성능 모델
        best_model = "Random Forest" if mcc_rf > mcc_if else "Isolation Forest"
        best_mcc = max(mcc_rf, mcc_if)
        
        print(f"\n최고 성능: {best_model} (MCC: {best_mcc:.4f})")
        
        # 상세 분류 보고서 (최고 성능 모델)
        print(f"\n{best_model} 상세 결과:")
        if best_model == "Random Forest":
            print(classification_report(y_test, y_pred_rf, 
                                      target_names=['정상', '불량'], 
                                      digits=4))
        else:
            print(classification_report(y_test, y_pred_if, 
                                      target_names=['정상', '불량'], 
                                      digits=4))
        
        print(f"\n실행 완료!")
        print(f"\n참고사항:")
        print(f"  - 더 많은 데이터 사용 시 성능 향상 기대")
        print(f"  - 하이퍼파라미터 튜닝으로 추가 개선 가능")
        print(f"  - 앙상블 기법으로 성능 향상 가능")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()