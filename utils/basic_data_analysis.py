#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bosch Production Line Performance - 기본 데이터 분석 (시각화 없음)
"""
import pandas as pd
import numpy as np
import warnings
import gc
import os

warnings.filterwarnings('ignore')

def analyze_data_overview():
    """데이터 전체 개요 분석"""
    print("=" * 60)
    print("🏭 BOSCH PRODUCTION LINE PERFORMANCE 실제 데이터 분석")
    print("=" * 60)
    
    data_dir = "C:/Users/ASUS/bosch/data/"
    
    # 파일 정보 수집
    files = {
        'train_numeric': f'{data_dir}train_numeric.csv',
        'train_categorical': f'{data_dir}train_categorical.csv', 
        'train_date': f'{data_dir}train_date.csv',
        'test_numeric': f'{data_dir}test_numeric.csv',
        'test_categorical': f'{data_dir}test_categorical.csv',
        'test_date': f'{data_dir}test_date.csv'
    }
    
    print("\n📊 데이터 파일 정보:")
    print("-" * 50)
    
    for name, path in files.items():
        try:
            size_mb = os.path.getsize(path) / (1024*1024)
            
            # 첫 줄만 읽어서 컬럼 수 확인
            header = pd.read_csv(path, nrows=0)
            num_cols = len(header.columns)
            
            # 빠른 행 수 계산
            chunk_size = 10000
            total_rows = 0
            for chunk in pd.read_csv(path, chunksize=chunk_size):
                total_rows += len(chunk)
                break  # 첫 번째 청크만으로 추정
            
            # 실제로는 전체 행 수를 정확히 계산하지만 시간이 오래 걸림
            # 첫 청크로 추정
            with open(path, 'r') as f:
                estimated_rows = sum(1 for _ in f) - 1  # 헤더 제외
            
            print(f"{name:20s}: {size_mb:7.1f} MB | ~{estimated_rows:7,} rows | {num_cols:4,} cols")
            
        except Exception as e:
            print(f"{name:20s}: Error - {str(e)}")
    
    return files

def analyze_train_numeric_detailed(data_dir, sample_size=50000):
    """숫자 데이터 상세 분석"""
    print(f"\n🔢 TRAIN NUMERIC 데이터 상세 분석")
    print("=" * 50)
    
    file_path = f'{data_dir}train_numeric.csv'
    print(f"샘플 크기: {sample_size:,}개 (메모리 절약을 위해)")
    
    # 샘플 데이터 로드
    print("\n⏳ 데이터 로딩 중...")
    df = pd.read_csv(file_path, nrows=sample_size)
    
    print(f"✅ 로드 완료: {df.shape[0]:,} x {df.shape[1]:,}")
    
    # 1. 기본 정보
    print(f"\n📋 기본 정보:")
    print(f"   • 제품 수: {df.shape[0]:,}개")
    print(f"   • 전체 특징 수: {df.shape[1]:,}개")
    print(f"   • 실제 측정 특징 수: {df.shape[1] - 2:,}개 (Id, Response 제외)")
    
    # 2. 목표 변수 분석
    print(f"\n🎯 목표 변수 (Response) 분석:")
    response_counts = df['Response'].value_counts().sort_index()
    failure_rate = df['Response'].mean()
    
    print(f"   • 정상품 (0): {response_counts[0]:,}개 ({(1-failure_rate)*100:.2f}%)")
    print(f"   • 불량품 (1): {response_counts[1]:,}개 ({failure_rate*100:.2f}%)")
    print(f"   • 불량률: {failure_rate:.4%}")
    print(f"   • 불균형 비율: 1:{int((1-failure_rate)/failure_rate)}")
    
    # 3. 결측값 상세 분석
    print(f"\n❓ 결측값 상세 분석:")
    missing_stats = df.isnull().sum()
    total_features = len(df.columns) - 2  # Id, Response 제외
    
    # 결측값 비율별 분류
    completely_empty = (missing_stats == len(df)).sum()
    over_95_empty = (missing_stats > len(df) * 0.95).sum()
    over_90_empty = (missing_stats > len(df) * 0.90).sum() 
    over_50_empty = (missing_stats > len(df) * 0.50).sum()
    over_10_empty = (missing_stats > len(df) * 0.10).sum()
    no_missing = (missing_stats == 0).sum()
    
    print(f"   전체 특징 수: {total_features:,}개")
    print(f"   완전히 비어있는 컬럼: {completely_empty:,}개 ({completely_empty/total_features*100:.1f}%)")
    print(f"   95% 이상 비어있는 컬럼: {over_95_empty:,}개 ({over_95_empty/total_features*100:.1f}%)")
    print(f"   90% 이상 비어있는 컬럼: {over_90_empty:,}개 ({over_90_empty/total_features*100:.1f}%)")
    print(f"   50% 이상 비어있는 컬럼: {over_50_empty:,}개 ({over_50_empty/total_features*100:.1f}%)")
    print(f"   10% 이상 비어있는 컬럼: {over_10_empty:,}개 ({over_10_empty/total_features*100:.1f}%)")
    print(f"   결측값이 전혀 없는 컬럼: {no_missing:,}개 ({no_missing/total_features*100:.1f}%)")
    
    # 4. 특징 그룹 분석 (스테이션별)
    print(f"\n🏭 스테이션/라인별 특징 분석:")
    feature_groups = {}
    
    for col in df.columns:
        if col not in ['Id', 'Response']:
            parts = col.split('_')
            if len(parts) >= 2:
                group = parts[0] + '_' + parts[1]
                if group not in feature_groups:
                    feature_groups[group] = []
                feature_groups[group].append(col)
    
    # 상위 15개 그룹
    sorted_groups = sorted(feature_groups.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"   총 {len(feature_groups)}개의 측정 스테이션 발견")
    print("   상위 15개 스테이션:")
    
    for i, (group, features) in enumerate(sorted_groups[:15], 1):
        # 각 그룹의 결측률 계산
        group_missing_rate = df[features].isnull().sum().sum() / (len(features) * len(df)) * 100
        print(f"   {i:2d}. {group}: {len(features):3d}개 특징 (결측률: {group_missing_rate:.1f}%)")
    
    return df, feature_groups

def create_engineered_features(df):
    """특징 공학"""
    print(f"\n🛠️ 특징 공학:")
    print("-" * 30)
    
    feature_df = pd.DataFrame()
    feature_df['Id'] = df['Id']
    
    # 숫자 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Id', 'Response'], errors='ignore')
    print(f"   처리할 숫자 특징: {len(numeric_cols):,}개")
    
    # 기본 집계 특징들
    print("   ⏳ 기본 통계 특징 생성 중...")
    feature_df['count_non_null'] = df[numeric_cols].count(axis=1)
    feature_df['count_zeros'] = (df[numeric_cols] == 0).sum(axis=1) 
    feature_df['missing_count'] = df[numeric_cols].isnull().sum(axis=1)
    feature_df['missing_ratio'] = feature_df['missing_count'] / len(numeric_cols)
    
    # 값이 있는 행에 대한 통계
    print("   ⏳ 통계량 계산 중...")
    feature_df['mean'] = df[numeric_cols].mean(axis=1, skipna=True)
    feature_df['std'] = df[numeric_cols].std(axis=1, skipna=True)
    feature_df['min'] = df[numeric_cols].min(axis=1, skipna=True)
    feature_df['max'] = df[numeric_cols].max(axis=1, skipna=True)
    feature_df['range'] = feature_df['max'] - feature_df['min']
    feature_df['median'] = df[numeric_cols].median(axis=1, skipna=True)
    feature_df['q25'] = df[numeric_cols].quantile(0.25, axis=1, numeric_only=True)
    feature_df['q75'] = df[numeric_cols].quantile(0.75, axis=1, numeric_only=True)
    feature_df['iqr'] = feature_df['q75'] - feature_df['q25']
    
    # 스테이션별 특징 (주요 스테이션만)
    print("   ⏳ 스테이션별 특징 생성 중...")
    station_groups = {}
    for col in numeric_cols:
        parts = col.split('_')
        if len(parts) >= 2:
            station = parts[0] + '_' + parts[1]
            if station not in station_groups:
                station_groups[station] = []
            station_groups[station].append(col)
    
    # 상위 10개 스테이션의 통계
    top_stations = sorted(station_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    for station, cols in top_stations:
        if len(cols) > 1:  # 특징이 2개 이상인 경우만
            station_data = df[cols]
            feature_df[f'{station}_mean'] = station_data.mean(axis=1, skipna=True)
            feature_df[f'{station}_count'] = station_data.count(axis=1)
            feature_df[f'{station}_std'] = station_data.std(axis=1, skipna=True)
    
    # 목표 변수 추가
    if 'Response' in df.columns:
        feature_df['Response'] = df['Response']
    
    print(f"   ✅ 완료! 총 {len(feature_df.columns)}개 특징 생성")
    
    return feature_df

def analyze_feature_importance(feature_df):
    """특징 중요도 분석 (상관관계 기반)"""
    print(f"\n📊 특징 중요도 분석:")
    print("-" * 40)
    
    if 'Response' not in feature_df.columns:
        print("목표 변수가 없어서 중요도 분석을 할 수 없습니다.")
        return None
    
    # 상관계수 계산
    feature_cols = [col for col in feature_df.columns if col not in ['Id', 'Response']]
    
    print(f"   분석할 특징 수: {len(feature_cols)}개")
    
    correlations = {}
    for col in feature_cols:
        try:
            corr = feature_df[col].corr(feature_df['Response'])
            if not np.isnan(corr):
                correlations[col] = corr
        except:
            continue
    
    # 절댓값 기준으로 정렬
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n   상위 20개 중요 특징 (상관계수 기준):")
    print("   " + "-" * 50)
    
    for i, (feature, corr) in enumerate(sorted_corr[:20], 1):
        direction = "↗️ 양의" if corr > 0 else "↘️ 음의"
        print(f"   {i:2d}. {feature:25s}: {corr:+.5f} ({direction})")
    
    # 그룹별 중요도 (평균)
    print(f"\n   스테이션별 평균 중요도:")
    print("   " + "-" * 40)
    
    station_importance = {}
    for feature, corr in correlations.items():
        if '_' in feature and feature.count('_') >= 1:
            parts = feature.split('_')
            if len(parts) >= 2:
                station = parts[0] + '_' + parts[1]
                if station not in station_importance:
                    station_importance[station] = []
                station_importance[station].append(abs(corr))
    
    # 평균 중요도 계산
    station_avg_importance = {
        station: np.mean(corrs) 
        for station, corrs in station_importance.items() 
        if len(corrs) > 0
    }
    
    sorted_stations = sorted(station_avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (station, avg_importance) in enumerate(sorted_stations[:10], 1):
        feature_count = len(station_importance.get(station, []))
        print(f"   {i:2d}. {station:15s}: {avg_importance:.5f} ({feature_count}개 특징)")
    
    return sorted_corr

def compare_failure_vs_normal(feature_df):
    """정상품 vs 불량품 비교 분석"""
    print(f"\n⚖️ 정상품 vs 불량품 비교 분석:")
    print("-" * 50)
    
    if 'Response' not in feature_df.columns:
        print("목표 변수가 없어서 비교 분석을 할 수 없습니다.")
        return
    
    normal = feature_df[feature_df['Response'] == 0]
    failure = feature_df[feature_df['Response'] == 1]
    
    print(f"   정상품: {len(normal):,}개")
    print(f"   불량품: {len(failure):,}개")
    
    # 주요 특징들 비교
    key_features = ['count_non_null', 'missing_ratio', 'mean', 'std', 'range']
    
    print(f"\n   주요 특징 비교:")
    print(f"   {'특징':15s} {'정상품 평균':>12s} {'불량품 평균':>12s} {'차이':>12s} {'비율':>10s}")
    print("   " + "-" * 70)
    
    for feature in key_features:
        if feature in feature_df.columns:
            normal_mean = normal[feature].mean()
            failure_mean = failure[feature].mean()
            diff = failure_mean - normal_mean
            ratio = failure_mean / normal_mean if normal_mean != 0 else float('inf')
            
            print(f"   {feature:15s} {normal_mean:12.3f} {failure_mean:12.3f} {diff:+12.3f} {ratio:10.3f}")
    
    # 통계적 유의성 (간단한 t-test 대신 평균 차이로 판단)
    print(f"\n   💡 주요 차이점:")
    
    # 측정 개수 차이
    normal_count = normal['count_non_null'].mean()
    failure_count = failure['count_non_null'].mean()
    if abs(failure_count - normal_count) > normal_count * 0.01:  # 1% 이상 차이
        direction = "더 많이" if failure_count > normal_count else "더 적게"
        print(f"   • 불량품은 정상품보다 측정을 {direction} 받음 ({failure_count:.1f} vs {normal_count:.1f})")
    
    # 결측률 차이
    normal_missing = normal['missing_ratio'].mean()
    failure_missing = failure['missing_ratio'].mean()
    if abs(failure_missing - normal_missing) > 0.01:  # 1%p 이상 차이
        direction = "높음" if failure_missing > normal_missing else "낮음"
        print(f"   • 불량품의 결측률이 {direction} ({failure_missing:.1%} vs {normal_missing:.1%})")
    
    # 측정값 차이
    normal_mean_val = normal['mean'].mean()
    failure_mean_val = failure['mean'].mean()
    if not np.isnan(normal_mean_val) and not np.isnan(failure_mean_val):
        if abs(failure_mean_val - normal_mean_val) > abs(normal_mean_val * 0.05):  # 5% 이상 차이
            direction = "높음" if failure_mean_val > normal_mean_val else "낮음"
            print(f"   • 불량품의 측정값 평균이 {direction} ({failure_mean_val:.3f} vs {normal_mean_val:.3f})")

def main():
    """메인 분석 함수"""
    try:
        start_time = pd.Timestamp.now()
        
        # 1. 전체 개요
        print("🚀 분석 시작...")
        files = analyze_data_overview()
        
        # 2. 숫자 데이터 상세 분석
        data_dir = "C:/Users/ASUS/bosch/data/"
        df, feature_groups = analyze_train_numeric_detailed(data_dir, sample_size=100000)
        
        # 3. 특징 공학
        feature_df = create_engineered_features(df)
        
        # 4. 특징 중요도 분석
        correlations = analyze_feature_importance(feature_df)
        
        # 5. 정상품 vs 불량품 비교
        compare_failure_vs_normal(feature_df)
        
        # 6. 최종 요약
        end_time = pd.Timestamp.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n" + "="*70)
        print("📋 최종 분석 결과 요약")
        print("="*70)
        
        failure_rate = df['Response'].mean()
        total_features = len(df.columns) - 2
        useful_features = len([col for col in df.columns if df[col].isnull().sum() < len(df) * 0.9])
        
        print(f"🔍 데이터셋 특성:")
        print(f"   • 분석 샘플: {df.shape[0]:,} x {df.shape[1]:,}")
        print(f"   • 불량률: {failure_rate:.4%} (극도로 불균형)")
        print(f"   • 전체 특징: {total_features:,}개")
        print(f"   • 유용한 특징: {useful_features:,}개 (90% 미만 결측)")
        print(f"   • 측정 스테이션: {len(feature_groups)}개")
        
        print(f"\n🛠️ 특징 공학 결과:")
        print(f"   • 생성된 집계 특징: {len(feature_df.columns)}개")
        
        if correlations and len(correlations) > 0:
            max_corr = max(abs(corr) for _, corr in correlations[:10])
            print(f"   • 최고 상관계수: {max_corr:.4f}")
        
        print(f"\n💡 핵심 인사이트:")
        print(f"   1. 극도로 불균형한 데이터 (불량률 {failure_rate:.2%})")
        print(f"   2. 대부분의 특징이 희소함 (90% 이상 결측)")
        print(f"   3. 결측값 패턴 자체가 중요한 정보원")
        print(f"   4. 집계 특징(count, mean 등)이 예측에 유용")
        print(f"   5. 스테이션별 분석 접근법이 필요")
        
        print(f"\n🎯 추천 모델링 전략:")
        print(f"   • 불균형 처리: SMOTE, 가중치 조정, 임계값 최적화")
        print(f"   • 특징 선택: 상관관계, 분산 기반 필터링")
        print(f"   • 모델: XGBoost, LightGBM (트리 기반 모델)")
        print(f"   • 평가: Matthews Correlation Coefficient (MCC)")
        
        print(f"\n⏱️ 분석 완료 시간: {processing_time:.1f}초")
        
        # 메모리 정리
        del df, feature_df
        gc.collect()
        
        print(f"\n✅ 모든 분석 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()