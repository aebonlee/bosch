#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import gc
import os

warnings.filterwarnings('ignore')

def analyze_data_overview():
    print("=" * 60)
    print("BOSCH PRODUCTION LINE PERFORMANCE - 실제 데이터 분석")
    print("=" * 60)
    
    data_dir = "C:/Users/ASUS/bosch/data/"
    
    files = {
        'train_numeric': f'{data_dir}train_numeric.csv',
        'train_categorical': f'{data_dir}train_categorical.csv', 
        'train_date': f'{data_dir}train_date.csv',
        'test_numeric': f'{data_dir}test_numeric.csv',
        'test_categorical': f'{data_dir}test_categorical.csv',
        'test_date': f'{data_dir}test_date.csv'
    }
    
    print("\n데이터 파일 정보:")
    print("-" * 50)
    
    for name, path in files.items():
        try:
            size_mb = os.path.getsize(path) / (1024*1024)
            header = pd.read_csv(path, nrows=0)
            num_cols = len(header.columns)
            
            with open(path, 'r') as f:
                estimated_rows = sum(1 for _ in f) - 1
            
            print(f"{name:20s}: {size_mb:7.1f} MB | {estimated_rows:8,} rows | {num_cols:4,} cols")
            
        except Exception as e:
            print(f"{name:20s}: Error - {str(e)}")
    
    return files

def analyze_train_numeric(data_dir, sample_size=50000):
    print(f"\nTRAIN NUMERIC 데이터 분석 (샘플: {sample_size:,}개)")
    print("=" * 50)
    
    file_path = f'{data_dir}train_numeric.csv'
    
    print("데이터 로딩 중...")
    df = pd.read_csv(file_path, nrows=sample_size)
    print(f"로드 완료: {df.shape[0]:,} x {df.shape[1]:,}")
    
    # 기본 정보
    print(f"\n기본 정보:")
    print(f"   제품 수: {df.shape[0]:,}개")
    print(f"   전체 특징 수: {df.shape[1]:,}개")
    print(f"   실제 측정 특징 수: {df.shape[1] - 2:,}개 (Id, Response 제외)")
    
    # 목표 변수 분석
    print(f"\n목표 변수 (Response) 분석:")
    response_counts = df['Response'].value_counts().sort_index()
    failure_rate = df['Response'].mean()
    
    print(f"   정상품 (0): {response_counts[0]:,}개 ({(1-failure_rate)*100:.2f}%)")
    print(f"   불량품 (1): {response_counts[1]:,}개 ({failure_rate*100:.2f}%)")
    print(f"   불량률: {failure_rate:.4%}")
    print(f"   불균형 비율: 1:{int((1-failure_rate)/failure_rate)}")
    
    # 결측값 분석
    print(f"\n결측값 분석:")
    missing_stats = df.isnull().sum()
    total_features = len(df.columns) - 2
    
    completely_empty = (missing_stats == len(df)).sum()
    over_95_empty = (missing_stats > len(df) * 0.95).sum()
    over_90_empty = (missing_stats > len(df) * 0.90).sum() 
    over_50_empty = (missing_stats > len(df) * 0.50).sum()
    no_missing = (missing_stats == 0).sum()
    
    print(f"   전체 특징 수: {total_features:,}개")
    print(f"   완전히 비어있는 컬럼: {completely_empty:,}개 ({completely_empty/total_features*100:.1f}%)")
    print(f"   95% 이상 비어있는 컬럼: {over_95_empty:,}개 ({over_95_empty/total_features*100:.1f}%)")
    print(f"   90% 이상 비어있는 컬럼: {over_90_empty:,}개 ({over_90_empty/total_features*100:.1f}%)")
    print(f"   50% 이상 비어있는 컬럼: {over_50_empty:,}개 ({over_50_empty/total_features*100:.1f}%)")
    print(f"   결측값이 없는 컬럼: {no_missing:,}개 ({no_missing/total_features*100:.1f}%)")
    
    # 특징 그룹 분석
    print(f"\n스테이션/라인별 특징 분석:")
    feature_groups = {}
    
    for col in df.columns:
        if col not in ['Id', 'Response']:
            parts = col.split('_')
            if len(parts) >= 2:
                group = parts[0] + '_' + parts[1]
                if group not in feature_groups:
                    feature_groups[group] = []
                feature_groups[group].append(col)
    
    sorted_groups = sorted(feature_groups.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"   총 {len(feature_groups)}개의 측정 스테이션 발견")
    print("   상위 15개 스테이션:")
    
    for i, (group, features) in enumerate(sorted_groups[:15], 1):
        group_missing_rate = df[features].isnull().sum().sum() / (len(features) * len(df)) * 100
        print(f"   {i:2d}. {group}: {len(features):3d}개 특징 (결측률: {group_missing_rate:.1f}%)")
    
    return df, feature_groups

def create_features(df):
    print(f"\n특징 공학:")
    print("-" * 30)
    
    feature_df = pd.DataFrame()
    feature_df['Id'] = df['Id']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Id', 'Response'], errors='ignore')
    print(f"   처리할 숫자 특징: {len(numeric_cols):,}개")
    
    print("   기본 통계 특징 생성 중...")
    feature_df['count_non_null'] = df[numeric_cols].count(axis=1)
    feature_df['count_zeros'] = (df[numeric_cols] == 0).sum(axis=1) 
    feature_df['missing_count'] = df[numeric_cols].isnull().sum(axis=1)
    feature_df['missing_ratio'] = feature_df['missing_count'] / len(numeric_cols)
    
    print("   통계량 계산 중...")
    feature_df['mean'] = df[numeric_cols].mean(axis=1, skipna=True)
    feature_df['std'] = df[numeric_cols].std(axis=1, skipna=True)
    feature_df['min'] = df[numeric_cols].min(axis=1, skipna=True)
    feature_df['max'] = df[numeric_cols].max(axis=1, skipna=True)
    feature_df['range'] = feature_df['max'] - feature_df['min']
    feature_df['median'] = df[numeric_cols].median(axis=1, skipna=True)
    
    print("   스테이션별 특징 생성 중...")
    station_groups = {}
    for col in numeric_cols:
        parts = col.split('_')
        if len(parts) >= 2:
            station = parts[0] + '_' + parts[1]
            if station not in station_groups:
                station_groups[station] = []
            station_groups[station].append(col)
    
    top_stations = sorted(station_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    for station, cols in top_stations:
        if len(cols) > 1:
            station_data = df[cols]
            feature_df[f'{station}_mean'] = station_data.mean(axis=1, skipna=True)
            feature_df[f'{station}_count'] = station_data.count(axis=1)
            feature_df[f'{station}_std'] = station_data.std(axis=1, skipna=True)
    
    if 'Response' in df.columns:
        feature_df['Response'] = df['Response']
    
    print(f"   완료! 총 {len(feature_df.columns)}개 특징 생성")
    
    return feature_df

def analyze_correlations(feature_df):
    print(f"\n특징 중요도 분석:")
    print("-" * 40)
    
    if 'Response' not in feature_df.columns:
        print("목표 변수가 없어서 중요도 분석을 할 수 없습니다.")
        return None
    
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
    
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n   상위 20개 중요 특징 (상관계수 기준):")
    print("   " + "-" * 50)
    
    for i, (feature, corr) in enumerate(sorted_corr[:20], 1):
        direction = "양의" if corr > 0 else "음의"
        print(f"   {i:2d}. {feature:25s}: {corr:+.5f} ({direction})")
    
    return sorted_corr

def compare_groups(feature_df):
    print(f"\n정상품 vs 불량품 비교 분석:")
    print("-" * 50)
    
    if 'Response' not in feature_df.columns:
        return
    
    normal = feature_df[feature_df['Response'] == 0]
    failure = feature_df[feature_df['Response'] == 1]
    
    print(f"   정상품: {len(normal):,}개")
    print(f"   불량품: {len(failure):,}개")
    
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

def save_results(df, feature_df, correlations):
    print(f"\n결과 저장 중...")
    
    try:
        # 기본 통계 저장
        summary = {
            'total_samples': len(df),
            'total_features': len(df.columns) - 2,
            'failure_rate': df['Response'].mean(),
            'missing_features_90pct': (df.isnull().sum() > len(df) * 0.9).sum(),
            'created_features': len(feature_df.columns)
        }
        
        with open('C:/Users/ASUS/bosch/analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write("Bosch Production Line Performance - 분석 요약\\n")
            f.write("=" * 50 + "\\n")
            f.write(f"분석 샘플 수: {summary['total_samples']:,}\\n")
            f.write(f"원본 특징 수: {summary['total_features']:,}\\n")
            f.write(f"불량률: {summary['failure_rate']:.4%}\\n")
            f.write(f"90% 이상 결측 특징: {summary['missing_features_90pct']:,}\\n")
            f.write(f"생성한 특징 수: {summary['created_features']:,}\\n\\n")
            
            if correlations:
                f.write("상위 10개 중요 특징:\\n")
                for i, (feature, corr) in enumerate(correlations[:10], 1):
                    f.write(f"{i:2d}. {feature}: {corr:+.5f}\\n")
        
        print("   analysis_summary.txt 저장 완료")
        
        # 생성한 특징 저장 (샘플)
        feature_df.head(1000).to_csv('C:/Users/ASUS/bosch/engineered_features_sample.csv', index=False)
        print("   engineered_features_sample.csv 저장 완료")
        
    except Exception as e:
        print(f"   저장 중 오류: {str(e)}")

def main():
    try:
        start_time = pd.Timestamp.now()
        
        print("분석 시작...")
        
        # 1. 전체 개요
        files = analyze_data_overview()
        
        # 2. 숫자 데이터 분석
        data_dir = "C:/Users/ASUS/bosch/data/"
        df, feature_groups = analyze_train_numeric(data_dir, sample_size=100000)
        
        # 3. 특징 공학
        feature_df = create_features(df)
        
        # 4. 상관관계 분석
        correlations = analyze_correlations(feature_df)
        
        # 5. 그룹 비교
        compare_groups(feature_df)
        
        # 6. 결과 저장
        save_results(df, feature_df, correlations)
        
        # 7. 최종 요약
        end_time = pd.Timestamp.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n" + "="*70)
        print("최종 분석 결과 요약")
        print("="*70)
        
        failure_rate = df['Response'].mean()
        total_features = len(df.columns) - 2
        useful_features = len([col for col in df.columns if df[col].isnull().sum() < len(df) * 0.9])
        
        print(f"데이터셋 특성:")
        print(f"   분석 샘플: {df.shape[0]:,} x {df.shape[1]:,}")
        print(f"   불량률: {failure_rate:.4%} (극도로 불균형)")
        print(f"   전체 특징: {total_features:,}개")
        print(f"   유용한 특징: {useful_features:,}개 (90% 미만 결측)")
        print(f"   측정 스테이션: {len(feature_groups)}개")
        
        print(f"\n특징 공학 결과:")
        print(f"   생성된 집계 특징: {len(feature_df.columns)}개")
        
        if correlations:
            max_corr = max(abs(corr) for _, corr in correlations[:10])
            print(f"   최고 상관계수: {max_corr:.4f}")
        
        print(f"\n핵심 인사이트:")
        print(f"   1. 극도로 불균형한 데이터 (불량률 {failure_rate:.2%})")
        print(f"   2. 대부분 특징이 희소함 (90% 이상 결측)")
        print(f"   3. 결측값 패턴이 중요한 정보원")
        print(f"   4. 집계 특징이 예측에 유용")
        print(f"   5. 스테이션별 접근이 필요")
        
        print(f"\n추천 모델링 전략:")
        print(f"   불균형 처리: SMOTE, 가중치 조정")
        print(f"   특징 선택: 상관관계, 분산 기반")
        print(f"   모델: XGBoost, LightGBM")
        print(f"   평가: Matthews Correlation Coefficient")
        
        print(f"\n분석 완료 시간: {processing_time:.1f}초")
        
        del df, feature_df
        gc.collect()
        
        print(f"\n모든 분석 완료!")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()