#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bosch Production Line Performance - 실제 데이터 분석
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import gc

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

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
            import os
            size_mb = os.path.getsize(path) / (1024*1024)
            
            # 첫 줄만 읽어서 컬럼 수 확인
            header = pd.read_csv(path, nrows=0)
            num_cols = len(header.columns)
            
            # 전체 행 수 확인 (더 빠른 방법)
            with open(path, 'r') as f:
                num_rows = sum(1 for line in f) - 1  # 헤더 제외
            
            print(f"{name:20s}: {size_mb:7.1f} MB | {num_rows:8,} rows | {num_cols:4,} cols")
            
        except Exception as e:
            print(f"{name:20s}: Error - {str(e)}")
    
    return files

def analyze_train_numeric_sample(data_dir, sample_size=100000):
    """숫자 데이터 샘플 분석"""
    print(f"\n🔢 TRAIN NUMERIC 데이터 분석 (샘플: {sample_size:,}개)")
    print("-" * 50)
    
    # 샘플 데이터 로드
    print("데이터 로딩 중...")
    df = pd.read_csv(f'{data_dir}train_numeric.csv', nrows=sample_size)
    
    print(f"샘플 크기: {df.shape[0]:,} x {df.shape[1]:,}")
    
    # 목표 변수 분석
    print(f"\n🎯 목표 변수 (Response) 분석:")
    response_counts = df['Response'].value_counts()
    failure_rate = df['Response'].mean()
    
    print(f"   정상품 (0): {response_counts[0]:,}개 ({(1-failure_rate)*100:.2f}%)")
    print(f"   불량품 (1): {response_counts[1]:,}개 ({failure_rate*100:.2f}%)")
    print(f"   불량률: {failure_rate:.4%}")
    
    # 결측값 분석
    print(f"\n❓ 결측값 분석:")
    missing_stats = df.isnull().sum()
    total_features = len(df.columns) - 2  # Id, Response 제외
    
    completely_empty = (missing_stats == len(df)).sum()
    mostly_empty = (missing_stats > len(df) * 0.9).sum() 
    half_empty = (missing_stats > len(df) * 0.5).sum()
    no_missing = (missing_stats == 0).sum()
    
    print(f"   전체 특징 수: {total_features:,}개")
    print(f"   완전히 비어있는 컬럼: {completely_empty:,}개 ({completely_empty/total_features*100:.1f}%)")
    print(f"   90% 이상 비어있는 컬럼: {mostly_empty:,}개 ({mostly_empty/total_features*100:.1f}%)")
    print(f"   50% 이상 비어있는 컬럼: {half_empty:,}개 ({half_empty/total_features*100:.1f}%)")
    print(f"   결측값이 없는 컬럼: {no_missing:,}개 ({no_missing/total_features*100:.1f}%)")
    
    # 특징 그룹 분석
    print(f"\n🏭 특징 그룹 분석:")
    feature_groups = {}
    
    for col in df.columns:
        if col not in ['Id', 'Response']:
            parts = col.split('_')
            if len(parts) >= 2:
                group = parts[0] + '_' + parts[1]
                if group not in feature_groups:
                    feature_groups[group] = 0
                feature_groups[group] += 1
    
    # 상위 10개 그룹
    sorted_groups = sorted(feature_groups.items(), key=lambda x: x[1], reverse=True)
    print("   상위 10개 측정 스테이션:")
    for i, (group, count) in enumerate(sorted_groups[:10], 1):
        print(f"   {i:2d}. {group}: {count:3d}개 특징")
    
    return df

def create_basic_features(df):
    """기본 통계 특징 생성"""
    print(f"\n🛠️ 특징 공학:")
    print("-" * 30)
    
    feature_df = pd.DataFrame()
    feature_df['Id'] = df['Id']
    
    # 숫자 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Id', 'Response'], errors='ignore')
    
    # 집계 특징들
    print("   기본 통계 특징 생성 중...")
    feature_df['count_non_null'] = df[numeric_cols].count(axis=1)
    feature_df['count_zeros'] = (df[numeric_cols] == 0).sum(axis=1)
    feature_df['missing_count'] = df[numeric_cols].isnull().sum(axis=1)
    feature_df['missing_ratio'] = feature_df['missing_count'] / len(numeric_cols)
    
    # 통계량
    feature_df['mean'] = df[numeric_cols].mean(axis=1)
    feature_df['std'] = df[numeric_cols].std(axis=1)
    feature_df['min'] = df[numeric_cols].min(axis=1)
    feature_df['max'] = df[numeric_cols].max(axis=1)
    feature_df['range'] = feature_df['max'] - feature_df['min']
    feature_df['median'] = df[numeric_cols].median(axis=1)
    
    # 스테이션별 집계 특징
    print("   스테이션별 특징 생성 중...")
    station_groups = {}
    for col in numeric_cols:
        parts = col.split('_')
        if len(parts) >= 2:
            station = parts[0] + '_' + parts[1]
            if station not in station_groups:
                station_groups[station] = []
            station_groups[station].append(col)
    
    # 주요 스테이션들의 통계
    for station, cols in list(station_groups.items())[:5]:  # 상위 5개만
        station_data = df[cols]
        feature_df[f'{station}_mean'] = station_data.mean(axis=1)
        feature_df[f'{station}_count'] = station_data.count(axis=1)
    
    # 목표 변수 추가
    if 'Response' in df.columns:
        feature_df['Response'] = df['Response']
    
    print(f"   생성된 특징 수: {len(feature_df.columns)}개")
    
    return feature_df

def analyze_correlations(feature_df):
    """상관관계 분석"""
    print(f"\n📊 특징-목표변수 상관관계 분석:")
    print("-" * 40)
    
    if 'Response' not in feature_df.columns:
        print("목표 변수가 없습니다.")
        return
    
    # 상관계수 계산
    feature_cols = [col for col in feature_df.columns if col not in ['Id', 'Response']]
    correlations = feature_df[feature_cols].corrwith(feature_df['Response'])
    correlations = correlations.sort_values(key=abs, ascending=False)
    
    print("상위 10개 상관관계:")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        direction = "양의" if corr > 0 else "음의"
        print(f"   {i:2d}. {feature:20s}: {corr:+.4f} ({direction})")
    
    return correlations

def create_visualizations(df, feature_df, correlations):
    """데이터 시각화"""
    print(f"\n📈 데이터 시각화 생성 중...")
    
    plt.figure(figsize=(20, 15))
    
    # 1. 목표 변수 분포
    plt.subplot(2, 4, 1)
    response_counts = df['Response'].value_counts()
    colors = ['lightgreen', 'salmon']
    plt.bar(['정상품', '불량품'], response_counts.values, color=colors)
    plt.title('목표 변수 분포')
    plt.ylabel('개수')
    for i, v in enumerate(response_counts.values):
        plt.text(i, v + max(response_counts.values) * 0.01, f'{v:,}', ha='center', fontweight='bold')
    
    # 2. 결측값 분포
    plt.subplot(2, 4, 2)
    missing_pct = (df.isnull().sum() / len(df) * 100)
    plt.hist(missing_pct, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('결측값 비율 (%)')
    plt.ylabel('특징 개수')
    plt.title('결측값 분포')
    
    # 3. 특징 개수별 분포
    plt.subplot(2, 4, 3)
    plt.hist(feature_df['count_non_null'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('측정된 특징 개수')
    plt.ylabel('제품 개수')
    plt.title('제품별 측정 특징 개수')
    
    # 4. 상관관계 Top 10
    plt.subplot(2, 4, 4)
    top_corr = correlations.head(10)
    colors = ['red' if x < 0 else 'blue' for x in top_corr.values]
    plt.barh(range(len(top_corr)), top_corr.values, color=colors)
    plt.yticks(range(len(top_corr)), top_corr.index, fontsize=8)
    plt.xlabel('상관계수')
    plt.title('상위 10개 상관관계')
    plt.grid(axis='x', alpha=0.3)
    
    # 5. 불량품 vs 정상품 비교 (측정 개수)
    plt.subplot(2, 4, 5)
    normal = feature_df[feature_df['Response'] == 0]['count_non_null']
    failure = feature_df[feature_df['Response'] == 1]['count_non_null']
    
    plt.hist(normal, bins=30, alpha=0.7, label='정상품', color='green', density=True)
    plt.hist(failure, bins=30, alpha=0.7, label='불량품', color='red', density=True)
    plt.xlabel('측정 특징 개수')
    plt.ylabel('밀도')
    plt.title('정상품 vs 불량품: 측정 개수')
    plt.legend()
    
    # 6. 결측값 비율 비교
    plt.subplot(2, 4, 6)
    normal_missing = feature_df[feature_df['Response'] == 0]['missing_ratio']
    failure_missing = feature_df[feature_df['Response'] == 1]['missing_ratio']
    
    plt.hist(normal_missing, bins=30, alpha=0.7, label='정상품', color='green', density=True)
    plt.hist(failure_missing, bins=30, alpha=0.7, label='불량품', color='red', density=True)
    plt.xlabel('결측값 비율')
    plt.ylabel('밀도')
    plt.title('정상품 vs 불량품: 결측값 비율')
    plt.legend()
    
    # 7. 평균값 비교
    plt.subplot(2, 4, 7)
    normal_mean = feature_df[feature_df['Response'] == 0]['mean'].dropna()
    failure_mean = feature_df[feature_df['Response'] == 1]['mean'].dropna()
    
    plt.hist(normal_mean, bins=30, alpha=0.7, label='정상품', color='green', density=True)
    plt.hist(failure_mean, bins=30, alpha=0.7, label='불량품', color='red', density=True)
    plt.xlabel('측정값 평균')
    plt.ylabel('밀도')
    plt.title('정상품 vs 불량품: 평균값')
    plt.legend()
    
    # 8. 표준편차 비교
    plt.subplot(2, 4, 8)
    normal_std = feature_df[feature_df['Response'] == 0]['std'].dropna()
    failure_std = feature_df[feature_df['Response'] == 1]['std'].dropna()
    
    plt.hist(normal_std, bins=30, alpha=0.7, label='정상품', color='green', density=True)
    plt.hist(failure_std, bins=30, alpha=0.7, label='불량품', color='red', density=True)
    plt.xlabel('측정값 표준편차')
    plt.ylabel('밀도')
    plt.title('정상품 vs 불량품: 변동성')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('C:/Users/ASUS/bosch/bosch_analysis_results.png', dpi=300, bbox_inches='tight')
    print("   시각화 저장됨: bosch_analysis_results.png")
    
    plt.show()

def main():
    """메인 분석 함수"""
    try:
        # 1. 전체 개요
        files = analyze_data_overview()
        
        # 2. 숫자 데이터 샘플 분석
        data_dir = "C:/Users/ASUS/bosch/data/"
        df = analyze_train_numeric_sample(data_dir, sample_size=100000)
        
        # 3. 특징 공학
        feature_df = create_basic_features(df)
        
        # 4. 상관관계 분석
        correlations = analyze_correlations(feature_df)
        
        # 5. 시각화
        create_visualizations(df, feature_df, correlations)
        
        # 6. 요약 리포트
        print(f"\n" + "="*60)
        print("📋 분석 결과 요약")
        print("="*60)
        
        failure_rate = df['Response'].mean()
        total_features = len(df.columns) - 2
        useful_features = len([col for col in df.columns if df[col].isnull().sum() < len(df) * 0.9])
        
        print(f"• 데이터 크기: {df.shape[0]:,} x {df.shape[1]:,}")
        print(f"• 불량률: {failure_rate:.2%} (매우 불균형)")
        print(f"• 전체 특징: {total_features:,}개")
        print(f"• 유용한 특징: {useful_features:,}개 (90% 미만 결측)")
        print(f"• 생성된 집계 특징: {len(feature_df.columns)}개")
        
        if len(correlations) > 0:
            max_corr = correlations.abs().max()
            print(f"• 최고 상관계수: {max_corr:.4f}")
        
        print(f"\n💡 주요 인사이트:")
        print(f"  1. 극도로 불균형한 데이터 - 특별한 처리 필요")
        print(f"  2. 대부분 특징이 희소 - 집계 특징이 중요")
        print(f"  3. 결측값 패턴 자체가 중요한 정보")
        print(f"  4. 스테이션별 분석이 효과적일 것으로 예상")
        
        # 메모리 정리
        del df, feature_df
        gc.collect()
        
        print(f"\n✅ 분석 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()