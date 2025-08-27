#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bosch 불량 검출 예제 실행 스크립트
간단하게 실행할 수 있는 버전
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(__file__))

def run_simple_autoencoder_example():
    """
    간단한 AutoEncoder 예제 실행
    """
    try:
        from autoencoder_fault_detection import main as autoencoder_main
        print("🤖 AutoEncoder 기반 이상 탐지 실행...")
        autoencoder_main()
    except Exception as e:
        print(f"AutoEncoder 예제 실행 중 오류: {e}")

def run_comprehensive_example():
    """
    종합 불량 검출 시스템 실행
    """
    try:
        from bosch_comprehensive_fault_detection import main as comprehensive_main
        print("🏭 종합 불량 검출 시스템 실행...")
        comprehensive_main()
    except Exception as e:
        print(f"종합 시스템 실행 중 오류: {e}")

def main():
    """
    메인 선택 메뉴
    """
    print("=" * 60)
    print("🏭 Bosch Production Line 불량 검출 예제")
    print("=" * 60)
    
    print("\n실행할 예제를 선택하세요:")
    print("1. AutoEncoder 기반 이상 탐지")
    print("2. 종합 불량 검출 시스템 (지도/비지도/딥러닝)")
    print("3. 두 예제 모두 실행")
    
    try:
        choice = input("\n선택 (1/2/3): ").strip()
        
        if choice == "1":
            run_simple_autoencoder_example()
        elif choice == "2":
            run_comprehensive_example()
        elif choice == "3":
            print("\n📋 1단계: AutoEncoder 예제")
            run_simple_autoencoder_example()
            print("\n" + "="*60)
            print("📋 2단계: 종합 시스템")
            run_comprehensive_example()
        else:
            print("잘못된 선택입니다. 종합 시스템을 실행합니다.")
            run_comprehensive_example()
            
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n실행 중 오류 발생: {e}")
        print("\n기본 예제를 실행합니다...")
        run_simple_autoencoder_example()

if __name__ == "__main__":
    main()