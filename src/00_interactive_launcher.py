#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏭 Bosch Fault Detection - 대화형 실행 런처
단계별 학습을 위한 메뉴 시스템
"""

import sys
import os
import subprocess

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(__file__))

def print_header():
    """헤더 출력"""
    print("=" * 80)
    print("🏭 Bosch Production Line Fault Detection - 학습 가이드")
    print("=" * 80)
    print("📚 단계별 학습 순서:")
    print("   1단계: 간단한 데모 (5분)")
    print("   2단계: AutoEncoder 심화 (15분)")
    print("   3단계: 종합 시스템 (30분)")
    print("=" * 80)

def run_step_1():
    """1단계: 간단한 데모 실행"""
    print("\n🎯 1단계: 빠른 시작 데모 실행")
    print("📋 내용: Random Forest vs Isolation Forest 비교")
    print("⏱️  예상 시간: 5분")
    try:
        result = subprocess.run([sys.executable, "01_simple_fault_detection_demo.py"], 
                              cwd=os.path.dirname(__file__))
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 1단계 실행 중 오류: {e}")
        return False

def run_step_2():
    """2단계: AutoEncoder 실행"""
    print("\n🧠 2단계: AutoEncoder 기반 이상 탐지")
    print("📋 내용: 딥러닝을 활용한 재구성 오차 기반 불량 탐지")
    print("⏱️  예상 시간: 15분")
    try:
        result = subprocess.run([sys.executable, "02_autoencoder_fault_detection.py"], 
                              cwd=os.path.dirname(__file__))
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 2단계 실행 중 오류: {e}")
        return False

def run_step_3():
    """3단계: 종합 시스템 실행"""
    print("\n🏭 3단계: 종합 불량 검출 시스템")
    print("📋 내용: 모든 접근법을 포함한 완전한 시스템")
    print("⏱️  예상 시간: 30분")
    try:
        result = subprocess.run([sys.executable, "03_comprehensive_fault_detection.py"], 
                              cwd=os.path.dirname(__file__))
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 3단계 실행 중 오류: {e}")
        return False

def show_menu():
    """메뉴 표시 및 선택 처리"""
    print("\n🎮 실행 옵션을 선택하세요:")
    print("1. 1단계만 실행 - 빠른 데모 (추천 시작점)")
    print("2. 2단계만 실행 - AutoEncoder 심화")
    print("3. 3단계만 실행 - 종합 시스템")
    print("4. 순차 실행 - 1→2→3 단계 모두")
    print("5. 종료")
    print("r. 요구사항 설치 가이드")
    
    return input("\n선택 (1-5, r): ").strip().lower()

def show_requirements_guide():
    """요구사항 설치 가이드"""
    print("\n📦 라이브러리 설치 가이드")
    print("=" * 50)
    print("✅ 최소 요구사항 (1단계 실행용):")
    print("   pip install pandas numpy scikit-learn")
    print()
    print("🔧 전체 기능 (모든 단계):")
    print("   pip install -r requirements.txt")
    print()
    print("🧠 딥러닝 옵션 (2, 3단계용):")
    print("   PyTorch: pip install torch torchvision")
    print("   TensorFlow: pip install tensorflow")
    print()
    print("💡 가상환경 권장:")
    print("   python -m venv venv")
    print("   venv\\Scripts\\activate  # Windows")
    print("   source venv/bin/activate  # Linux/Mac")

def main():
    """메인 함수"""
    print_header()
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            success = run_step_1()
            if success:
                print("\n✅ 1단계 완료! 다음 단계를 시도해보세요.")
            
        elif choice == '2':
            success = run_step_2()
            if success:
                print("\n✅ 2단계 완료! AutoEncoder 학습 완료!")
            
        elif choice == '3':
            success = run_step_3()
            if success:
                print("\n✅ 3단계 완료! 모든 모델 비교 완료!")
            
        elif choice == '4':
            print("\n🚀 순차 실행 시작...")
            
            # 1단계
            if run_step_1():
                print("\n✅ 1단계 성공!")
                input("Enter를 눌러 2단계 계속...")
                
                # 2단계
                if run_step_2():
                    print("\n✅ 2단계 성공!")
                    input("Enter를 눌러 3단계 계속...")
                    
                    # 3단계
                    if run_step_3():
                        print("\n🎉 모든 단계 완료! 축하합니다!")
                        print("📊 이제 README_fault_detection.md를 참고하여")
                        print("   고급 커스터마이징을 시도해보세요!")
                        break
        
        elif choice == '5':
            print("\n👋 프로그램을 종료합니다.")
            break
            
        elif choice == 'r':
            show_requirements_guide()
            
        else:
            print("❌ 잘못된 선택입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예기치 못한 오류: {e}")
        print("📋 문제 해결을 위해 README_fault_detection.md의 문제해결 섹션을 참고하세요.")