#!/bin/bash

VENV_DIR=".venv"
PYTHON=$(which python3)
DATA_DIR="data"

# Python 존재 여부 확인
if [ -z "$PYTHON" ]; then
    echo "Python3가 설치되어 있지 않습니다."
    exit 1
fi

# 가상환경 확인 및 생성
if [ ! -d "$VENV_DIR" ]; then
    echo "Python 가상환경 생성 중..."
    $PYTHON -m venv "$VENV_DIR"
    
    if [ $? -ne 0 ]; then
        echo "가상환경 생성 실패"
        exit 1
    fi
    echo "가상환경 생성 완료: $VENV_DIR"
else
    echo "가상환경이 이미 존재합니다: $VENV_DIR"
fi

# 가상환경 활성화
source $VENV_DIR/bin/activate

# requirements.txt가 존재하고 패키지가 설치되지 않은 경우에만 설치
if [ -f "requirements.txt" ]; then
    # pip list를 통해 패키지 설치 여부 확인 (간단한 방법)
    if ! pip list | grep -q streamlit; then
        echo "패키지 설치 진행"
        pip install -r requirements.txt
    else
        echo "패키지가 이미 설치되어 있습니다."
    fi
fi

# data 디렉토리 확인 및 클론
if [ ! -d "$DATA_DIR" ]; then
    echo "data 디렉토리가 없습니다. Hugging Face 저장소에서 가져오는 중..."
    
    # Hugging Face 저장소
    REPO_URL="https://huggingface.co/rwr9857/SKN14-3rd-3Team"
    CLONE_DIR="SKN14-3rd-3Team"

    # Git, Git LFS 설치 확인
    if ! command -v git &> /dev/null || ! git lfs &> /dev/null; then
        echo "Git 또는 Git LFS가 설치되어 있지 않습니다."
        exit 1
    fi

    # 클론
    echo "Hugging Face 저장소 클론 중..."
    git lfs install
    git clone "$REPO_URL"

    if [ $? -ne 0 ]; then
        echo "저장소 클론 실패"
        exit 1
    fi

    # data 폴더 이동
    if [ -d "$CLONE_DIR/data" ]; then
        echo "data 디렉토리를 프로젝트 루트로 이동 중..."
        mv "$CLONE_DIR/data" ./

        if [ $? -eq 0 ]; then
            echo "이동 완료: ./data"
            rm -rf "$CLONE_DIR"
        else
            echo "data 이동 실패"
            exit 1
        fi
    else
        echo "$CLONE_DIR/data 디렉토리가 존재하지 않습니다."
        exit 1
    fi
    
    # 데이터 초기화 스크립트 실행
    echo "데이터 초기화 중..."
    python rag_img_input.py
    python rag_manuals_input.py
else
    echo "data 디렉토리가 이미 존재합니다. 클론 및 초기화를 건너뜁니다."
fi

# Streamlit 실행
echo "Streamlit 애플리케이션 실행 중..."
streamlit run app.py