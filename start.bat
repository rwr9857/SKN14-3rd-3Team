@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

set VENV_DIR=.venv
set DATA_DIR=data

:: Python 존재 여부 확인
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)

:: 가상환경 확인 및 생성
if not exist "%VENV_DIR%" (
    echo Python 가상환경 생성 중...
    python -m venv "%VENV_DIR%"
    
    if %errorlevel% neq 0 (
        echo 가상환경 생성 실패
        pause
        exit /b 1
    )
    echo 가상환경 생성 완료: %VENV_DIR%
) else (
    echo 가상환경이 이미 존재합니다: %VENV_DIR%
)

:: 가상환경 활성화
call "%VENV_DIR%\Scripts\activate.bat"

:: requirements.txt가 존재하고 패키지가 설치되지 않은 경우에만 설치
if exist "requirements.txt" (
    :: pip list를 통해 패키지 설치 여부 확인
    pip list | findstr /i "streamlit" > nul 2>&1
    if %errorlevel% neq 0 (
        echo 패키지 설치 진행
        pip install -r requirements.txt
    ) else (
        echo 패키지가 이미 설치되어 있습니다.
    )
)

:: data 디렉토리 확인 및 클론
if not exist "%DATA_DIR%" (
    echo data 디렉토리가 없습니다. Hugging Face 저장소에서 가져오는 중...
    
    :: Hugging Face 저장소
    set REPO_URL=https://huggingface.co/rwr9857/SKN14-3rd-3Team
    set CLONE_DIR=SKN14-3rd-3Team

    :: Git, Git LFS 설치 확인
    git --version > nul 2>&1
    if %errorlevel% neq 0 (
        echo Git이 설치되어 있지 않습니다.
        pause
        exit /b 1
    )
    
    git lfs version > nul 2>&1
    if %errorlevel% neq 0 (
        echo Git LFS가 설치되어 있지 않습니다.
        pause
        exit /b 1
    )

    :: 클론
    echo Hugging Face 저장소 클론 중...
    git lfs install
    git clone "%REPO_URL%"

    if %errorlevel% neq 0 (
        echo 저장소 클론 실패
        pause
        exit /b 1
    )

    :: data 폴더 이동
    if exist "%CLONE_DIR%\data" (
        echo data 디렉토리를 프로젝트 루트로 이동 중...
        move "%CLONE_DIR%\data" ".\data"

        if %errorlevel% equ 0 (
            echo 이동 완료: .\data
            rmdir /s /q "%CLONE_DIR%"
        ) else (
            echo data 이동 실패
            pause
            exit /b 1
        )
    ) else (
        echo %CLONE_DIR%\data 디렉토리가 존재하지 않습니다.
        pause
        exit /b 1
    )
    
    :: 데이터 초기화 스크립트 실행
    echo 데이터 초기화 중...
    python rag_img_input.py
    python rag_manuals_input.py
) else (
    echo data 디렉토리가 이미 존재합니다. 클론 및 초기화를 건너뜁니다.
)

:: Streamlit 실행
echo Streamlit 애플리케이션 실행 중...
streamlit run app.py

pause