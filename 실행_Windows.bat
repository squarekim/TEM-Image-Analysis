@echo off
chcp 65001 >nul
cd /d "%~dp0"
title TEM Particle Analyzer

echo ============================================
echo   TEM Particle Analyzer
echo ============================================
echo.

set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if not defined PY python --version >nul 2>&1 && set PY=python
if not defined PY (
    echo [오류] 파이썬을 찾을 수 없습니다.
    echo.
    echo   1^) https://www.python.org/downloads/ 에서 파이썬을 설치하세요.
    echo   2^) 설치 화면 맨 아래 "Add python.exe to PATH" 를 반드시 체크하세요.
    echo   3^) 설치 후 이 파일을 다시 실행하세요.
    echo.
    pause
    exit /b 1
)

echo 파이썬 확인 완료.
%PY% --version
echo.

echo 필요한 프로그램을 확인하는 중입니다. 처음 한 번은 몇 분 걸릴 수 있습니다...
%PY% -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo [오류] 설치에 실패했습니다. 인터넷 연결을 확인하고 다시 시도하세요.
    echo 회사 네트워크라면 방화벽 때문일 수 있습니다.
    echo.
    pause
    exit /b 1
)

echo 준비 완료. 프로그램을 시작합니다.
echo (이 검은 창은 프로그램이 켜져 있는 동안 함께 열려 있습니다. 닫지 마세요.)
echo.

%PY% run.py
if errorlevel 1 (
    echo.
    echo [오류] 실행 중 문제가 발생했습니다. 위의 메시지를 복사해서 알려주세요.
    echo.
    pause
    exit /b 1
)
