@echo off
chcp 65001 >nul
cd /d "%~dp0"
title TEM Particle Analyzer

echo ============================================
echo   TEM Particle Analyzer
echo ============================================
echo.

rem "python" alone can resolve to the Microsoft Store stub, which exits without
rem running anything, so each candidate has to prove it can execute code.
set PY=
for %%C in ("py -3" "python" "python3") do (
    if not defined PY (
        %%~C -c "print('ok')" >nul 2>&1 && set "PY=%%~C"
    )
)
if not defined PY (
    echo [오류] 파이썬이 설치되어 있지 않습니다.
    echo.
    echo   1^) https://www.python.org/downloads/ 접속
    echo   2^) 노란색 "Download Python" 버튼 클릭해서 설치 파일 받기
    echo   3^) 받은 파일 실행 후, 첫 화면 맨 아래
    echo      "Add python.exe to PATH" 체크박스를 반드시 체크하세요. ^(중요^)
    echo   4^) "Install Now" 클릭
    echo   5^) 설치가 끝나면 이 파일을 다시 더블클릭하세요.
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
