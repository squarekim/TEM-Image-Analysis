#!/bin/bash
cd "$(dirname "$0")"

echo "============================================"
echo "   TEM Particle Analyzer"
echo "============================================"
echo

PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "[오류] 파이썬을 찾을 수 없습니다."
    echo
    echo "  1) https://www.python.org/downloads/ 에서 파이썬을 설치하세요."
    echo "  2) 설치 후 이 파일을 다시 더블클릭하세요."
    echo
    read -n 1 -s -r -p "아무 키나 누르면 닫힙니다..."
    exit 1
fi

echo "파이썬 확인 완료."
"$PY" --version
echo

echo "필요한 프로그램을 확인하는 중입니다. 처음 한 번은 몇 분 걸릴 수 있습니다..."
if ! "$PY" -m pip install --quiet --disable-pip-version-check -r requirements.txt; then
    echo
    echo "[오류] 설치에 실패했습니다. 인터넷 연결을 확인하고 다시 시도하세요."
    echo "회사 네트워크라면 방화벽 때문일 수 있습니다."
    echo
    read -n 1 -s -r -p "아무 키나 누르면 닫힙니다..."
    exit 1
fi

echo "준비 완료. 프로그램을 시작합니다."
echo "(이 터미널 창은 프로그램이 켜져 있는 동안 함께 열려 있습니다. 닫지 마세요.)"
echo

if ! "$PY" run.py; then
    echo
    echo "[오류] 실행 중 문제가 발생했습니다. 위의 메시지를 복사해서 알려주세요."
    echo
    read -n 1 -s -r -p "아무 키나 누르면 닫힙니다..."
    exit 1
fi
