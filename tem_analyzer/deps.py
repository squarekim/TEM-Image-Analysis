"""Report missing third-party packages as instructions rather than a traceback.

Running the app on a machine where the requirements were never installed
otherwise fails with a bare "No module named 'cv2'", which does not say which
command fixes it or that pip must target the same interpreter.
"""

import importlib.util
import os
import sys

# import name -> distribution name shown to the user
REQUIRED = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "PyQt5": "PyQt5",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
}


def missing():
    return [pkg for mod, pkg in REQUIRED.items()
            if importlib.util.find_spec(mod) is None]


def check(exit_on_missing=True):
    absent = missing()
    if not absent:
        return True

    exe = os.path.basename(sys.executable) or "python"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    req = os.path.join(root, "requirements.txt")

    print("=" * 60)
    print(" 필요한 패키지가 설치되어 있지 않습니다.")
    print("=" * 60)
    print()
    print(" 없는 패키지: " + ", ".join(absent))
    print()
    print(" 아래 명령을 실행하세요 (처음 한 번만, 몇 분 걸립니다):")
    print()
    print(f'     "{sys.executable}" -m pip install --user -r "{req}"')
    print()
    print(" 설치 후 프로그램을 다시 실행하세요.")
    print()
    print(" 회사 네트워크에서 설치가 막히면 사내 미러 주소를 IT에 문의하거나,")
    print(" 아래처럼 신뢰 호스트를 지정해 보세요:")
    print()
    print(f'     "{sys.executable}" -m pip install --user \\')
    print("         --trusted-host pypi.org --trusted-host files.pythonhosted.org \\")
    print(f'         -r "{req}"')
    print()
    print("=" * 60)

    if exit_on_missing:
        try:
            input(" Enter 키를 누르면 종료합니다...")
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(1)
    return False
