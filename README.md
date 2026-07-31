# TEM Particle Analyzer

TEM(투과전자현미경) 이미지에서 구형 입자를 자동으로 검출하고 크기를 측정하는 프로그램입니다.

## 주요 기능

- **자동 입자 검출**: OpenCV 기반 이미지 처리로 구형 입자 자동 인식
- **스케일바 자동 인식**: OCR을 통한 TEM 이미지 스케일바 자동 감지 (실패 시 수동 입력 가능)
- **크기 측정**: 각 입자의 직경, 면적 측정
- **통계 분석**: 평균, 표준편차, D10, D50, D90 자동 계산
- **분포 히스토그램**: 입자 크기 분포를 시각적으로 표시
- **Excel 내보내기**: 측정 결과를 Excel 파일로 저장

## 설치

```bash
pip install -r requirements.txt
```

이것만으로 바로 실행 가능합니다. 스케일바는 수동으로 입력하면 됩니다.

### (선택) 스케일바 자동 OCR 인식

스케일바 텍스트를 자동으로 읽으려면 Tesseract OCR을 추가 설치하세요:

```bash
pip install pytesseract
```

+ Tesseract 바이너리 설치:
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) 에서 설치

설치하지 않아도 프로그램은 정상 동작하며, 수동 입력 모드로 자동 전환됩니다.

## 실행

```bash
python run.py
# 또는
python -m tem_analyzer
```

## 사용법

1. **이미지 열기** - TEM 이미지 파일(PNG, JPG, TIFF 등)을 선택합니다
2. 스케일바가 자동 감지되지 않으면 수동으로 입력합니다
3. 필요 시 분석 파라미터(최소/최대 면적)를 조정합니다
4. **분석 실행** - 입자가 자동으로 검출되고 측정됩니다
5. **Excel 내보내기** - 결과를 Excel 파일로 저장할 수 있습니다
