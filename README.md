# pullup-analyzer

<p align="center">
  <img src="resource/tilte.png" alt="Pull-Up Analyzer" width="920">
</p>

<p align="center">
  풀업 영상을 자동으로 분석해서 <b>반복 횟수</b>, <b>상태</b>, <b>각도</b>, <b>점수</b>, <b>레벨</b>이 포함된 결과 영상을 생성합니다.
</p>

<p align="center">
  <img src="resource/demo.gif" alt="Pull-Up Analyzer Demo" width="360">
</p>

<p align="center">
  <a href="resource/demo.mp4">고화질 데모 영상 보기</a>
  ·
  <a href="SCORING.md">점수 기준 보기</a>
</p>

<p align="center">
  <a href="https://www.youtube.com/@DailyPullUp_%ED%92%80%EC%97%85%EB%A7%A8/shorts">
    <img
      src="https://img.shields.io/badge/YouTube-DailyPullUp_%ED%92%80%EC%97%85%EB%A7%A8-FF0000?logo=youtube&logoColor=white"
      alt="Dailypullup_풀업맨 YouTube"
    >
  </a>
</p>

<p align="center">
  유튜브 채널 <b>Dailypullup_풀업맨</b>에서 Shorts 기반 데모와 풀업 기록도 함께 확인할 수 있습니다.
</p>

## 한눈에 보기

- `videos/` 폴더에 `.mp4` 영상을 넣고 `python demo.py`를 실행하면 됩니다.
- 결과 영상은 `results/` 폴더에 저장됩니다.
- README에는 `GIF`로 바로 보이는 미리보기를 넣고, 자세한 확인은 `MP4` 링크로 연결하는 방식을 추천합니다.

## 주요 기능

- `.mp4` 풀업 영상을 자동 분석
- YOLO pose 모델 기반 사람 자세 추정
- 풀업 반복 횟수 자동 카운트
- `PULL UP`, `PULL DOWN`, `DEAD HANG` 상태 시각화
- 팔 각도, 스켈레톤, 최고 높이, 기준 높이 오버레이
- 회차별 획득 점수와 누적 점수 계산
- 레벨 구간 표시
- 콘솔 진행률 표시 및 실시간 상태 요약
- 결과 영상 저장 후 원본 오디오 재병합

## 빠른 시작

1. `videos/` 폴더에 분석할 `.mp4` 영상을 넣습니다.
2. 아래 명령으로 실행합니다.

```bash
python demo.py
```

3. 결과는 `results/` 폴더에서 확인합니다.

## 설치

### 권장 환경

- Python `3.10+`
- `ffmpeg`, `ffprobe`
- Linux 또는 WSL 환경 권장
- CUDA가 가능하면 `GPU 0` 우선 사용, 실패 시 자동으로 CPU 전환

### Python 패키지 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 시스템 패키지 설치

결과 영상 저장과 오디오 병합은 `ffmpeg/ffprobe` 기준으로 동작합니다.

Ubuntu / Debian 예시:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## 실행 방법

기본 실행:

```bash
python demo.py
```

추론 옵션 조정:

```bash
python demo.py --conf 0.50 --iou 0.45
```

## 폴더 구조

```text
pullup-analyzer/
├── demo.py
├── requirements.txt
├── LICENSE
├── README.md
├── SCORING.md
├── pullup_analyzer/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── console.py
│   ├── rendering.py
│   └── state.py
├── models/
├── videos/
├── results/
└── resource/
```

폴더 용도:

- `models/`: YOLO pose 모델 파일 보관
- `videos/`: 입력 영상 보관
- `results/`: 분석 결과 영상 저장
- `resource/`: 타이틀 이미지 등 리소스 보관

## 지원 모델

지원 모델:

- `yolo26n/s/m/l/x-pose.pt`

기본 모델은 `yolo26m-pose.pt`입니다.  
`models/` 폴더에 기본 모델이 없으면 자동으로 다운로드합니다.

다른 모델을 쓰고 싶으면 실행 전에 환경변수로 지정할 수 있습니다.

```bash
PULLUP_MODEL_NAME=yolo26s-pose.pt python demo.py
```

## 결과 영상에서 볼 수 있는 정보

- 현재 상태: `READY`, `DEAD HANG`, `PULL UP`, `PULL DOWN`
- 반복 횟수
- 그립 타입
- 현재 각도 점수
- 회차별 평가 등급
- 회차별 획득 점수
- 누적 점수
- 레벨
- 최고 높이 / 목표 높이 기준선

## 점수 체계

이 프로젝트는 단순 반복 횟수만 세지 않고, 자세 품질까지 반영해서 `누적 Score`를 계산합니다.

- 회차별 점수 누적
- 높이, 각도, 중심 안정성, 데드행, 탑홀딩 반영
- 좋은 자세일수록 같은 1회라도 더 높은 점수 획득

자세한 기준은 아래 문서를 참고하세요.

- [SCORING.md](SCORING.md)

## 저장 방식

- 결과 영상은 고화질 `H.264` 기반으로 저장됩니다.
- 원본 영상에 오디오가 있으면 마지막에 다시 붙입니다.
- 마지막 프레임은 최종 점수와 레벨을 확인할 수 있도록 몇 초간 유지됩니다.

## 참고 사항

- 현재는 `videos/` 폴더의 `.mp4` 파일만 처리합니다.
- 결과 파일명은 기본적으로 `<원본파일명>_result.mp4` 형식입니다.
- 콘솔은 스크롤이 과하게 밀리지 않도록 진행률 바와 상태줄 중심으로 출력합니다.
- 타이틀 이미지는 `resource/` 폴더의 파일을 사용합니다.

## 레퍼런스

- Ultralytics Pose Task Docs: https://docs.ultralytics.com/tasks/pose/
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics
