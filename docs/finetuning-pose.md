# YOLO26n COCO17 pose 추가 학습

이 실험은 기존 `yolo26n-pose.pt`의 **사람 박스 + COCO 17관절** 추가 학습이다.
최신 요청에 따라 nano pose만 사용한다. VisDrone의 pedestrian/people 박스는 관절 정답이
없으므로 이 학습에 사용하지 않는다. 기존 일반 장애물 탐지기, 80클래스 탐지 가중치,
서비스 코드, `.env`, 기존 pose 가중치는 변경하지 않는다. 결과는 별도 경로의 opt-in
후보이며 드론 영상에서의 성능 개선이나 안전성 검증을 뜻하지 않는다.

## 데이터와 논문

| 자료 | 1차 출처 | 이 실험에서의 역할 |
|---|---|---|
| COCO 2017 Keypoints | [COCO 원논문](https://arxiv.org/abs/1405.0312), [공식 다운로드](https://cocodataset.org/#download), [공식 API](https://github.com/cocodataset/cocoapi) | 실제 RGB 이미지와 원본 `person_keypoints_train2017.json`, `person_keypoints_val2017.json` 사용 |
| UAV-Human, CVPR 2021 | [논문](https://arxiv.org/abs/2104.00946), [공식 저장소](https://github.com/sutdcv/UAV-Human), [pose 도구](https://github.com/sutdcv/UAV-Human/tree/master/uavhumanposetools) | UAV 촬영, pose 22,476프레임. 후속 항공 도메인 검증 후보이며 현재 실험에는 사용하지 않음 |
| HRNet, CVPR 2019 | [논문](https://arxiv.org/abs/1902.09212) | 고해상도 관절 표현의 중요성. 본 실험에서 HRNet을 학습하거나 재현하지 않음 |
| YOLO26 | [공식 논문](https://arxiv.org/abs/2606.03748), [공식 설명](https://docs.ultralytics.com/models/yolo26/) | 기존 nano pose 모델과 Ultralytics 8.4.183 사용 |
| VisDrone | [논문](https://arxiv.org/abs/2001.06303), [공식 자료](https://github.com/VisDrone/VisDrone-Dataset) | 항공 사람 박스 데이터이지만 COCO17 관절 GT가 없어 제외 |

COCO annotation은 [공식 이용 조건](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)의
CC BY 4.0이며 COCO Consortium에 출처를 표시해야 한다. 이미지 저작권은 원 소유자에게
있으며 개별 이미지/Flickr 조건을 따른다. 이미지 전체에 단일 CC BY 4.0 허가가 있다는
뜻은 아니다. Ultralytics 코드·모델의 [AGPL-3.0/상용 라이선스](https://www.ultralytics.com/license)도
데이터 라이선스와 별개다. UAV-Human의 별도 배포·사용 조건은 실제 도입 전에 공식
배포처에서 확인해야 하며 이번 실행에서는 다운로드하지 않았다.

## 분할과 변환

원본 공식 annotation ZIP만 원격에 내려받고 필요한 이미지만 공식 S3에서 선택 다운로드한다.
전체 train2017 이미지 ZIP은 받지 않는다. 기본 선택은 train2017에서 1,600장,
val2017에서 내부 검증 250장과 최종 test 250장이다. 여기의 `test`는 원래 COCO test2017이
아니라 **정답이 공개된 val2017 내부의 독립 평가 부분집합**이다.

선택은 `SHA256("42:<image_id>")` 정렬로 고정한다. image ID 단위로 나누며 train은 공식
train2017, val/test는 공식 val2017을 유지한다. val/test ID와 실제 이미지 SHA256의 교차
중복을 검사한다. COCO 정지 사진에는 신뢰할 video/scene ID가 없어 scene-disjoint라고
주장하지 않는다. 인접 영상 프레임을 무작위 분할한 자료를 만들지 않는다.
최종 test는 epoch 선택, 조기 종료, 하이퍼파라미터 변경에 사용하지 않는다.

관절이 하나 이상 있는 noncrowd 사람이 포함된 이미지를 고른다. 선택된 이미지에서는 표기 관절이 하나 이상 있는
noncrowd 사람의 박스와 관절만 유지한다. 전 관절 미표기인 사람은 COCO pose 관례와
Ultralytics 8.4.183 공식 converter에 맞춰 제외하며 제외 수를 별도로 남긴다.
따라서 순수 배경 이미지 분포를 대표하지 않는다. COCO `category_id=1`은 YOLO `0: person`으로
변환한다. YOLO 한 행은 `0 cx cy w h`와 17개의 `(x,y,v)`, 총 56값이다.
박스는 이미지 경계로 clipping하고 좌표는 폭·높이로 나눈다. 관절 `v=0`은 `(0,0,0)`,
`v=1`은 가려졌지만 표기된 관절, `v=2`는 보이는 관절로 보존한다. 표기된 관절이 경계
밖이면 실패하며 임의 경계 관절을 만들지 않는다. 원본 train/val 전체 annotation에서
표기 관절의 경계 위반이 없는 것도 별도 확인했다. 좌우 반전은 COCO 관절 쌍 인덱스를
사용한다. 중복 행은 한 번만 보존한다.

`iscrowd=1`은 제외한다. Native YOLO 평가는 COCO crowd/segmentation ignore 영역 처리를
그대로 구현하지 않으므로, crowd 및 미표기 사람의 예측이 false positive가 될 수 있다.
누락 관절을 pseudo-label로 채우지 않는다. 수치는 Ultralytics native pose OKS 평가이며
COCO 공식 평가 서버 점수와 동일하다고 주장하지 않는다. COCO17 OKS sigma를 쓰고
Ultralytics 8.4.183은 면적을 박스 면적의 0.53배로 근사한다.

## 실행 환경과 재현 명령

로컬 확인은 기존 `vision/.venv`의 Python 3.13, torch 2.7.1+cu126,
Ultralytics 8.4.183을 사용한다. 실제 GPU 실험은 coordinator가 관리하는 RTX 4090 24GB
원격 환경의 Python 3.11.13, 같은 torch/Ultralytics 버전에서 수행한다.
로컬 새 venv나 의존성 업그레이드는 없다. 아래는 `vision/` 기준이며 원격에서는
`PYTHON=python`, 스크립트의 업로드 위치를 사용한다.

```sh
PYTHON=.venv/bin/python
mkdir -p training/pose/sources
curl --fail --location --retry 3 \
  --output training/pose/sources/annotations_trainval2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
$PYTHON tools/finetune_pose.py prepare \
  --annotations training/pose/sources/annotations_trainval2017.zip \
  --output training/pose/dataset_pose17 --train-images 1600 --val-images 250 --test-images 250
$PYTHON tools/finetune_pose.py train \
  --weights yolo26n-pose.pt --data training/pose/dataset_pose17/data.yaml \
  --project training/pose/runs --name nano_pose20 --epochs 20 \
  --imgsz 640 --batch 8 --freeze 10 --device 0 --memory-fraction 0.3
$PYTHON tools/finetune_pose.py eval \
  --baseline yolo26n-pose.pt --tuned training/pose/runs/nano_pose20/weights/best.pt \
  --data training/pose/dataset_pose17/data.yaml --split test --imgsz 640 --batch 8 \
  --device 0 --memory-fraction 0.3 --output training/pose/results/heldout.json
```

학습은 AdamW, lr0=0.0001, warmup_bias_lr=0.0001, lrf=0.1, weight_decay=0.0005,
seed=42, deterministic=True, freeze=10, patience=5, warmup 1epoch이다. AMP와 디스크 이미지
캐시는 끄고 mosaic/mixup도 끈다. batch와 nbs를 같게 하여 불필요한 gradient accumulation을
피한다. resize/letterbox, RGB 입력, 0–1 정규화는 고정 버전 Ultralytics가 수행한다.
640은 속도·메모리와 작은 관절 보존의 절충이다. 320 등으로 줄이면 먼 사람의 관절이
몇 픽셀로 줄 수 있으며 이 실험에서 별도 해상도 비교나 드론 거리별 검증은 하지 않았다.

모든 CUDA 모델 로드·학습·평가는 실행 호스트의
`/tmp/neurovector-vision-finetune-gpu.lock`에 exclusive `flock`을 잡는다. lock을 얻은 뒤
여유 VRAM을 확인하고 프로세스 메모리 상한을 설정한다. 학습과 평가 CLI를 분리해
사이에 depth 작업에 GPU를 반환한다. 기존 다른 학습 프로세스를 중단하지 않는다.

`weights/best.pt`는 내부 val fitness로 선택한 별도 추론 후보, `weights/last.pt`는 마지막
epoch 추론 가중치, `resume.pt`는 optimizer/scaler/EMA 상태를 포함한 strip 이전 복사본이다.
중단된 실행은 `--weights <run>/resume.pt --resume`으로 남은 기존 epoch 계획만 재개할 수
있으며 optimizer가 strip된 pretrained/final 추론 checkpoint로 resume하면 실패한다. 완료 후 더 긴 추가
학습은 `--resume` 없이 best.pt에서 새로운 이름/epoch으로 warm-start한다. 이때 optimizer는
새로 시작하며 원래 실행의 정확한 연속 재개라고 부르지 않는다.

학습 결과는 자동으로 서비스에 적용하지 않는다. 이 pose checkpoint는 일반 80클래스
장애물 탐지 동작을 학습하지 않는다. 후속 검토에서 후보를 명시적으로 선택하려면
`vision/`에서 아래처럼 pose 후보를 opt-in 하고 다른 모델도 기존 nano 가중치를 명시한다. 이 문서의 명령은 실행하거나 `.env`에
저장하지 않았다. 개선이 없으면 기존 baseline을 계속 사용한다.

```sh
NEUROVECTOR_MODEL_PATH="$PWD/yolo26n.pt" \
NEUROVECTOR_FACE_MODEL_PATH="$PWD/training/pose/runs/nano_pose20/weights/best.pt" \
NEUROVECTOR_DEPTH_MODEL_PATH="$PWD/yolo26n-depth.pt" \
  .venv/bin/python -m neurovector_vision
```

로컬에는 전체 RGB 자료 대신 manifest/변환 GT와 첫 test 이미지 한 장만 보존한다.
전체 학습을 다시 실행하려면 위 `prepare`의 output을 새 디렉터리로 지정해 선택 이미지를
재구성하고 train/eval의 data 경로도 그 디렉터리로 바꾼다. 이미 완성된 output은 덮어쓰지 않는다.
실제 원격 명령 배열은 `training/pose/commands.json`에 별도로 보존한다.

## 측정 산출물

데이터 준비 후 고정된 규모는 다음과 같다. 각 사람은 표기 관절이 하나 이상 있다.

| 분할 | 이미지 | 사람/박스 | 표기 관절 | 제외 crowd | 제외 전 관절 미표기 사람 |
|---|---:|---:|---:|---:|---:|
| train | 1,600 | 4,284 | 46,617 | 122 | 2,127 |
| 내부 val | 250 | 664 | 7,555 | 20 | 281 |
| 독립 test | 250 | 669 | 7,178 | 16 | 274 |

실제 학습은 설정한 최대 20epoch 중 patience=5에 의해 6epoch에서 종료됐다.
실제 optimizer update는 **1,200회**, 학습 가능 parameter는 **2,563,720개**, 학습·최종 val까지
**315.8초**였다. backbone `model.0.`–`model.9.`와 `.dfl`을 freeze했다.
코드 SHA256 및 dataset summary SHA256은 실행 호스트와 로컬 보존본이 일치한다.

독립 test **250장 / 유효 사람 669명 / 표기 관절 7,178개**의 한 번의 고정 비교 결과다.
모든 표의 값은 %이며 차이는 percentage point다.

| 지표 | 기존 nano pose | 추가 학습 best.pt | 차이 |
|---|---:|---:|---:|
| Pose mAP50 | 82.23 | 81.58 | -0.65 |
| Pose mAP50–95 | 57.85 | 54.67 | -3.18 |
| Pose precision | 85.57 | 89.29 | +3.72 |
| Pose recall | 78.03 | 75.64 | -2.39 |
| Box mAP50 | 92.33 | 92.16 | -0.18 |
| Box mAP50–95 | 67.05 | 65.35 | -1.70 |
| Box precision | 92.06 | 91.75 | -0.32 |
| Box recall | 81.17 | 80.87 | -0.30 |

**개선되지 않았다. 기존 `yolo26n-pose.pt`를 유지한다.** Precision 단독 증가는 AP/recall
감소를 상쇄하지 못한다. 학습 후보는 후속 연구·재현용으로만 보존하며 자동 배포하지 않았다.
최종 test 결과를 본 뒤 epoch/학습률/분할/해상도를 바꾸거나 추가 후보를 선정하지 않았다.
이 결과는 작은 COCO 부분집합에서의 continuation 실험이며, 이미 전체 COCO로 사전 학습된
모델보다 좋아질 것을 보장하지 않는다. 항공 카메라/거리/가림별 검증은 별도로 남아 있다.

- 비교 원본: [heldout.json](../training/pose/results/heldout.json)
- 실제 업데이트·환경·hash: [training_evidence.json](../training/pose/runs/nano_pose20/training_evidence.json)
- 세부 epoch: [results.csv](../training/pose/runs/nano_pose20/results.csv)
- 실제 명령: [commands.json](../training/pose/commands.json)
- 실제 변환 GT와 원본/이미지 hash: [summary.json](../training/pose/dataset_pose17/summary.json)
- 학습 후보: `training/pose/runs/nano_pose20/weights/best.pt` (7,791,342bytes)
- 초기 가중치 SHA256: `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9`
- 후보 SHA256: `f4652424a2f7d602b18c1f23da86e686499ed55ef8b6eb754fbe68a4245cbf5b`

로컬 CPU로 후보를 다시 로드하고 첫 test 이미지에 COCO17 forward를 실행했으며,
optimizer 상태가 보존된 resume 복사본(epoch=5, 계획 20epoch, optimizer state 399개,
EMA update 1,200회)의 metadata도 확인했다. 재개 학습 자체를 추가 실행하지는 않았다.
로드·forward 증거는 [checkpoint_validation.json](../training/pose/checkpoint_validation.json)에 있다.
순수 변환/분할/가시성/cached-image 재사용 테스트 **13개**와 repo 설정의 ruff가
로컬에서 통과했고, Python 3.11 원격에서도 같은 테스트와 동등한 ruff 규칙이 통과했다. `dataset_pose17/summary.json`과 split manifest는
원본 annotation ZIP/JSON·이미지 ID·이미지 바이트의 SHA256과 이미지/사람/관절 수를 남긴다.
`training_evidence.json`은 실제 optimizer update 수, freeze 목록, 코드·초기 가중치 hash,
환경과 정확한 명령을 남기고 `results/heldout.json`은 같은 test/전처리/라벨을 사용한
baseline 대비 tuned의 box/pose mAP50, mAP50–95, precision, recall을 저장한다.

Precision/recall은 native evaluator의 confidence sweep에서 선택한 best-F1 지점의 값이다.
AP는 confidence 0.001부터 집계하며 box는 IoU, pose는 OKS 0.50 및 0.50–0.95를 사용한다.
