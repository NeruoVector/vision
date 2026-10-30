# Nano depth · pose 파인튜닝

학습과 별도 테스트를 완료했지만 **두 학습본 모두 원본 nano보다 성능이 낮았다.**
학습본은 연구 결과로 보관하고, 사용 모델로는 원본 nano 가중치를 권장한다.
서비스 기본 설정과 원본 가중치는 변경하지 않았다.

| 모델 | 실제 학습 | 별도 테스트 지표 | 원본 → 학습본 | 판단 |
| --- | --- | --- | --- | --- |
| `yolo26n-depth` | 15 epoch · 960 updates | 128장, RMSE(m), 낮을수록 좋음 | 0.8121 → 1.0742 | 오차 증가 |
| `yolo26n-pose` | 6 epoch · 1,200 updates | 250장, pose mAP50–95, 높을수록 좋음 | 57.85% → 54.67% | 정확도 감소 |

Pose는 최대 20 epoch 계획에서 내부 검증 기준 `patience=5`로 조기 종료했다.
테스트 결과를 보고 학습 설정이나 체크포인트를 다시 선택하지 않았다.
Depth는 RTX 4090에서 학습하고, 두 가중치의 비교는 같은 로컬 RTX 3060 Ti에서 했다.
Pose의 학습과 비교는 RTX 4090에서 실행했다.

Orca의 Codex 작업 두 개가 각각 데이터 준비·학습·평가를 담당했다.

| 작업 | 데이터 | 평가 |
| --- | --- | --- |
| Depth | TUM RGB-D의 실제 RGB·깊이 쌍, 깊이 PNG 값 ÷ 5000 = m | 별도 시퀀스에서 보정 없는 AbsRel·RMSE·δ1, 중앙값 정렬 지표 별도 기록 |
| Pose | COCO 2017의 실제 사람 17개 키포인트, 가시성 0/1/2 유지 | 별도 이미지에서 원본·학습 모델의 pose 및 box mAP 비교 |

각 작업의 데이터 출처, 분할, 명령, 가중치, 측정 결과는 아래 보고서에 기록한다.

- [Depth 보고서](finetuning-depth.md)
- [Pose 보고서](finetuning-pose.md)

연구용 학습본:

- [Depth best.pt](../training/depth/runs/yolo26n-tum15/weights/best.pt)
- [Pose best.pt](../training/pose/runs/nano_pose20/weights/best.pt)

원본 nano: [depth](../yolo26n-depth.pt), [pose](../yolo26n-pose.pt).
Depth의 보정 없는 AbsRel도 0.4017 → 0.4168로 악화했다.
Pose의 박스 mAP50–95도 67.05% → 65.35%로 낮아졌다.

학습과 체크포인트 선택에는 train과 내부 val만 사용한다. 최종 test는 학습·보정에
사용하지 않는다. 데이터 ID와 SHA256, 학습 인자, 실제 optimizer update 수를 저장한다.

## 실행 환경과 비용

Vast.ai의 이번 작업 전용 인스턴스는 `58314269`이며, 한국 리전 RTX 4090 24GB와
40GB 디스크를 사용한다. 대여 시 저장 공간 포함 요금은 시간당 $0.411111이었다.
승인된 총예산은 $10이다. 2시간 뒤 폐기하는 로컬 systemd 타이머를 설정했으며,
결과를 회수한 뒤 인스턴스를 폐기하고 타이머를 해제했다. 2026-10-30 07:33 KST에
공급자 인스턴스 목록에서 이번 인스턴스가 사라진 것을 확인했다. 계산·저장 비용의
상한 시간 기준 추정은 **$0.17**이며 전송료는 별도다. 최종 청구 확정액은 아니다.
[폐기 확인](../training/cloud/destroy-confirmation.json)과
[비용 산출](../training/cloud/cost.json)을 보관한다. 기존의 다른 인스턴스는 변경하지 않았다.

Python 3.11.13, PyTorch 2.7.1+cu126, Ultralytics 8.4.183을 사용한다.
클라우드에서는 공유 GPU 파일 잠금으로 작업 충돌을 막았다. Depth 최종 평가는
선택 데이터 전체를 회수한 뒤 로컬 GPU에서 pose 작업과 병행했다.
생성 데이터·로그·체크포인트·클라우드 기록은 `vision/training/`에 보관하며 Git에서 제외한다.

## 확인한 동작

`vision` 테스트 140개와 새 도구의 Ruff 검사가 통과했다.
실제 학습 가중치를 기존 `DepthEstimator`와 `FaceDistanceDetector`에 넣어 CPU 추론도
확인했다. Depth는 입력 크기의 유한한 양수 깊이 맵을 반환했고, pose는 nano·17관절
형식을 유지하며 얼굴·신체 부위 결과를 반환했다. 이 검사는 파일 호환성을 확인하며,
위 테스트의 정확도 저하를 상쇄하는 근거가 아니다.

## 참고 근거와 적용 범위

- [Depth Anything V2](https://arxiv.org/abs/2406.09414): 사전 학습된 깊이 표현을 미터 단위 정답으로 파인튜닝하는 접근을 참고했다.
- [Ultralytics depth 파인튜닝 안내](https://docs.ultralytics.com/tasks/depth/#fine-tuning-on-your-own-data): 사전 학습 가중치, 낮은 학습률, AdamW를 적용한다.
- [TUM RGB-D 데이터와 논문](https://cvg.cit.tum.de/data/datasets/rgbd-dataset): RGB와 깊이의 시간 대응, 센서 깊이 단위를 따른다.
- [COCO 논문](https://arxiv.org/abs/1405.0312), [COCO keypoint task](https://cocodataset.org/#keypoints-2017): 실제 사람 주석과 COCO 17개 관절 규약을 사용한다.

이 실험은 실내 깊이 및 COCO 사람 pose에 대한 제한된 데이터 적응이다.
실제 드론·사용자 카메라에서의 거리 정확도, 손가락 키포인트, 야외 장거리 성능은
이 테스트만으로 검증되지 않는다. Pose는 기존 COCO 학습 도메인에서의 추가 학습이며,
새로운 도메인에 대한 성능 향상으로 해석하지 않는다.
