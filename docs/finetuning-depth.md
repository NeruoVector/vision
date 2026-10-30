# YOLO26n-depth 실측 미세조정

최종 실험은 **기존 `yolo26n-depth.pt`만** 사용한다. 원격 RTX 4090에서 학습 512장·내부 검증 64장·독립 테스트 128장으로 15 epoch를 실행한다. 운영 코드, 환경 설정, 기본 모델 경로와 기존 가중치는 변경하지 않는다. 최종 결과와 판단은 `training/depth/results/report-ko.md`에 기록한다.

## 데이터와 연구 근거

| 1차 자료 | 이번 적용 |
|---|---|
| [Sturm 외, TUM RGB-D benchmark, IROS 2012](https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf) | 실제 Kinect RGB와 등록된 깊이 정답을 제공한다. 서로 다른 녹화 시퀀스 세 개를 사용했다. |
| [Eigen 외, NeurIPS 2014](https://arxiv.org/abs/1406.2283) | 단안 깊이의 스케일 불확실성과 log 깊이 손실을 다룬다. 절대 스케일도 감독하도록 이번 SILog는 `dlam=0.5`로 설정했다. 논문 전체 재현은 아니다. |
| [ZoeDepth, 2023](https://arxiv.org/abs/2302.12288) | 상대 깊이와 미터 깊이 적응을 구분한다. 이미지별 스케일 정렬과 실제 미터 오차를 별도로 평가하는 근거다. ZoeDepth 모델을 학습한 것은 아니다. |
| [Depth Anything V2, 2024](https://arxiv.org/abs/2406.09414) | 깊이 데이터 품질과 사전학습의 중요성을 다룬다. 연구 참고 자료이며 해당 모델이나 의사 정답을 사용하지 않았다. |

다른 후보는 [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)(공식 labeled 자료 약 2.8GB)와 [KITTI depth](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)(정답 다운로드 약 14GB)다. 작은 실내 실측 적응과 명시적 시퀀스 분리를 위해 TUM을 선택했다.

[TUM 공식 형식](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)에 따라 RGB/깊이는 640×480 등록 영상이며, 깊이는 uint16 PNG `/ 5000` 미터, `0`은 결측이다. 원본 PNG 바이트를 보존하고 YAML의 `depth_scale: 5000`으로 변환한다. [Ultralytics 데이터 형식](https://docs.ultralytics.com/datasets/depth/)의 `images/<split>`와 `depth/<split>` 대응을 따른다.

| split | 공식 TUM 시퀀스 | 프레임 수 | 사용처 |
|---|---|---:|---|
| train | `freiburg1_desk` | 512 | optimizer update |
| val | `freiburg1_xyz` | 64 | epoch 선택과 학습 종료 자동 calibration |
| test | `freiburg1_plant` | 128 | 학습 완료 후 고정된 checkpoint의 독립 추론 |

seed는 `20261030`이다. 타임스탬프를 정렬해 20ms 이내 RGB/깊이를 중복 없는 최근접 방식으로 연결하고 시퀀스별 `random.Random(seed).sample`로 뽑는다. 내용 해시 또는 시퀀스가 split 사이에 겹치면 실패한다. 같은 Freiburg 실내 환경의 다른 녹화이므로 **시퀀스 분리이며 물리적 장소 전체의 분리는 아니다**. 인접 프레임의 상관성과 TUM의 사전학습 포함 가능성을 배제할 수 없어 zero-shot 평가라고 주장하지 않는다.

`training/depth/data/tum-nano/manifest.json`에는 모든 선택 ID, 원본 tar member, 타임스탬프, 개별 RGB/깊이 SHA256과 원본 URL·크기·SHA256이 있다. 원본 archive는 원격 자체 임시 폴더에서 준비한 뒤 삭제하고 선택 PNG만 보존한다.

| archive | bytes | SHA256 |
|---|---:|---|
| desk | 344011403 | `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c` |
| xyz | 448204271 | `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98` |
| plant | 695620135 | `2a5b3db30aaec442c5116fb6269f865a33f22db72263d12fff3675a0342b9bd0` |

데이터는 [TUM 명시 CC BY 4.0](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)의 J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers에 귀속된다. 선택과 파일 이름 변경만 수행하며 PNG 내용은 변경하지 않는다. Ultralytics 코드·모델은 저장소의 MIT 표기와 별도로 [Ultralytics 라이선스](https://www.ultralytics.com/license)(AGPL-3.0 / Enterprise)의 적용 대상이다.

## 실행과 재현

도구는 Python 3.11 이상을 지원한다. 원격 기존 환경은 Python 3.11.13, torch `2.7.1+cu126`, ultralytics `8.4.183`이다. 로컬 검증에는 기존 `vision/.venv`를 사용한다. 이 도구를 위해 의존성을 업그레이드하거나 새 환경을 만들지 않는다.

```bash
cd /workspace/vision
python tools/finetune_depth.py prepare --output training/depth/data/tum-nano
python tools/finetune_depth.py train --weights yolo26n-depth.pt --run training/depth/runs/yolo26n-tum15 --epochs 15 --imgsz 512 --batch 8 --freeze 0 --device 0
# 아래 비교는 전체 데이터/가중치를 회수한 로컬에서 실행
cd /home/jeonjaemin/orca/platform/vision
.venv/bin/python tools/finetune_depth.py evaluate --weights yolo26n-depth.pt --imgsz 512 --device 0 --output training/depth/results/pretrained-test.json
.venv/bin/python tools/finetune_depth.py evaluate --weights training/depth/runs/yolo26n-tum15/weights/best.pt --imgsz 512 --device 0 --output training/depth/results/tuned-test.json
```

기존 output/run/평가 JSON은 덮어쓰지 않는다. 재실행에는 새 경로를 지정한다. 데이터는 `--manifest`, run은 `--run`, 사전 다운로드 TGZ는 `prepare --archives`로 지정할 수 있다. 로컬 재현은 `cd vision` 후 `python` 대신 `.venv/bin/python`을 사용한다. 선택한 704쌍 전체를 로컬에 회수해 원격 반환 후에도 검증할 수 있다. 원격 YAML은 `configs/dataset-remote.yaml`에 보존하고 회수 데이터의 YAML은 로컬 경로로 변경한다. 새 위치에서는 새 output에 `prepare`를 실행해 해당 경로와 동일 분할을 재생성할 수 있다. 최종 고정 checkpoint 비교만 코디네이터 요청으로 같은 로컬 RTX 3060 Ti에서 두 모델 모두 실행하며, 학습은 RTX 4090에서만 수행했다. 각 JSON의 `evaluation_environment`에 평가 환경을 따로 기록한다.

nano 전체 네트워크를 학습하며 고정 레이어는 0개다. batch와 nominal batch는 모두 8, accumulation 없이 epoch당 64 update, 15 epoch에서 960 update와 7,680회 학습 예제를 계획했다. 실제 값은 run의 `training_evidence.json`으로 확인한다. AdamW lr `1e-4`, 최종 LR 비율 `0.1`, weight decay `1e-4`, warmup 0, `dlog=1`, `dgrad=0.5`, `dlam=0.5`, workers 0, deterministic seed를 사용한다. 기하·색상 증강은 끄고 FP32(`amp=False`)로 실행한다. 실제 인자는 run의 `args.yaml`에 남긴다.

vendor는 완료 checkpoint의 optimizer state를 제거한다. 새 run에서 저장 가중치를 시작점으로 추가 적응할 수 있지만 완전한 optimizer resume은 아니다. best/last 두 checkpoint만 보존한다. 원격 CUDA 로드·학습·평가는 `/tmp/neurovector-vision-finetune-gpu.lock`의 exclusive `flock` 안에서 수행하며 각 실행 종료 시 반환한다. 잠금 후 여유 VRAM을 재확인하고 batch>1 학습에는 12,000MiB 이상을 요구한다. 모델의 depth task와 nano scale을 검사해 다른 모델은 거부한다. 로컬에서는 확정된 두 가중치의 비교 평가만 수행하며 로컬 학습이나 checkpoint 재선택은 없다.

## 테스트 누수 방지와 미터 지표

`DepthTrainer.final_eval`은 validation 데이터로 checkpoint calibration을 변경한다. 따라서 생성 YAML에는 **test 항목이 없고** 학습에 `split=val`만 전달한다. test는 epoch 선택, 보정 fitting과 하이퍼파라미터 결정에 사용하지 않는다. 최종 test 추론은 저장된 calibration을 그대로 적용하며 추가 보정하지 않는다.

원본 640×480 GT에서 `0.001 < GT < 10m`인 픽셀만 사용해 이미지별 metric을 구하고 이미지 간 동일 가중 평균을 낸다. GT 결측을 채우지 않는다.

- `abs_rel = mean(abs(pred-GT)/GT)`
- `rmse_m = sqrt(mean((pred-GT)^2))`
- `delta1 = mean(max(pred/GT,GT/pred) < 1.25)`

JSON의 `raw_metric`은 GT 기반 스케일 정렬과 예측 상한 clipping이 없다. `vendor_align_none_clipped`는 vendor `align="none"`과 `[0.001,10]`m clipping을 사용한다. `vendor_align_median_clipped`는 각 test 이미지의 GT 중앙값으로 스케일을 맞춘 참고 지표이며, 실제 서비스에서 사용할 수 없는 정답 기반 보정이므로 미터 정확도의 근거가 아니다. 유효 GT의 비정상/음수 예측은 조용히 제외하지 않고 실패시킨다. [공식 성능표](https://docs.ultralytics.com/tasks/depth/)의 TTA·정렬 지표와 이번 TUM subset 단일 추론 결과는 직접 비교하지 않는다.

각 모델 JSON에는 가중치·manifest SHA256, calibration 계수, 이미지별 metric과 유효 픽셀 수가 있다. 가중치 변화량, optimizer update, 실제 epoch, 파라미터 수, GPU 메모리 사용량과 소요 시간은 `training_evidence.json`에 기록한다. 최종 결론은 실제 raw 결과에 따르며 개선을 전제하지 않는다.

## 검증과 제한

합성 16-bit PNG와 실제 TGZ fixture로 결정적 준비, 미터 변환, split 분리, 원본 해시, 덮어쓰기 방지, 경로 이탈·변조·비정상 예측 처리, 스케일 정렬이 raw 오차를 가리는 사례를 검사한다.

```bash
cd vision
.venv/bin/ruff check tools/finetune_depth.py tests/test_finetune_depth.py
.venv/bin/pytest tests/test_finetune_depth.py
```

동일 센서·실내 환경과 프레임 상관성의 영향이 남는다. 야외·드론·사람/얼굴별 거리, 시간적 안정성, 서비스 통합, 전체 TUM/NYU/KITTI benchmark 검증은 포함하지 않는다. 이전 로컬 대형 파일럿은 최종 nano 실험에서 제외했고 제가 생성한 해당 checkpoint는 삭제했다.
