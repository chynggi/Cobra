# Cobra RTX 3060 12GB 최적화 보고서

- 환경: RTX 3060 12GB (WDDM, 드라이버 sysmem fallback 기본값), i5-10400, RAM 64GB, Windows 11, Python 3.11, torch 2.5.1+cu124, diffusers(번들) / transformers 4.48.3 / gradio 5.22.0 — **의존성 변경 없음**
- 대상: 공식 `app.py` (원본 그대로 보존) → `app_optimized.py` + `cobra_utils/postprocess.py` + `cobra_utils/vram.py`
- 모든 수치는 이 PC에서의 실측값. 원자료: `bench/results/*.json`, `*.log`. 전체 표: `bench/results/summary.md` (`python bench/summarize.py`로 재생성)
- 조건: 공식 Quick Examples 4개(line ex0/ex2, line+shadow ex0/ex2), 고정 seed, 동일 color hint(UI의 `get_color_value`/`draw_square`를 그대로 호출해 배치), **Top K = 3 (UI 기본값)**, 10 steps. 공식/최적화는 각각 별도 프로세스에서 실행

---

## 1. 결론 요약

| | 공식 app.py | 최적화 (1.5×, 기본값) |
|---|---:|---:|
| Colorize 전체 시간 (4개 예제 중앙값) | **37.1 – 65.5 s** (개별 실행 35.4 – 91.6 s) | **13.6 – 14.1 s** (개별 실행 편차 ±0.1 s) |
| Post-processing | 37 – 74 s | **3.7 – 3.8 s** |
| 최대 VRAM (allocated / reserved) | 11.3 / 13.2 GB | 8.0 / 10.5 GB |
| 시스템 RAM으로 넘친 GPU 메모리 (spill) | **2.3 – 2.6 GB** | **0** (유휴 시 값 78 MB와 동일) |
| 출력 | 1008×1440 | **공식과 픽셀 단위로 동일** (4개 예제 모두 PSNR ∞) |
| Top K 20 (80 reference patch) | 779.8 s, spill 22.8 GB | **87.9 s**, spill 3.4 GB, OOM 없음 |

**병목의 정체 (실측):** post-processing이 느린 원인은 연산량이 아니었다. **VRAM 초과분이 Windows 드라이버의 sysmem fallback으로 시스템 RAM에 배치되는 것**이 원인이다. 공식 코드는 post-processing에서 allocated 11.3 GB / reserved 13.2 GB까지 올라가 12 GB를 넘는다. 그 순간부터 MultiResNet(22–49 s)과 VAE decode(14–24 s)가 수십 배 느려진다. 같은 연산이 VRAM 안에서 돌면 MultiResNet은 1.7 s, decode는 1.0 s다. 실행마다 넘치는 양이 달라서 공식 버전은 같은 입력에서도 35–92 s로 흔들린다.

---

## 2. Baseline 프로파일 (공식 app.py, 단계별 동기화 측정, GPU ms)

`bench/baseline_profiled.py`는 공식 `colorize_image` 코드를 그대로 복사하고 timer만 삽입한 것이다. GPU 시간은 `torch.cuda.Event`, wall-clock은 `perf_counter`로 쟀고, 단계마다 `synchronize`하는 것은 프로파일 모드에서만이다.

| 단계 | line_ex0 | line_ex2 | shadow_ex0 | shadow_ex2 | 단계 peak alloc |
|---|---:|---:|---:|---:|---:|
| line extraction (전처리 버튼) | 390 | 318 | 515 | 334 | – |
| reference retrieval (CLIP fp32 포함) | 3 960 | 5 010 | 1 830 | 822 | 5.2 GB |
| diffusion (10 steps) | 9 945 | 9 448 | 9 465 | 9 427 | 7.8 GB |
| A. colorized resize | 18 | 20 | 18 | 20 | |
| B. test_low_color 생성 | 16 | 15 | 14 | 15 | |
| C. query_image_vae_ 생성 | 16 | 14 | 15 | 14 | |
| D. VAE encode #1 (colorized) | 515 | 512 | 514 | 510 | 7.0 GB |
| E. VAE encode #2 (line art) | 513 | 512 | 511 | 512 | 7.6 GB |
| F. hidden_list concat | 11 | 11 | 11 | 11 | |
| **G. MultiResNet** | **49 086** | **26 060** | **27 738** | **22 179** | **10.6 GB** |
| **H. VAE decode** | **23 559** | **16 773** | **16 633** | **13 929** | **11.6 GB** |
| I. tensor → PIL | 79 | 79 | 79 | 79 | |
| torch.cuda.empty_cache() | 447 | 423 | 342 | 328 | |

resize·변환·concat은 합쳐서 0.15 s 미만이라 병목이 아니다. 같은 1008×1440 해상도에서 encode는 0.5 s인데 MultiResNet과 decode만 수십 초가 걸린다. 이 두 단계는 메모리 피크 구간(10.6 → 11.6 GB)과 정확히 일치한다.

**최적화 후 같은 단계 (GPU ms):** retrieval CLIP 205–611, diffusion 9 515–9 566, encode #1 513–521, encode #2 419–420, MultiResNet **1 720–1 737** (peak 8.2 GB), decode **967–981** (peak 7.7 GB), tensor→PIL 28–29, cache 해제 35–42.

---

## 3. 최종 benchmark (Top K 3, 실제 Colorize 경로, 3회 중앙값)

Diffusion과 Postprocess는 동기화 프로파일 실행에서 잰 값이다. 최적화 1.25×/1.0×는 `e2e − no-postprocess e2e`로 산출했고 `~`로 표시했다. Peak VRAM은 PyTorch allocated / reserved, 단위는 초.

**line + shadow (shadow_ex0, 672×960)**

| Configuration | Steps | Postprocess Scale | Diffusion | Postprocess | Total | Peak VRAM | Spill | Output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline (공식) | 10 | 1.50 | 9.46 | 45.53 | **41.33** (42.5, 40.1) | 11.33 / 13.21 GB | 2.57 GB | 1008×1440 |
| Optimized | 10 | 1.50 | 9.57 | 3.76 | **14.06** | 8.02 / 10.45 GB | 0 | 1008×1440 |
| Optimized A | 10 | 1.25 | – | ~2.51 | **12.82** | 6.65 / 8.42 GB | 0 | 840×1200 |
| Optimized B | 10 | 1.00 | – | ~1.57 | **11.88** | 6.32 / 8.17 GB | 0 | 672×960 |
| No postprocess | 10 | – | – | 0 | **10.31** | 6.32 / 8.17 GB | 0 | 672×960 |

**line (line_ex0, 672×960)**

| Configuration | Steps | Postprocess Scale | Diffusion | Postprocess | Total | Peak VRAM | Spill | Output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline (공식) | 10 | 1.50 | 9.95 | 73.81 | **65.51** (39.4, 91.6) | 11.32 / 13.05 GB | 2.25 GB | 1008×1440 |
| Optimized | 10 | 1.50 | 9.53 | 3.76 | **13.90** | 8.02 / 10.45 GB | 0 | 1008×1440 |
| Optimized A | 10 | 1.25 | – | ~2.56 | **12.72** | 6.65 / 8.42 GB | 0 | 840×1200 |
| Optimized B | 10 | 1.00 | – | ~1.58 | **11.74** | 6.32 / 8.15 GB | 0 | 672×960 |
| No postprocess | 10 | – | – | 0 | **10.16** | 6.32 / 8.15 GB | 0 | 672×960 |

**개선폭 (1.5×, 공식과 같은 출력)**

| Case | Baseline | Optimized | Improvement |
|---|---:|---:|---:|
| line_ex0 | 65.51 s | 13.90 s | **78.8 % (4.71×)** |
| line_ex2 | 59.83 s | 14.10 s | **76.4 % (4.24×)** |
| shadow_ex0 | 41.33 s | 14.06 s | **66.0 % (2.94×)** |
| shadow_ex2 | 37.06 s | 13.56 s | **63.4 % (2.73×)** |

다른 시점에 다른 프로세스로 반복 측정하면 같은 코드도 최대 약 1 s 차이가 났다(GPU 온도와 클럭 영향).

**Inference steps (최적화, 1.5×, diffusion-only / full)**

| Steps | shadow_ex0 | line_ex0 | 육안 품질 (`bench/results/steps_*.png`) |
|---:|---|---|---|
| 5 | 5.96 / 10.53 s | 5.94 / 10.35 s | 머리카락·질감이 흐리고 노이즈가 남음 |
| 8 | 8.13 / 12.70 s | 8.12 / 12.54 s | 10 steps와 거의 구분 안 됨 |
| 10 | 9.55 / 14.12 s | 9.54 / 13.94 s | 기준 |
| 12 | 11.02 / 15.57 s | 10.99 / 15.37 s | 10과 차이 거의 없음 |
| 15 | 13.16 / 17.71 s | 13.14 / 17.55 s | 10과 차이 거의 없음 |

step당 약 0.72 s다. full − diffusion(= retrieval + post-processing)은 step 수와 무관하게 약 4.5 s다.

**Top K 20 (80 reference patch, 1회)**

| | shadow_ex0 | line_ex2 |
|---|---|---|
| 공식 | 779.8 s, peak alloc 23.4 GB / reserved 31.6 GB, spill 22.8 GB | 측정 생략 (shadow_ex0 하나에 warm-up 포함 30분 이상 소요) |
| 최적화 | **87.9 s**, 11.5 / 14.2 GB, spill 3.4 GB | **78.8 s**, spill 3.4 GB |

80개 reference의 K/V cache만 약 6.5 GB라서 12 GB 안에 다 들어가지 않는다. 최적화 버전은 저메모리 모드(아래 4절)로 OOM 없이 끝나고 공식 대비 약 9배 빠르지만, Top K 20 자체는 여전히 느리다. 아래 추천 설정을 참고하라.

---

## 4. 무엇을 바꿨고, 각각 얼마나 효과가 있었나

### 채택 (모두 실측 근거)

| 변경 | 위치 | 효과 (실측) | 결과 동일성 |
|---|---|---|---|
| line-art 쪽 VAE encode에서 mid-block(attention), conv_out, quant_conv 생략. 공식 코드도 이 출력은 버리고 skip feature만 썼다 | `encode_skip_features` | encode #2 512 → 420 ms. 1차 실험에서 공식 13.06 s → 4.49 s(spill 2.06 → 0.72 GB 감소 포함) | **bit-exact** |
| decode 중 사용한 skip feature 즉시 해제 | `decode_with_skip_features` | 최대 해상도 feature(743 MB)가 마지막 up block 동안 메모리에 남지 않음 | **bit-exact** |
| MultiResNet: in-place ReLU, 처리한 입력 즉시 해제 | `LeanMultiHiddenResNetModel` | MultiResNet 단계 peak 10.6 → 8.2 GB | **bit-exact** (공식과 같은 training-mode BN) |
| PIL→Tensor 정규화와 Tensor→PIL 양자화를 GPU에서 수행 (CPU로는 uint8만 전송) | `pil_to_normalized_tensor`, `tensor_to_pil` | B 15 → 8 ms, I 79 → 29 ms | **bit-exact** |
| CLIP image encoder fp32 → **fp16** | `CLIP_DTYPE` | retrieval CLIP 1 410 → 455 ms(동일 입력 비교), VRAM −1.26 GB | top-k로 선택된 patch와 **전체 유사도 순위가 fp32와 동일** (2개 케이스) |
| `argsort` 전체 정렬 → `topk` | `run_colorization` | 원래 1 ms라 시간 이득은 없음 | 선택 patch와 순서 동일 |
| **phase 경계(diffusion 전, post-processing 전)에서, 쓰지 않는 cache가 1 GB를 넘을 때만 `empty_cache()`** | `release_unused_cache` | 1회 35–42 ms. 끄면 reserved 11.8–12.1 GB, spill 0.40–0.76 GB, e2e 14.1 / 15.1 / 17.8 s, line_ex0는 **233 s**까지 악화. 켜면 reserved 10.5 GB, spill 0 | 결과 무관 |
| 공식 코드가 매 실행 끝에 무조건 호출하던 `empty_cache()` 제거 | | 매번 0.33–0.45 s 절약 | 결과 무관 |
| K/V cache 추정 기반 **저메모리 모드** (ComfyUI 방식): CLIP을 CPU로 offload하고 pipeline VAE `enable_slicing()`(reference patch를 1장씩 encode) | `reference_kv_cache_bytes`, `run_colorization` | Top K 3에서는 발동하지 않아 비용 0. Top K 20에서는 단일 3.08 GiB activation이 사라지고 CLIP 1.26 GB를 확보해 OOM 없이 완료. CLIP은 다음 retrieval 때 GPU로 복귀(~0.2 s) | Top K 3 diffusion 출력이 공식과 **픽셀 동일**. slicing을 항상 켜면 cuDNN 반올림 차이로 diffusion 출력이 PSNR 56–61 dB만큼 달라져서 조건부로 바꿈 |
| UI: `Enable Post Processing` 체크박스, `Postprocess Scale` 슬라이더(0.75–1.5, 기본 1.5) | Gradio | 위 표의 1.25×, 1.0×, No postprocess 선택 가능 | 기본값은 공식과 동일 |
| `COBRA_BENCHMARK=1`일 때만 단계별 synchronize와 로그 | `StageTimer` | production 경로에는 동기화가 없음 | – |

### 측정 후 기각

| 후보 | 결과 | 이유 |
|---|---|---|
| `set_per_process_memory_fraction`으로 allocator 상한 설정 | Top K 3에서는 spill 0. 하지만 한 번 설정하면 해제할 수 없어 Top K 20이 **CUDA OOM**. 출력도 공식과 PSNR 56–60 dB로 달라짐(cuDNN 알고리즘 선택이 바뀜) | 공식(느리지만 완료)보다 나빠짐 → phase 경계 cache 해제로 대체 |
| MultiResNet BatchNorm을 conv에 fold (eval-mode) | 공식 대비 PSNR 36 dB | **공식 app은 `MultiResNetModel.eval()`을 호출하지 않아 BN이 이미지별 통계(training mode)로 동작**한다. fold/eval은 다른 계산이 되므로 training mode를 유지 |
| VAE tiling | PSNR 22 dB, 결과 붕괴 | `tiled_decode`가 skip feature(hidden_list)를 무시함 |
| VAE slicing (post-processing) | 효과 없음 | batch=1에서는 실행되지 않는 코드 경로 |
| bf16 VAE + MultiResNet | 속도 이득 없음, fp16 대비 PSNR 52–54 dB | Ampere에서 fp16과 tensor core 처리량이 같고 정밀도만 낮음 |
| line extraction 모델 fp16/bf16 | line_ex0는 PSNR 61 dB였지만 shadow_ex0는 **PSNR 21 dB (최대 83 차이)** | 입력에 따라 선화가 바뀜 → fp32 유지 (0.15 s 절약 포기) |
| CLIP bf16 | top-k는 같지만 전체 순위가 fp32와 달라짐 | fp16 채택 |
| `cudnn.benchmark`, channels_last MultiResNet | 이득 없음 / 더 느림 | |
| level-0 strip 분할 MultiResNet | fp32에서 결과는 동일했으나 fragmentation으로 reserved가 증가 | 이득 없음 |
| line-art VAE feature 캐시 (같은 선화를 다시 채색할 때 재사용) | 0.4 s 절약. 대신 0.56 GB가 상주해 오히려 느려진 경우가 있었음 | 제거 |
| CLIP 상시 offload | 왕복 0.6 s 추가 (GPU→CPU 0.40 s, CPU→GPU 0.20 s) | 메모리 부족이 예상될 때만 offload |
| PEFT LoRA를 base weight에 merge | diffusion 11.44 → 10.59 s (−7.4 %), 미병합 대비 PSNR 57 dB | **미적용**: style 전환 시 base weight를 복원하는 로직이 필요하고, 출력이 fp16 반올림만큼 공식과 달라짐. 원하면 다음 단계로 적용 가능 |
| 모델을 fp16 파일로 미리 저장 | 로드 27.6 s 중 파일 읽기는 상위 비용에 없음(OS 캐시). 대부분 모델 생성, 랜덤 초기화, import | 이득이 미미함. 실제로 줄이려면 meta-device 초기화(구조 변경)가 필요 |

### `torch.inference_mode` / autocast
- CLIP retrieval과 post-processing은 `inference_mode`로 전환했다(오류 없음, 결과 동일). diffusion pipeline은 공식대로 `no_grad`를 유지했다.
- 대형 모델이 모두 이미 fp16이라 autocast를 적용할 곳이 없었다.

---

## 5. 품질 영향

같은 입력·hint·seed로 만든 최종 출력을 비교했다. `bench/results/compare_*.png`에 원본, 최적화, 1.25×, 1.0×, no-postprocess 출력과 10배 증폭한 차이맵이 있다.

| 출력 | 공식 대비 PSNR | 최대 픽셀 차 | ΔE (색) | 선명도 (Laplacian) | hint 색 오차 | 선 보존 (선 픽셀 밝기) |
|---|---:|---:|---:|---|---|---|
| **최적화 1.5× (기본)** | **∞ (4개 모두 픽셀 동일)** | 0 | 0 | 공식과 동일 | 공식과 동일 | 공식과 동일 |
| 최적화 1.25× | 36.2 – 40.8 dB | 50 – 119 | 0.50 – 1.47 | 공식과 거의 동일 | 공식 ±2 | 공식 ±0.6 |
| 최적화 1.0× | 34.2 – 39.5 dB | 79 – 172 | 0.90 – 2.52 | 출력 해상도가 낮음 | 공식 ±6 | line_ex2: 32.2 vs 37.8 (선이 더 진함) |
| No postprocess | 26.1 – 31.4 dB | 205 – 255 | 1.38 – 3.97 | 스크린톤 같은 입자가 남음 | 공식 ±2 | 공식 ±3 |

- **1.5×(기본값)는 공식 app.py와 결과가 완전히 같다.** 따라서 색상 hint, reference 색, 선화, shadow, 이미 채색된 눈·홍조·혀·헤일로가 모두 그대로다.
- 1.25×와 1.0×는 공식 과정을 낮은 해상도로 돌리는 선택 옵션이다. 색은 유지되고 차이는 선과 가장자리의 질감에 몰려 있다(차이맵 참고). no-postprocess는 피부와 불꽃에 입자 노이즈가 남는다.
- Top K 20은 여전히 메모리가 부족해 일부 spill이 생긴다. 결과는 정상적으로 완료된다.

---

## 6. RTX 3060 12GB 추천 설정

| 항목 | 추천 | 근거 |
|---|---|---|
| steps | **10** (빠르게: 8) | 8로 줄이면 약 −1.4 s, 육안 차이는 거의 없음. 5는 품질 저하가 보임 |
| postprocess scale | **1.5** (기본) / 미리보기 **1.0** | 1.5도 이제 14 s 안팎. 1.0은 약 −2.2 s, 출력 해상도 672×960 |
| style | 용도에 맞게 (line / line + shadow 속도 동일) | 두 style의 시간 차이가 측정 오차 범위 |
| top_k | **3 – 6** | Top K 3: 14 s, spill 0. Top K 20: 79–88 s (K/V cache만 6.5 GB로 12 GB 초과) |
| dtype | DiT/ControlNet/VAE/MultiResNet **fp16**(공식 유지), CLIP **fp16**, line model **fp32** | 4절 |
| Enable Post Processing | 최종본은 On, 색/hint를 반복 시험할 때는 Off (약 −3.8 s) | |

---

## 7. 파일

| 파일 | 내용 |
|---|---|
| `app_optimized.py` | 공식 app.py 기반 최적화 버전 (UI 기능은 전부 유지하고 옵션 2개 추가) |
| `cobra_utils/postprocess.py` | `postprocess_colorized_image()` 등 post-processing 함수화 |
| `cobra_utils/vram.py` | phase 경계 cache 해제, 가용 VRAM 계산 |
| `cobra_optimization.patch` | 위 코드 전체 diff (`app.py` → `app_optimized.py` + 신규 파일) |
| `bench/` | 측정 스크립트, 동등성 테스트, 결과(`results/`) |

실행:
```bash
python app_optimized.py
```
단계별 프로파일:
```bash
COBRA_BENCHMARK=1 python app_optimized.py
```
재측정: `python bench/test_equivalence.py`, `python bench/run_baseline.py --impl app` 또는 `--impl app_optimized [--postprocess-scale 1.25] [--no-postprocess] [--top-k 20]`, 이후 `python bench/summarize.py`.

참고: `bench/results/experiments.json`(1차 실험)은 이전 버전 스크립트의 결과다. 이후 제거한 변형(eval-mode BN fold, strip 분할, feature cache)이 들어 있다. 또 이 실험의 line_ex0 케이스는 같은 프로세스에서 CLIP과 line model의 dtype을 바꿨다 되돌리는 실험 뒤에 실행되어, 입력이 기준선과 다르다. 다만 케이스 안에서 변형끼리 비교한 결과는 유효하다.
