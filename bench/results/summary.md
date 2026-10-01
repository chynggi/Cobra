## End-to-end (Colorize click), top_k=3, 10 steps, median of runs

Diffusion and post-processing come from the separate synchronized per-stage run ('-' = not profiled). Total e2e is the production code path without extra synchronization.

| Configuration | Case | Postprocess scale | Diffusion, profiled (s) | Postprocess, profiled (s) | Total e2e (s) | e2e runs (s) | Peak alloc / reserved (GB) | Spill to shared mem (MB, idle≈78) | Output |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Baseline (official app.py) | line_ex0 | 1.50 | 9.95 | 73.81 | **65.51** | 39.4, 91.6 | 11.32 / 13.05 | 2254 | 1008x1440 |
| Baseline (official app.py) | line_ex2 | 1.50 | 9.45 | 44.00 | **59.83** | 44.2, 75.4 | 11.33 / 13.21 | 2574 | 1008x1440 |
| Baseline (official app.py) | shadow_ex0 | 1.50 | 9.46 | 45.53 | **41.33** | 42.5, 40.1 | 11.33 / 13.21 | 2574 | 1008x1440 |
| Baseline (official app.py) | shadow_ex2 | 1.50 | 9.43 | 37.27 | **37.06** | 35.4, 38.7 | 11.27 / 13.32 | 2446 | 1200x1200 |
| Optimized | line_ex0 | 1.50 | 9.53 | 3.76 | **13.90** | 13.9, 13.9, 13.9 | 8.02 / 10.45 | 78 | 1008x1440 |
| Optimized | line_ex2 | 1.50 | 9.56 | 3.76 | **14.10** | 14.1, 14.1, 14.1 | 8.02 / 10.45 | 78 | 1008x1440 |
| Optimized | shadow_ex0 | 1.50 | 9.57 | 3.76 | **14.06** | 14.1, 14.0, 14.1 | 8.02 / 10.45 | 78 | 1008x1440 |
| Optimized | shadow_ex2 | 1.50 | 9.52 | 3.72 | **13.56** | 13.5, 13.6, 13.6 | 7.99 / 10.11 | 78 | 1200x1200 |
| Optimized A | line_ex0 | 1.25 | - | - | **12.72** | 12.8, 12.7, 12.7 | 6.65 / 8.42 | 78 | 840x1200 |
| Optimized A | line_ex2 | 1.25 | - | - | **12.86** | 12.9, 12.9, 12.8 | 6.65 / 8.42 | 78 | 840x1200 |
| Optimized A | shadow_ex0 | 1.25 | - | - | **12.82** | 12.8, 12.8, 12.8 | 6.65 / 8.42 | 78 | 840x1200 |
| Optimized A | shadow_ex2 | 1.25 | - | - | **12.36** | 12.4, 12.4, 12.3 | 6.62 / 8.12 | 78 | 1000x1000 |
| Optimized B | line_ex0 | 1.00 | - | - | **11.74** | 11.7, 11.7, 11.7 | 6.32 / 8.15 | 78 | 672x960 |
| Optimized B | line_ex2 | 1.00 | - | - | **11.94** | 11.9, 11.9, 11.9 | 6.32 / 8.15 | 78 | 672x960 |
| Optimized B | shadow_ex0 | 1.00 | - | - | **11.88** | 11.9, 11.9, 11.9 | 6.32 / 8.17 | 78 | 672x960 |
| Optimized B | shadow_ex2 | 1.00 | - | - | **11.40** | 11.4, 11.4, 11.4 | 6.29 / 8.12 | 78 | 800x800 |
| No postprocess | line_ex0 | - | - | - | **10.16** | 10.2, 10.2, 10.2 | 6.32 / 8.15 | 78 | 672x960 |
| No postprocess | line_ex2 | - | - | - | **10.28** | 10.3, 10.3, 10.3 | 6.32 / 8.15 | 78 | 672x960 |
| No postprocess | shadow_ex0 | - | - | - | **10.31** | 10.3, 10.3, 10.3 | 6.32 / 8.17 | 78 | 672x960 |
| No postprocess | shadow_ex2 | - | - | - | **9.84** | 9.8, 9.8, 9.8 | 6.29 / 8.12 | 78 | 800x800 |

## Improvement vs baseline (median e2e)

| Case | Baseline (s) | Optimized 1.5x (s) | Improvement | Optimized 1.25x (s) | Optimized 1.0x (s) | No postprocess (s) |
|---|---:|---:|---:|---:|---:|---:|
| line_ex0 | 65.51 | 13.90 | 78.8% (4.71x) | 12.72 | 11.74 | 10.16 |
| line_ex2 | 59.83 | 14.10 | 76.4% (4.24x) | 12.86 | 11.94 | 10.28 |
| shadow_ex0 | 41.33 | 14.06 | 66.0% (2.94x) | 12.82 | 11.88 | 10.31 |
| shadow_ex2 | 37.06 | 13.56 | 63.4% (2.73x) | 12.36 | 11.40 | 9.84 |

## Per-stage profile at 1.5x (synchronized, GPU ms, median)


**line_ex0** — baseline total 88.18 s, optimized total 13.98 s

| Stage | Baseline GPU ms | Baseline stage peak alloc MB | Optimized GPU ms | Optimized stage peak alloc MB |
|---|---:|---:|---:|---:|
| retrieval.prep | 73 | 4878 | 56 | 3600 |
| retrieval.clip_query | 2680 | 4940 | - | - |
| retrieval.clip_refs_preproc | 123 | 4888 | - | - |
| retrieval.clip_refs_encode | 1015 | 5113 | - | - |
| retrieval.similarity_sort | 1 | 4887 | - | - |
| retrieval.select_patches | 49 | 4886 | - | - |
| retrieval.grid | 19 | 4886 | - | - |
| diffusion | 9945 | 7754 | 9530 | 6474 |
| A.resize | 18 | 4886 | 24 | 3605 |
| B.test_low_color | 16 | 4895 | 8 | 3634 |
| C.query_image_vae_ | 16 | 4903 | 33 | 3642 |
| D.vae_encode_color | 515 | 7029 | 519 | 6273 |
| E.vae_encode_bw | 513 | 7562 | 419 | 5749 |
| F.concat | 11 | 7032 | 11 | 5732 |
| G.multires | 49086 | 10577 | 1737 | 8214 |
| H.vae_decode | 23559 | 11596 | 978 | 7682 |
| I.to_pil | 79 | 8253 | 29 | 3639 |
| empty_cache | 447 | 4886 | - | - |
| retrieval.clip | - | - | 500 | 3718 |
| retrieval.select | - | - | 50 | 3605 |
| release_cache | - | - | 38 | 3605 |

**line_ex2** — baseline total 58.89 s, optimized total 14.14 s

| Stage | Baseline GPU ms | Baseline stage peak alloc MB | Optimized GPU ms | Optimized stage peak alloc MB |
|---|---:|---:|---:|---:|
| retrieval.prep | 90 | 4878 | 70 | 3600 |
| retrieval.clip_query | 387 | 4940 | - | - |
| retrieval.clip_refs_preproc | 150 | 4891 | - | - |
| retrieval.clip_refs_encode | 4313 | 5194 | - | - |
| retrieval.similarity_sort | 1 | 4890 | - | - |
| retrieval.select_patches | 51 | 4889 | - | - |
| retrieval.grid | 18 | 4889 | - | - |
| diffusion | 9448 | 7757 | 9558 | 6476 |
| A.resize | 20 | 4889 | 25 | 3607 |
| B.test_low_color | 15 | 4897 | 9 | 3636 |
| C.query_image_vae_ | 14 | 4906 | 34 | 3644 |
| D.vae_encode_color | 512 | 7032 | 518 | 6275 |
| E.vae_encode_bw | 512 | 7565 | 419 | 5751 |
| F.concat | 11 | 7034 | 11 | 5735 |
| G.multires | 26060 | 10579 | 1737 | 8216 |
| H.vae_decode | 16773 | 11598 | 976 | 7684 |
| I.to_pil | 79 | 8255 | 29 | 3641 |
| empty_cache | 423 | 4889 | - | - |
| retrieval.clip | - | - | 605 | 3758 |
| retrieval.select | - | - | 48 | 3607 |
| release_cache | - | - | 42 | 3607 |

**shadow_ex0** — baseline total 57.18 s, optimized total 14.16 s

| Stage | Baseline GPU ms | Baseline stage peak alloc MB | Optimized GPU ms | Optimized stage peak alloc MB |
|---|---:|---:|---:|---:|
| retrieval.prep | 94 | 4878 | 71 | 3600 |
| retrieval.clip_query | 376 | 4940 | - | - |
| retrieval.clip_refs_preproc | 149 | 4891 | - | - |
| retrieval.clip_refs_encode | 1138 | 5194 | - | - |
| retrieval.similarity_sort | 1 | 4890 | - | - |
| retrieval.select_patches | 52 | 4889 | - | - |
| retrieval.grid | 19 | 4889 | - | - |
| diffusion | 9465 | 7757 | 9566 | 6476 |
| A.resize | 18 | 4889 | 24 | 3606 |
| B.test_low_color | 14 | 4897 | 8 | 3635 |
| C.query_image_vae_ | 15 | 4906 | 31 | 3644 |
| D.vae_encode_color | 514 | 7032 | 521 | 6274 |
| E.vae_encode_bw | 511 | 7565 | 419 | 5750 |
| F.concat | 11 | 7034 | 11 | 5734 |
| G.multires | 27738 | 10579 | 1737 | 8215 |
| H.vae_decode | 16633 | 11598 | 981 | 7684 |
| I.to_pil | 79 | 8255 | 28 | 3640 |
| empty_cache | 342 | 4889 | - | - |
| retrieval.clip | - | - | 611 | 3757 |
| retrieval.select | - | - | 51 | 3607 |
| release_cache | - | - | 36 | 3606 |

**shadow_ex2** — baseline total 47.86 s, optimized total 13.59 s

| Stage | Baseline GPU ms | Baseline stage peak alloc MB | Optimized GPU ms | Optimized stage peak alloc MB |
|---|---:|---:|---:|---:|
| retrieval.prep | 33 | 4878 | 15 | 3600 |
| retrieval.clip_query | 390 | 4940 | - | - |
| retrieval.clip_refs_preproc | 36 | 4883 | - | - |
| retrieval.clip_refs_encode | 296 | 4957 | - | - |
| retrieval.similarity_sort | 1 | 4881 | - | - |
| retrieval.select_patches | 48 | 4880 | - | - |
| retrieval.grid | 18 | 4880 | - | - |
| diffusion | 9427 | 7725 | 9515 | 6446 |
| A.resize | 20 | 4880 | 23 | 3602 |
| B.test_low_color | 15 | 4889 | 7 | 3631 |
| C.query_image_vae_ | 14 | 4897 | 26 | 3639 |
| D.vae_encode_color | 510 | 7007 | 513 | 6250 |
| E.vae_encode_bw | 512 | 7535 | 420 | 5730 |
| F.concat | 11 | 7009 | 11 | 5714 |
| G.multires | 22179 | 10526 | 1720 | 8178 |
| H.vae_decode | 13929 | 11538 | 967 | 7649 |
| I.to_pil | 79 | 8220 | 29 | 3636 |
| empty_cache | 328 | 4880 | - | - |
| retrieval.clip | - | - | 205 | 3640 |
| retrieval.select | - | - | 68 | 3602 |
| release_cache | - | - | 35 | 3602 |

## Steps sweep (optimized, 1.5x)

| Case | Steps | Diffusion only (s) | Full (s) |
|---|---:|---:|---:|
| line_ex0 | 5 | 5.94 | 10.35 |
| line_ex0 | 8 | 8.12 | 12.54 |
| line_ex0 | 10 | 9.54 | 13.94 |
| line_ex0 | 12 | 10.99 | 15.37 |
| line_ex0 | 15 | 13.14 | 17.55 |
| shadow_ex0 | 5 | 5.96 | 10.53 |
| shadow_ex0 | 8 | 8.13 | 12.70 |
| shadow_ex0 | 10 | 9.55 | 14.12 |
| shadow_ex0 | 12 | 11.02 | 15.57 |
| shadow_ex0 | 15 | 13.16 | 17.71 |

## Output quality vs the official output (same inputs, hints, seed)

PSNR at the official 1.5x size (lower-res outputs bicubic-upscaled); colour difference (mean CIE76 dE) and sharpness (Laplacian variance) at the diffusion resolution; hint error = RGB distance between each hint colour and the output inside its 16x16 square; line darkness = mean luminance on line-art pixels.

| Case | Output | PSNR vs official (dB) | max abs diff | identical px | dE vs official | Sharpness | Hint error | Line darkness |
|---|---|---:|---:|---:|---:|---:|---|---:|
| line_ex0 | official app.py 1.5x | inf | 0 | 100.0% | 0.00 | 1323 | 5, 14, 172 | 47.8 |
| line_ex0 | optimized 1.5x | inf | 0 | 100.0% | 0.00 | 1323 | 5, 14, 172 | 47.8 |
| line_ex0 | optimized 1.25x | 39.44 | 50 | nan% | 1.23 | 1294 | 5, 14, 172 | 48.4 |
| line_ex0 | optimized 1.0x | 37.28 | 94 | nan% | 1.92 | 1808 | 4, 13, 172 | 44.7 |
| line_ex0 | no post-process | 31.44 | 205 | nan% | 2.62 | 2405 | 7, 16, 170 | 45.4 |
| line_ex2 | official app.py 1.5x | inf | 0 | 100.0% | 0.00 | 1708 | 146, 49 | 37.8 |
| line_ex2 | optimized 1.5x | inf | 0 | 100.0% | 0.00 | 1708 | 146, 49 | 37.8 |
| line_ex2 | optimized 1.25x | 36.19 | 119 | nan% | 1.47 | 1696 | 147, 51 | 37.9 |
| line_ex2 | optimized 1.0x | 34.17 | 172 | nan% | 2.52 | 3045 | 146, 55 | 32.2 |
| line_ex2 | no post-process | 27.56 | 249 | nan% | 3.65 | 3780 | 146, 47 | 40.8 |
| shadow_ex0 | official app.py 1.5x | inf | 0 | 100.0% | 0.00 | 1749 | 198, 67 | 45.2 |
| shadow_ex0 | optimized 1.5x | inf | 0 | 100.0% | 0.00 | 1749 | 198, 67 | 45.2 |
| shadow_ex0 | optimized 1.25x | 36.36 | 74 | nan% | 1.44 | 1723 | 198, 67 | 45.2 |
| shadow_ex0 | optimized 1.0x | 34.42 | 89 | nan% | 2.46 | 2958 | 198, 69 | 44.5 |
| shadow_ex0 | no post-process | 26.07 | 255 | nan% | 3.97 | 3697 | 197, 67 | 47.0 |
| shadow_ex2 | official app.py 1.5x | inf | 0 | 100.0% | 0.00 | 785 | 14, 165 | 37.3 |
| shadow_ex2 | optimized 1.5x | inf | 0 | 100.0% | 0.00 | 785 | 14, 165 | 37.3 |
| shadow_ex2 | optimized 1.25x | 40.81 | 79 | nan% | 0.50 | 766 | 15, 166 | 37.4 |
| shadow_ex2 | optimized 1.0x | 39.53 | 79 | nan% | 0.90 | 1298 | 16, 167 | 36.7 |
| shadow_ex2 | no post-process | 31.28 | 213 | nan% | 1.38 | 1778 | 13, 163 | 38.3 |

## Top K = 20 (80 reference patches), 1 run

| Implementation | Case | Total e2e (s) | Peak alloc / reserved (GB) | Spill (MB) |
|---|---|---:|---:|---:|
| Baseline | shadow_ex0 | 779.75 | 23.39 / 31.56 | 22828 |
| Optimized | shadow_ex0 | 87.85 | 11.52 / 14.15 | 3374 |
| Optimized | line_ex2 | 78.78 | 11.52 / 14.29 | 3438 |
