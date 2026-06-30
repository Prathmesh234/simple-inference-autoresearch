# CHECKPOINT — read me first next session

_Last updated: 2026-06-30. Branch: `autoresearch/jun30` (forked from `main` @ 876a470,
which has all jun27/28 wins merged). NOTE: every box reads differently — jun30's box
baseline best_agg_tps=9997.7 (vs jun27's 9393, jun11's 8009). Compare WITHIN this branch._

## 30-second TLDR
Autonomously maximizing **`best_agg_tps`** (aggregate decode tok/s, instruct b128, T=1,
measured INSIDE `torch.profiler`) on a pure PyTorch+Triton Llama-3.1-8B engine, single
RTX 6000 Ada (48 GB, 96 MB L2, ~960 GB/s HBM, ~360 TFLOP/s bf16, ~720 TOPS int8). Rules:
`program.md`. Loop: hypothesis → standalone min-bench → implement → parity/coherence →
profiler A/B (iso instruct b1+b128 fast, full sweep to record) → keep if it beats ~5%
sweep noise (use same-process A/B for sub-2% wins) else discard. Log to `results.tsv` +
`workspace/`. Push after every win. NEVER stop to ask.

## jun30 session: best_agg_tps WALLED, frontier pushed hard
The b128 decode HEADLINE is at its int8-GEMM optimum (~10000, +/- ~5% box/thermal noise)
and did NOT move this session — it is converged (see "THE WALL"). Instead jun30 pushed the
OTHER frontier axes (program.md = "push the WHOLE frontier outward"):
| # | change | effect | mechanism |
|---|---|---|---|
| EXP-21 | fused Triton top-k/top-p/sample kernel | headline +0.73% (same-proc A/B; flat in noisy full sweep) | sample() is the only EAGER work post-graph-replay; collapse ~9 ops -> 1. NOTE: CPU dispatch OVERLAPS the replay, so only the ~40us tail GPU-time is saved (not op-count). Small. |
| EXP-22 | weight-only int8 o_proj for b1-b64 | **seq_tps_b1 94.7->100.3 (+5.9%)**, b32/b64 +5-6% | o_proj was the LAST bf16 weight in low-batch decode (bf16 cuBLAS GEMV @ M=1 is inefficient AND HBM-bound). int8 halves the read. Bit-faithful W8A16. b128 unchanged (already W8A8). |
| EXP-23 | W8A8 int8 qkv+gate_up at PREFILL (M>16) | **TTFT -30%** (instruct b128 366->250ms, summarize 14074->9818ms), **peak_vram 35.82->28.72 (-7.1GB)** | int8 act already emitted by add_norm_quant (was unused at prefill); int8 MMA ~2x bf16 at large M. Fused W8A8 swiglu never materialises the (M,2I) bf16 combined (~7GB at b128x1k-tok) -> the VRAM win. Skip bf16 normed at prefill too (emit_bf16=M<=16). |
| EXP-24 | W8A8 int8 down at PREFILL (M>64) | TTFT another -13% (cumulative **-40%**: instruct 366->216ms, summarize 14074->8516ms), peak_vram 28.72->30.30 (down self-quant +1.6GB, still -5.5GB vs baseline) | down was the last bf16-MMA prefill GEMM; self-quant int8 act (~1.8GB transient) fits the headroom EXP-23 freed. |
Current HEAD (pushed): best_agg_tps ~9950 (flat, noise), seq_tps_b1 100.8, peak_vram 30.30.
GOTCHA fixed: added int64 output offsets to _w8a8_gemm/_w8a8_swiglu_fwd (prefill M*N nears int32).

## THE WALL — why best_agg_tps (b128 decode) is converged
b128 decode is HBM-bound on the ~7GB int8 weight stream (~7.3ms floor; measured ~12.8ms =
1.75x, the gap is un-hidden int8 MMA at M=128 + the L2-resident GEMMs' compute time). Every
projection (qkv, gate_up+swiglu, down, o_proj, lm_head) is W8A8 int8 tensor-core, all tiles
re-confirmed optimal on this box (gate_up (128,64,256,8,2)=147us, 1.26x its 122us HBM floor).
The ONLY byte-reduction lever (int4) is DEAD every way (see below). flash (instruct kv~64)
is L2-resident + tile-optimal. sampler fused (EXP-21). => no remaining faithful decode lever.

## CONFIRMED DEAD this session (do NOT retry without a NEW mechanism)
- **int4 gate_up** (the elephant) — DEAD at ALL batch sizes. b128: kernel overhead-bound
  (g=64 forces K=64 dots, hand-Triton 0.81-0.90x, native tinygemm 0.1x; EXP-14/O). b1
  (bandwidth-bound GEMV where the K=64 wall is hidden): FAITHFULNESS-dead — naive int4 g64
  top5 0.80/top50 0.98/rel 0.11; AWQ a=0.5 lifts top5->1.0 but DROPS top50->0.90, rel->0.14
  (trades off, no clean win); rel ~2x the int8 bar. (faith_awq_int4_gateup.py)
- **int4 KV** (incl. GROUPED) — DEAD (faithfulness). Per-vector rel 0.25 (EXP-19); grouped
  g=8/16/32 rel 0.13-0.21, all >>10x the int8 bar (0.014), while byte savings SHRINK with
  finer groups (g=8 only 1.35x vs int8). Can't be both faithful and smaller. (faith_int4_kv_grouped.py)
- **GQA-grouped int8 flash** (EXP-20, the jun28 WIP) — DEAD for the profiler's kv range.
  Faithful (cos 0.99998) but 1.00-1.08x at kv<=1068 (the longest cell); L2 absorbs the
  redundant sibling reads (concurrent working set fits 96MB L2 to ~kv1500). Wins 1.22-1.32x
  only at kv>=2048 (unreached). NOT wired. (bench_gqa_int8_flash.py)
- gate_up tile re-sweep (jun30, optimal), W8A16 b1 tile re-sweep (optimal, 1.0-1.03x alts).
- (cumulative dead from prior sessions still hold: split-K W8A8, transposed layout, grouped
  quant fusion, KV-int8 at SHORT ctx, spec-decode (profiler 1-tok/call incompatible).)

## Frontier state after jun30 (where any remaining gains are)
- **Decode headline (b128)**: WALLED. Needs a Marlin/CUTLASS-grade int4 GEMM (cp.async +
  pre-shuffled weights + in-register dequant overlapping MMA) — multi-day, beyond Triton,
  AND int4 faithfulness is borderline. Not a loop iteration.
- **Interactivity (b1-b64)**: EXP-22 lifted it; b1 GEMMs now all int8, near their floors
  (gate_up 1.14x). int4 too lossy. Near-converged.
- **TTFT/prefill**: EXP-23/24 -40%. Remaining prefill cost is the bf16 flash attention
  (O(T^2) causal) + the small o_proj (cuBLAS, skipped — only ~23ms at instruct b128). An
  int8 prefill flash is a big change + faithfulness risk (prompt attention, not just KV).
- **Long-ctx decode (the 5000-7000 cells)**: flash-dominated (44% at summarize b128),
  int8-KV HBM-bound at 1.28x its read floor (nw=1 occupancy-max, tile-optimal). int4 KV
  dead, GQA L2-absorbed. WALLED.
- **VRAM**: 30.30GB (-5.5GB vs baseline). Lots of headroom (was 35.82).

## Run env (MUST re-set every session — fresh box)
```
export UV_CACHE_DIR=~/.cache/uv XDG_CONFIG_HOME=~/uvconfig PATH=~/.local/bin:$PATH \
       HOME=/home/ubuntu PYTHONPATH=/home/ubuntu/simple-inference-autoresearch HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p ~/uvconfig
source workspace/scripts/_env.sh   # this snippet, written jun30
```
uv install if missing: `curl -LsSf https://astral.sh/uv/install.sh | sh` (then the env above
fixes the ~/.config/uv permission error). torch 2.12.0+cu130, triton 3.7.0. Model cached.
Profiler: `nohup uv run python profiling/profile_engine.py [--flavors instruct --batch-sizes
1 128] > run.log 2>&1 &` (iso ~5min, full sweep ~12min). Grep `^best_agg_tps:|^seq_tps_b1:|
^peak_vram_gb:|^best_agg_at:`. Model load ~3-4min. `results.tsv`+`run*.log` gitignored. Push:
`PAT=$(grep '^GITHUB_PAT=' .env|cut -d= -f2-); GIT_TERMINAL_PROMPT=0 git -c core.askpass= push
"https://x-access-token:${PAT}@github.com/Prathmesh234/simple-inference-autoresearch" HEAD:refs/heads/autoresearch/jun30`.
Git identity: Prathmesh234 / ppbhatt500@gmail.com + `Co-authored-by: Copilot
<223556219+Copilot@users.noreply.github.com>`. Same-process A/B (workspace/scripts/ab_sampler.py)
is the reliable way to measure sub-2% wins (full sweep noise ~5% dwarfs them).
