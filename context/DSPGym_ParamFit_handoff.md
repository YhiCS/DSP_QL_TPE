# DSPGym_ParamFit — Session Handoff
Date: 2026-05-09 (updated)

## Project Goal
Fit Q-learning parameters to individual participant navigation routes in the DSP maze task using Bayesian optimization (Optuna TPE). One job per (participant × environment) on SLURM.

**Benchmark:** grid sweep achieves ~70% of participants above 0.5 similarity. BO started at ~50% and the session worked toward closing that gap.

---

## Capstone Summary (for sharing with researchers)

- **Learning:** The agent imitates a guided tour of the maze, then freely explores using a softmax policy that gradually shifts from random to exploitative across 500 episodes.
- **Fitting:** Nine parameters governing learning rate, discounting, temperature annealing, and reward structure are tuned per participant via Bayesian optimisation (TPE) to maximise how closely the simulated route matches the human's actual path.
- **Similarity:** Routes are compared as weighted directed graphs — each edge scored by how often both human and agent traversed it — giving a [0, 1] overlap metric that penalises pacing and rewards matching the participant's specific detour pattern.

---

## File
`scripts/DSPGym_ParamFit.py`

---

## The 9 Parameters (PARAM_SPACE)

| Param | Range | Role |
|---|---|---|
| alpha | (0.01, 0.5) | Learning rate |
| gamma | (0.5, 0.99) | Discount factor |
| temperature | (0.1, 20.0) | Initial softmax exploration |
| temp_end | (0.001, 5.0) | Final softmax temperature |
| temp_curve | (0.5, 4.0) | Annealing shape |
| derailPenalty | (-200, -1e-5) | Penalty for leaving route |
| stepPenalty | (-10, -1e-5) | Per-step penalty |
| goalReward | (1, 200) | Reward for reaching goal |
| default_Q_Strength | (0.01, 3.0) | Initial Q-value strength |
| rev_strength | (0.0, 5.0) | HER reversal strength |

Keys absent from both SHARED_KEYS and PERPAR_KEYS use their `_worker` default values and are not optimized.

---

## Design Decisions & What Was Tried

### Sampler choice: TPE (multivariate=True)
- CMA-ES was evaluated and **rejected** — Q-learning landscapes are rugged/multimodal due to discrete state transitions; CMA-ES assumes smooth Gaussian landscapes and performs worse.
- TPE with `multivariate=True` models the joint distribution and handles rugged landscapes better.
- MAP-Elites: interesting for participant-type discovery (future work) but not suited to single-objective convergence. Also ruled out — diversity and similarity are correlated, so MAP-Elites' diversity-as-objective framing doesn't map cleanly here.
- Multi-objective (NSGA-II): **ruled out** — switching from TPE loses sample efficiency; "unique routes" as a second objective is easily gamed by low-similarity params; and the Pareto front complicates Stage 2 warm-starting.

### TPE sampler knobs
- **`gamma`**: callable `(n: int) → int` returning the number of "good" trials. Default ≈ `lambda n: min(int(ceil(0.1 * n)), 25)` (top 10%). To increase exploration: `lambda n: max(1, int(n * 0.25))`. **Not a float — passing a float raises `TypeError`.**
- **`prior_weight`**: weight of the uniform prior relative to the KDE. Higher = more exploratory, self-attenuating as trials accumulate. `1.5` confirmed better than `2.0` (see below).
- **`n_ei_candidates`**: candidates sampled to maximise EI (default 24). Higher = more exploitative surrogate.
- **`group`**: only relevant for conditional parameter spaces (some params only appear in certain branches). No effect here — all params always suggested unconditionally.

### prior_weight experiment
- `prior_weight=1.5`: early stop fires at ~100 trials, ~15-16 min. **Confirmed best.**
- `prior_weight=2.0`: too exploratory — hit 300-trial limit (48 min), similarity dropped 0.01 vs 1.5. TPE never built enough confidence to converge. **Revert to 1.5 pending.**
- Runtime calibration: each `run_q` call ≈ 1–2 s; 20 pairs × 5 seeds / 20 cores ≈ 9–10 s per trial.

### N_EVAL_SEEDS experiment
- `N_EVAL_SEEDS=10`: landscape too smooth — `_mean_best` stabilises immediately after warmup (trial ~100), early stop fires with only ~20 TPE-guided trials. **Not recommended.**
- `N_EVAL_SEEDS=5`: current production setting. Balances smoothing with sufficient TPE exploitation time.
- `N_EVAL_SEEDS=1` restores fast/single-seed mode but brittle.

### Seed design: base seed + N_EVAL_SEEDS averaging
- `STUDY_SEED` is the base seed. Each `_worker` call evaluates over `N_EVAL_SEEDS` consecutive seeds and returns **mean similarity** and **SD**.
- **Why averaging:** `run_q` showed large inter-seed variance. Single-seed objectives caused S2 results to score lower than S1 cherry-picks even with the same params.
- Route stored is always from the **canonical (first) seed** for consistent cherry-picking.
- **SD column (`similarity_sd`):** population SD across N_EVAL_SEEDS seeds. High SD flags params sensitive to random initialisation.

### Three-study sequential design (tried and abandoned)
- Ran three sequential studies, warm-starting each with the top-5 trials from the previous.
- Result: slightly worse than single study. Pooling trials across studies with different seeds makes scores incomparable.

### Key insight on the two metrics in Stage 1
- `study.best_value` = mean similarity of ONE param set across ALL pairs for that trial (what TPE optimizes).
- `_EarlyStop._mean_best` = mean of each pair's all-time best across all trials (accumulated cherry-pick proxy).
- These diverge: best_value ~0.4478, _mean_best ~0.6250 in a typical run.
- `_mean_best` is the RIGHT early stop criterion.

### Why cherry-picking outperforms Stage 2
- Stage 1 runs all pairs with ONE param set per trial. Each pair cherry-picks its rank-1 trial independently — 300 diverse shots across the full space.
- Stage 2 per-pair TPE only sees its own pair's signal (noisier surrogate) and has fewer effective shots.
- With SHARED_KEYS == PERPAR_KEYS: `fixed_params` is overwritten by Optuna suggestions every trial, so trial 0 = cherry-pick exactly → S2 ≥ S1_pair is guaranteed by trial 0.

### Stage 2 monotonicity guarantee
- S2_pair output always falls back to S1_pair if Stage 2 didn't improve.
- S2_pair row is written for **every** pair (above-threshold pairs copy S1_pair directly).
- Use `stage == "S2_pair"` as the single final-outcome filter.

### Stage 2 multi-restart (new this session)
- **Problem:** Stage 2 only dispatches `len(s2_pairs)` tasks. With ~10 below-threshold pairs and 20 SLURM cores, 10 cores sit idle for the entire Stage 2 duration.
- **Solution:** `n_restarts = max(1, processes // len(s2_pairs))` — fills idle cores with independent restarts per pair. Each restart uses a different seed offset (`STUDY_SEED + r * 1000`). Fewer below-threshold pairs → more restarts per pair → more exploration where it's needed most.
- Result collection: `s2_results` is `{pid: [(study, rs), ...]}` (list per pair). DataFrame builder merges all restarts' trials into one sorted list and cherry-picks the best unique route across all restarts.

### Conditional parameter space (evaluated, not implemented)
- Natural candidates: annealing on/off (`temperature + temp_end/temp_curve`), HER on/off (`rev_strength`).
- **Ruled out** in favour of soft boundaries: `temp_end` range overlaps with `temperature`, `rev_strength` starts at 0.0 — TPE can naturally find the "off" state without an explicit categorical branch. Categorical params also disrupt `multivariate=True`'s continuous surrogate.

### CPU usage patterns on SLURM
- **Stage 1 pulsing:** burst (20 workers in parallel per trial) → quiet (TPE surrogate fitting, single-threaded) → next burst. Gap is small (~seconds) relative to worker compute; duty cycle is high.
- **Stage 2 stable 50%:** fixed by multi-restart above.

### Low-similarity participant observation
- Participants with low similarity took detours the algorithm doesn't replicate.
- Temperature controls randomness, not direction — reward structure may be misspecified for these participants.
- Check best-fit temp_end values for low-similarity pairs to see if the ceiling (5.0) is binding.

---

## Current Configuration (as of session end)

```python
STUDY_SEED        = 51121
N_EVAL_SEEDS      = 5      # 3-5 recommended; 10 is too smooth (early stop fires immediately)

SHARED_KEYS = (
    "alpha", "gamma",
    "temperature", "temp_end", "temp_curve",
    "derailPenalty", "default_Q_Strength", "rev_strength",
    # "stepPenalty",
)
PERPAR_KEYS = (
    "derailPenalty", "default_Q_Strength", "rev_strength",
    # "stepPenalty",
)

S2_SIMILARITY_THRESHOLD = 0.5
```

**TPE sampler (Stage 1):** `prior_weight=2.0` — **pending revert to 1.5** (2.0 hit trial limit with worse result).

**Toggle modes:**
- `PERPAR_KEYS = None` → Stage 1 only (group + S1_pair output).
- `SHARED_KEYS = ()` → Stage 2 only, per-pair full-range BO (worse — no warm-start).
- Both non-empty → two-stage hierarchical (current).
- `SHARED_KEYS == PERPAR_KEYS` → guarantees S2 trial 0 = S1_pair (monotonic improvement).

---

## Code Architecture (updated this session)

### Function map

| Function | Role |
|---|---|
| `_worker` | Run one pair over N_EVAL_SEEDS, return (mean similarity, SD, canonical route) |
| `_make_objective` | Objective factory: `pool=None` → sequential (Stage 2); `pool=<Pool>` → parallel starmap (Stage 1). Returns `(objective_fn, route_store)` keyed by `(trial.number, pair_id)` |
| `_run_study` | Shared TPE boilerplate: create study, optional warm-start enqueue, `study.optimize` with EarlyStop. Called by both `_s1_run` and `_s2_run` |
| `_s1_run` | Stage 1: builds pool + objective via `_make_objective`, delegates to `_run_study`. Returns `(study, route_store)` |
| `_s2_run` | Stage 2 worker for one pair: builds objective via `_make_objective(pool=None)`, delegates to `_run_study`. Returns `(pair_id, study, route_store)` |
| `_s1_best_per_pair` | Cherry-picks each pair's best trial from Stage 1 |
| `_EarlyStop` | Callback tracking per-pair accumulated bests; stops when mean stable for `patience` trials |
| `fit_participant` | Main orchestrator: pre-load maps → Stage 1 → cherry-pick → Stage 2 (multi-restart) → build df |

### Parallelisation axes
- **Stage 1:** intra-trial — 20 pairs run in parallel per trial via persistent `Pool`. Pool reused across all trials (no fork/join overhead per trial).
- **Stage 2:** inter-study — N below-threshold pairs × n_restarts studies run in parallel. Each process runs one pair's full TPE study sequentially (no inner pool).

### Stage 1
- TPE study; `n_startup = len(SHARED_KEYS) * 10`; `warmup = n_startup`; `patience = 30`.
- `prior_weight=1.5` (pending revert from 2.0); `multivariate=True`.
- One persistent `Pool` (fork) shared across all trials.
- Objective = mean similarity across all pairs (zeros included).
- Early stop tracks `_mean_best` (per-pair accumulated bests), not `study.best_value`.

### Stage 2
- Runs for pairs where S1_pair similarity < `S2_SIMILARITY_THRESHOLD`.
- `n_restarts = max(1, processes // len(s2_pairs))` — fills idle SLURM cores.
- Each restart: different seed (`STUDY_SEED + r * 1000`), warm-starts from S1 cherry-pick as trial 0.
- `n_startup = max(10, n_dims * 5)`; `warmup = n_startup`; `n_trials = min(n_trials, 300)`.
- Cherry-pick merges all restarts' trials; monotonicity guarantee preserved.

### Output DataFrame schema
Columns: `subjID, env, pair, stage, similarity, similarity_sd, deviation, sim_route, [9 param cols]`

`stage` is an ordered categorical: `"group" < "S1_pair" < "S2_pair"`

| stage | meaning | deviation | similarity_sd |
|---|---|---|---|
| group | Stage 1 group-best params (same params every pair) | 0.0 by definition | SD across N_EVAL_SEEDS |
| S1_pair | Per-pair cherry-picked rank-1 from Stage 1 | L2 dist from group params / range width | SD of cherry-picked trial |
| S2_pair | Final outcome: Stage 2 best if improved, else S1_pair | same metric | SD of winning trial |

`similarity_sd = 0.0` when `N_EVAL_SEEDS = 1`.

### Deviation metric
L2 distance from group params, normalised by each param's full range width, computed only over `s1_best` keys:
```
deviation = sqrt( sum_k ( (param_k - group_k) / (hi_k - lo_k) )^2 )   for k in s1_best
```

---

## Pending / Next Steps

1. **Revert `prior_weight` to 1.5** — 2.0 confirmed worse (hit trial limit, similarity dropped 0.01).
2. **Seed sweep** — change `STUDY_SEED` and compare results to assess Stage 1 robustness. If variance across seeds is high, multi-restart Stage 1 is justified.
3. **Compare SHARED_KEYS == PERPAR_KEYS vs current disjoint config** — disjoint: Stage 2 explores only 3 dims but SHARED_KEYS constraint may limit recovery; equal: Stage 2 is a full per-pair re-fit guaranteed to start at S1_pair.
4. **Investigate low-similarity pairs** — plot per-pair trajectories by stage; check if temp_end ceiling (5.0) is binding.
5. **Potential reward structure investigation** — if low-similarity pairs show systematic detours, the reward function may not capture the participant's actual navigation heuristic.
