"""
DSP_QL_TPE.py

Fits Q-learning parameters to individual participant routes using Bayesian
optimization (Optuna TPE sampler).

SLURM: one job per (participant × environment).
  SLURM_ARRAY_TASK_ID indexes into the flat list of (subjID, env_prefix) pairs.

Output: ./dump/paramFit/paramFit_{subjID}_{env}.pkl
  A DataFrame with columns: subjID, env, pair, rank, similarity, deviation,
  sim_route, stage, plus one column per parameter in PARAM_SPACE.

Two-stage design
----------------
Stage 1 (shared):  one TPE study optimizes ALL PARAM_SPACE keys jointly across
  all pairs → a single cohesive parameter profile for the participant.
Stage 2 (per-pair): SHARED_KEYS are fixed to Stage 1's best; PERPAR_KEYS are
  re-optimized independently per pair in a narrow range around Stage 1's best,
  capturing per-trial adaptation (e.g. temperature shifts as the participant
  learns the environment).

Modes (set via SHARED_KEYS / PERPAR_KEYS at the top of this file):
  Both non-empty  →  hierarchical two-stage  (default)
  PERPAR_KEYS=()  →  Stage 1 only — one param set for all pairs
  SHARED_KEYS=()  →  Stage 2 only — per-pair BO on all params (grid-sweep style)

Node format: human routes use raw graph node tuples (snapped_node); simulated
routes are converted from RouteNodeIdx → node tuples via dspMapBuilder so both
sides share the same representation before path_similarity is applied.
"""

import os
import random
import warnings
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import optuna
from optuna.importance import get_param_importances, PedAnovaImportanceEvaluator
from multiprocessing import get_context, cpu_count

import dspMapBuilder
from DSPGym import run_q

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths are resolved relative to this file so the script works from any CWD.
_HERE     = Path(__file__).resolve().parent
DATA_DIR  = _HERE / "data"
DUMP_DIR  = _HERE / "dump" / "paramFit"

# Populated once in the parent before any Pool is created; fork workers
# inherit the pages copy-on-write so loadMap is never called in a worker.
_PAIR_NODES: dict = {}

# ---------------------------------------------------------------------------
# Seed / Stage configuration  ← edit here to change behaviour
# ---------------------------------------------------------------------------
STUDY_SEED   = 51121  # base seed; evaluation seeds are STUDY_SEED … STUDY_SEED+N_EVAL_SEEDS-1
N_EVAL_SEEDS = 5      # seeds averaged per worker call; 1 = fast but brittle, 3-5 = robust

# Keys optimized jointly across all pairs in Stage 1, then fixed in Stage 2.
# Keys optimized independently per pair in Stage 2.
# Keys absent from both use their _worker default values.
SHARED_KEYS = (
               "alpha", "gamma", 
               "temperature", "temp_end", "temp_curve",
               "derailPenalty", "default_Q_Strength","rev_strength",
            #    "stepPenalty",
               )

PERPAR_KEYS = (
            #    "alpha", "gamma", 
            #    "temperature", "temp_end", "temp_curve",
               "derailPenalty", "default_Q_Strength","rev_strength",
            #    "stepPenalty",
               )

PERPAR_KEYS = None

# Stage 2 only runs for pairs whose Stage-1 cherry-pick similarity is below
# this threshold.  Pairs at or above it keep their S1_pair result.
S2_SIMILARITY_THRESHOLD = 0.5
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parameter space  (name → (low, high))
# ---------------------------------------------------------------------------
PARAM_SPACE = {
    # Q-learning dynamics
    "alpha":       (0.01,  0.5),
    "gamma":       (0.5,   0.99),
    "temperature": (0.1,  20.0),
    "temp_end":    (0.001,  5.0),   # final softmax temperature
    "temp_curve":  (0.5,    4.0),   # annealing shape: <1 fast-then-slow, >1 slow-then-fast

    # Reward structure
    "derailPenalty": (-200.0, -1e-5),
    "stepPenalty":   ( -10.0, -1e-5),
    # "goalReward":    (   1.0, 200.0),

    # Route dependence
    "default_Q_Strength": (1e-5, 3.0),
    "rev_strength":       (0.0,  5.0),
}

# ---------------------------------------------------------------------------
# Path similarity  (copied from DSPQL_analysis so this script is self-contained)
# ---------------------------------------------------------------------------
def _path_to_digraph(path):
    G = nx.DiGraph()
    for u, v in zip(path, path[1:]):
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G

def path_similarity(path1, path2):
    if len(path1) < 2 or len(path2) < 2:
        return 0.0
    g1 = _path_to_digraph(path1)
    g2 = _path_to_digraph(path2)
    all_edges = set(g1.edges()) | set(g2.edges())
    if not all_edges:
        return 0.0
    scores = []
    for edge in all_edges:
        if edge in g1.edges() and edge in g2.edges():
            w1, w2 = g1[edge[0]][edge[1]]["weight"], g2[edge[0]][edge[1]]["weight"]
            scores.append(min(w1, w2) / max(w1, w2))
        else:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 4)

# ---------------------------------------------------------------------------
# Worker: run one pair, return similarity to the human route for that pair
# ---------------------------------------------------------------------------
def _worker(pair, human_route, params, seed):
    """Average similarity over N_EVAL_SEEDS consecutive seeds for robustness.
    Route is taken from the canonical (first) seed for consistent cherry-picking.
    """
    nodes = _PAIR_NODES[pair]
    human_max_visits = max(Counter(human_route).values()) if human_route else 1

    scores = []
    canonical_route = []

    for s in range(seed, seed + N_EVAL_SEEDS):
        np.random.seed(s)
        random.seed(s)
        try:
            out = run_q(
                episodes=500,
                map_type="OG",
                pair=pair,
                alpha=params.get("alpha", 0.05),
                gamma=params.get("gamma", 0.95),
                temperature=params.get("temperature", 5),
                temp_end=params.get("temp_end", None),
                temp_curve=params.get("temp_curve", 2),
                derailPenalty=params.get("derailPenalty", -1),
                stepPenalty=params.get("stepPenalty", -1),
                goalReward=params.get("goalReward", 100),
                default_Q_Strength=params.get("default_Q_Strength", 1.0),
                rev_strength=params.get("rev_strength", 1),
                useTour=True,
                max_steps=200,
                plotting=False,
                show_train_progress=False,
            )
        except Exception:
            scores.append(0.0)
            continue

        if out is None or "RouteNodeIdx" not in out:
            scores.append(0.0)
            continue
        route_idx = out["RouteNodeIdx"]
        if not route_idx or len(route_idx) < 2:
            scores.append(0.0)
            continue

        sim_route = [nodes[i] for i in route_idx]

        # Penalize pacing routes: if any node is visited more times than the most-visited
        # node in the human route, the agent is oscillating beyond the human's own
        # backtracking pattern. Uses the human route as a per-pair dynamic threshold so
        # legitimate revisits (47% of human routes have at least one) are not excluded.
        if max(Counter(sim_route).values()) > human_max_visits:
            scores.append(0.0)
            if s == seed:
                canonical_route = sim_route
            continue

        score = path_similarity(sim_route, list(human_route))
        scores.append(score)
        if s == seed:
            canonical_route = sim_route

    return float(np.mean(scores)), float(np.std(scores)), canonical_route

# ---------------------------------------------------------------------------
# Optuna objective factory  (Stage 1 — all pairs in parallel per trial)
# ---------------------------------------------------------------------------
def _make_objective(dspTrial_pairs, fixed_seed,
                    param_space=None, fixed_params=None, *, pool=None):
    """
    Returns (objective_fn, route_store).

    pool=None    : run _worker calls sequentially (single-pair Stage 2 case).
    pool=<Pool>  : run all pairs in parallel via pool.starmap (Stage 1 case).
    param_space  : dict of params to optimize (defaults to full PARAM_SPACE).
    fixed_params : dict of params held constant every trial (e.g. Stage 1 best).
    route_store  : keyed by (trial_number, pair_id) → sim_route list.

    The fixed seed makes the objective a pure function of params so TPE's
    surrogate can attribute score changes to params alone, not to randomness.
    """
    if param_space is None:
        param_space = PARAM_SPACE
    if fixed_params is None:
        fixed_params = {}
    pair_ids = [pair for pair, _ in dspTrial_pairs]
    route_store = {}

    def objective(optuna_trial):
        params = dict(fixed_params)
        params.update({
            name: optuna_trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in param_space.items()
        })
        tasks = [
            (pair, human_route, params, fixed_seed)
            for pair, human_route in dspTrial_pairs
        ]
        if pool is not None:
            results = pool.starmap(_worker, tasks, chunksize=1)
        else:
            results = [_worker(*task) for task in tasks]

        scores = []
        for pair_id, (score, sd, route) in zip(pair_ids, results):
            optuna_trial.set_user_attr(pair_id, float(score))
            optuna_trial.set_user_attr(f"{pair_id}_sd", float(sd))
            route_store[(optuna_trial.number, pair_id)] = route
            scores.append(score)

        valid = scores
        return float(np.mean(valid)) if valid else 0.0

    return objective, route_store
    
# ---------------------------------------------------------------------------
# Parameter importance helper
# ---------------------------------------------------------------------------
# PedAnova normalises in-scope importances to sum=1; do NOT mix Stage 1 and
# Stage 2 importance values arithmetically (their in-scope subsets differ).
def _param_importance(study, *, target):
    """Return {param_name: importance} for every PARAM_SPACE key.
    Keys outside the study's search space (or on any failure) are NaN.
    PedAnova assumes minimization, so callers must negate the metric.
    """
    nan_dict = {k: float("nan") for k in PARAM_SPACE}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp = get_param_importances(
                study,
                evaluator=PedAnovaImportanceEvaluator(target_quantile=0.1),
                target=target,
            )
    except (ValueError, RuntimeError):
        return nan_dict
    return {k: float(imp.get(k, float("nan"))) for k in PARAM_SPACE}


# ---------------------------------------------------------------------------
# Stage 1 runner  (shared-param TPE study across all pairs)
# ---------------------------------------------------------------------------
def _run_study(objective_fn, param_space, pair_ids,
               seed, n_startup, warmup, n_trials, patience, *,
               warmup_params=None, prior_weight=1.0, show_progress_bar=False):
    """
    Create and run one TPE study. Returns the completed study.

    Handles the boilerplate shared by _s1_run and _s2_run:
    study creation, optional warm-start enqueue, optimize, and EarlyStop.
    The objective function and route_store are owned by the caller.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=len(param_space) > 1,
            prior_weight=prior_weight,
        ),
    )
    if warmup_params:
        clamped = {
            k: max(lo, min(hi, warmup_params[k]))
            for k, (lo, hi) in param_space.items()
            if k in warmup_params
        }
        if clamped:
            study.enqueue_trial(clamped)
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        callbacks=[_EarlyStop(pair_ids=pair_ids, patience=patience, warmup=warmup)],
        show_progress_bar=show_progress_bar,
    )
    return study


def _s1_run(dspTrial_pairs, ctx, n_trials, patience, processes):
    """
    Run Stage 1: one TPE study optimizing SHARED_KEYS jointly across all pairs.
    Returns (study, route_store).
    """
    param_space  = {k: PARAM_SPACE[k] for k in SHARED_KEYS}
    n_startup    = len(param_space) * 10
    pair_ids     = [pid for pid, _ in dspTrial_pairs]

    print(f"  [Stage 1] joint fit, seed={STUDY_SEED}", flush=True)
    with ctx.Pool(processes=processes) as pool:
        obj_fn, route_store = _make_objective(
            dspTrial_pairs, STUDY_SEED,
            param_space=param_space, pool=pool,
        )
        study = _run_study(
            obj_fn, param_space, pair_ids,
            seed=STUDY_SEED, n_startup=n_startup, warmup=n_startup,
            n_trials=n_trials, patience=patience,
            prior_weight=2.0, show_progress_bar=True,
        )
    n_done = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    print(f"    {n_done} trials, best={study.best_value:.4f}", flush=True)
    return study, route_store
    
# ---------------------------------------------------------------------------
# Per-pair best from Stage 1: cherry-pick each pair's top trial
# ---------------------------------------------------------------------------
def _s1_best_per_pair(s1_study, s1_route_store, dspTrial_pairs):
    """Return {pair_id: {"params", "similarity", "route"}} for each pair's
    best trial from s1_study (sorted by pair score then group mean)."""
    completed    = [t for t in s1_study.trials if t.state.name == "COMPLETE"]
    all_pair_ids = [pid for pid, _ in dspTrial_pairs]
    result = {}
    for pair_id, _ in dspTrial_pairs:
        sorted_trials = sorted(
            completed,
            key=lambda t: (
                t.user_attrs.get(pair_id, 0.0),
                np.mean([t.user_attrs.get(pid, 0.0) for pid in all_pair_ids]),
            ),
            reverse=True,
        )
        seen = set()
        for t in sorted_trials:
            route = s1_route_store.get((t.number, pair_id), [])
            if tuple(route) not in seen:
                seen.add(tuple(route))
                result[pair_id] = {
                    "params":        dict(t.params),
                    "similarity":    t.user_attrs.get(pair_id, 0.0),
                    "similarity_sd": t.user_attrs.get(f"{pair_id}_sd", 0.0),
                    "route":         route,
                }
                break
    return result

# ---------------------------------------------------------------------------
# Stage 2 runner  (one pair — called via Pool.starmap across all pairs)
# ---------------------------------------------------------------------------
def _s2_run(pair_id, human_route, fixed_params, param_space,
            fixed_seed, n_trials, n_startup, warmup, patience,
            warmup_params=None):
    """
    Run a Stage-2 Optuna study for ONE pair.  SHARED_KEYS are already baked
    into fixed_params; only PERPAR_KEYS (via param_space) are optimized.
    Returns (pair_id, study, route_store).
    """
    obj_fn, route_store = _make_objective(
        [(pair_id, human_route)], fixed_seed,
        param_space=param_space, fixed_params=fixed_params,
    )
    study = _run_study(
        obj_fn, param_space, [pair_id],
        seed=fixed_seed, n_startup=n_startup, warmup=warmup,
        n_trials=n_trials, patience=patience,
        warmup_params=warmup_params,
    )
    return pair_id, study, route_store

# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------
class _EarlyStop:
    """Stop when the mean of per-pair BEST TRUE similarities (from user_attrs)
    hasn't improved by min_delta for `patience` trials after warmup.

    Tracks per-pair bests rather than study.best_value so that the
    duplicate-penalised objective (which guides TPE exploration) does not
    corrupt the stopping criterion.  A perfect match on one pair raises that
    pair's best to 1.0 but does not stop the search; the run continues until
    ALL pairs' bests have converged.
    """
    def __init__(self, pair_ids, patience=30, min_delta=1e-4, warmup=25):
        self.pair_ids       = pair_ids
        self.patience       = patience
        self.min_delta      = min_delta
        self.warmup         = warmup
        self._best_per_pair = {pid: 0.0 for pid in pair_ids}
        self._mean_best     = 0.0
        self._no_improve    = 0

    def __call__(self, study, trial):
        if trial.number < self.warmup:
            return
        for pid in self.pair_ids:
            score = trial.user_attrs.get(pid, 0.0)
            if score > self._best_per_pair[pid]:
                self._best_per_pair[pid] = score
        new_mean = float(np.mean(list(self._best_per_pair.values())))
        if new_mean > self._mean_best + self.min_delta:
            self._mean_best  = new_mean
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            print(
                f"  Early stop at {trial.number}: mean per-pair best similarity={self._mean_best:.4f} "
                f"unchanged for {self.patience} consecutive trials.",
                flush=True,
            )
            study.stop()

# ---------------------------------------------------------------------------
# Fit one participant × environment
# ---------------------------------------------------------------------------
def fit_participant(subj_id, env_prefix, n_trials=300, patience=30, processes=None):
    """
    Returns (s1_study, df) or (None, None) if no data found.

    s1_study : the Stage 1 optuna.Study (None if SHARED_KEYS is empty).
    df       : one row per (pair × stage) with columns:
                 subjID, env, pair, stage, similarity, deviation, sim_route,
                 plus one column per PARAM_SPACE key.
               stage is a categorical: "group" | "S1_pair" | "S2_pair"
                 "group"   — Stage 1 group-best params (same for every pair)
                 "S1_pair" — per-pair cherry-pick from Stage 1 trials
                 "S2_pair" — Stage 2 per-pair re-fit (only pairs below threshold)
               deviation — L2 distance from group params, normalised by range width.

    Stage behaviour is controlled by SHARED_KEYS / PERPAR_KEYS at module level:
      Both non-empty  →  two-stage hierarchical fit  (default)
      PERPAR_KEYS=()  →  Stage 1 only (group + S1_pair rows)
      SHARED_KEYS=()  →  Stage 2 only (all params free per pair)
    """
    allTrajGrid = pd.read_pickle(DATA_DIR / "subjTrajNodes.pkl").reset_index(drop=True)

    mask = (
        (allTrajGrid["SubjectNum"] == subj_id) &
        (allTrajGrid["pairs"].str.startswith(env_prefix))
    )
    subj_env = allTrajGrid[mask]
    if subj_env.empty:
        print(f"  No data for subj={subj_id} env={env_prefix}", flush=True)
        return None, None

    subjRoutes = (
        subj_env
        .groupby(["SubjectNum", "pairs"], as_index=False)["snapped_node"]
        .apply(list)
    )
    subjRoutes.columns = ["subjID", "pairs", "RouteNodes"]
    dspTrial_pairs = list(zip(subjRoutes["pairs"], subjRoutes["RouteNodes"]))

    print(f"  {len(dspTrial_pairs)} dspTrials found", flush=True)

    # Load each pair's node list once in the parent; fork workers inherit via copy-on-write.
    for pair, _ in dspTrial_pairs:
        if pair not in _PAIR_NODES:
            G, *_ = dspMapBuilder.loadMap(map_type="OG", pair=pair)
            _PAIR_NODES[pair] = sorted(G.nodes, key=str)

    if processes is None:
        processes = max(1, cpu_count())

    ctx            = get_context("fork")
    s1_study       = None
    s1_best        = {}
    s1_route_store = {}

    # ── Stage 1: shared param profile across all pairs ───────────────────────
    if SHARED_KEYS:
        s1_study, s1_route_store = _s1_run(dspTrial_pairs, ctx, n_trials, patience, processes)
        s1_best = s1_study.best_trial.params

    # ── Per-pair best from Stage 1 ───────────────────────────────────────────
    s1_pair_best = {}
    if s1_study is not None:
        s1_pair_best = _s1_best_per_pair(s1_study, s1_route_store, dspTrial_pairs)

    # ── Stage 1 parameter importance ─────────────────────────────────────────
    # Group: importance against the mean-similarity objective (one dict, reused).
    # Per-pair: importance against this pair's user_attr similarity (cached for
    # reuse by both the S1_pair row and any S2_pair fallback row).
    nan_imp = {k: float("nan") for k in PARAM_SPACE}
    s1_group_imp = (
        _param_importance(s1_study, target=lambda t: -(t.value or 0.0))
        if s1_study is not None else dict(nan_imp)
    )
    s1_pair_imp = {}
    if s1_study is not None:
        for pid, _ in dspTrial_pairs:
            s1_pair_imp[pid] = _param_importance(
                s1_study,
                target=lambda t, p=pid: -t.user_attrs.get(p, 0.0),
            )

    # ── Stage 2: per-pair re-fit for pairs below similarity threshold ────────
    s2_results  = {}
    fixed_params = {k: s1_best[k] for k in SHARED_KEYS} if SHARED_KEYS else {}
    if PERPAR_KEYS:
        # Use full param range — below-threshold pairs need unconstrained search.
        param_space_s2 = {k: PARAM_SPACE[k] for k in PERPAR_KEYS}
        if not SHARED_KEYS:
            for k in PARAM_SPACE:
                if k not in param_space_s2:
                    param_space_s2[k] = PARAM_SPACE[k]

        n_dims_s2    = len(param_space_s2)
        n_startup_s2 = max(10, n_dims_s2 * 5)
        warmup_s2    = n_startup_s2 #+ 10
        s2_n_trials  = min(n_trials, 300)

        s2_pairs = [
            (pid, route) for pid, route in dspTrial_pairs
            if s1_pair_best.get(pid, {}).get("similarity", 0.0) < S2_SIMILARITY_THRESHOLD
        ]
        print(
            f"  [Stage 2] {len(s2_pairs)}/{len(dspTrial_pairs)} pairs below "
            f"threshold={S2_SIMILARITY_THRESHOLD}, seed={STUDY_SEED}", flush=True,
        )
        if s2_pairs:
            n_restarts = 1#max(1, processes // len(s2_pairs))
            s2_tasks = [
                (pair_id, human_route, fixed_params, param_space_s2,
                 STUDY_SEED + r * 1000, s2_n_trials, n_startup_s2, warmup_s2, patience,
                 s1_pair_best.get(pair_id, {}).get("params"))
                for pair_id, human_route in s2_pairs
                for r in range(n_restarts)
            ]
            print(f"    {n_restarts} restart(s) per pair, {len(s2_tasks)} total S2 tasks",
                  flush=True)
            with ctx.Pool(processes=processes) as pool:
                s2_result_list = pool.starmap(_s2_run, s2_tasks, chunksize=1)
            s2_results = {}
            for pid, study, rs in s2_result_list:
                s2_results.setdefault(pid, []).append((study, rs))

    # ── Build output DataFrame ───────────────────────────────────────────────
    # Deviation: L2 distance from group (Stage 1 best) params,
    # normalised by each param's full range width.
    def _dev(params):
        if not s1_best:
            return 0.0
        diffs = []
        for k in s1_best:
            lo, hi = PARAM_SPACE[k]
            w = (hi - lo) or 1.0
            diffs.append((params.get(k, s1_best[k]) - s1_best[k]) / w)
        return float(np.linalg.norm(diffs))

    param_cols = list(PARAM_SPACE.keys())
    rows = []

    def _imp_cols(imp_dict):
        return {f"imp_{k}": imp_dict.get(k, float("nan")) for k in PARAM_SPACE}

    for pair_id, _ in dspTrial_pairs:
        base = {"subjID": subj_id, "env": env_prefix, "pair": pair_id}
        s1_pair_imp_dict = s1_pair_imp.get(pair_id, nan_imp)

        # "group": Stage 1 best-trial params applied to this pair
        if s1_best:
            group_route = s1_route_store.get((s1_study.best_trial.number, pair_id), [])
            group_sim   = s1_study.best_trial.user_attrs.get(pair_id, 0.0)
            group_sd    = s1_study.best_trial.user_attrs.get(f"{pair_id}_sd", 0.0)
            rows.append({
                **base,
                "stage":         "group",
                "similarity":    group_sim,
                "similarity_sd": group_sd,
                "deviation":     0.0,
                "sim_route":     group_route,
                **s1_best,
                **_imp_cols(s1_group_imp),
            })

        # "S1_pair": per-pair best cherry-picked from Stage 1 trials
        cp = s1_pair_best.get(pair_id)
        if cp:
            rows.append({
                **base,
                "stage":         "S1_pair",
                "similarity":    cp["similarity"],
                "similarity_sd": cp["similarity_sd"],
                "deviation":     _dev(cp["params"]),
                "sim_route":     cp["route"],
                **cp["params"],
                **_imp_cols(s1_pair_imp_dict),
            })

        # "S2_pair": final outcome — Stage 2 best if it improved, S1_pair otherwise.
        # Pairs above threshold skip Stage 2 entirely; they also get an S2_pair row
        # copying S1_pair so callers can always use stage=="S2_pair" as the final result.
        cp = s1_pair_best.get(pair_id)
        if pair_id in s2_results:
            s1_floor = cp["similarity"] if cp else 0.0
            completed_s2 = sorted(
                [
                    (t, rs)
                    for study, rs in s2_results[pair_id]
                    for t in study.trials
                    if t.state.name == "COMPLETE"
                ],
                key=lambda t_rs: t_rs[0].user_attrs.get(pair_id, 0.0),
                reverse=True,
            )
            # Merge restarts into a single study for importance computation only;
            # winner selection above is unchanged.
            merged_s2 = optuna.create_study(direction="maximize")
            for study, _rs in s2_results[pair_id]:
                merged_s2.add_trials(
                    [t for t in study.trials if t.state.name == "COMPLETE"]
                )
            s2_imp = _param_importance(
                merged_s2,
                target=lambda t, p=pair_id: -t.user_attrs.get(p, 0.0),
            )
            s2_row = None
            seen = set()
            for t, rs in completed_s2:
                route = rs.get((t.number, pair_id), [])
                if tuple(route) in seen:
                    continue
                seen.add(tuple(route))
                s2_sim = t.user_attrs.get(pair_id, 0.0)
                if s2_sim > s1_floor:
                    full_params = {**fixed_params, **t.params}
                    s2_row = {
                        **base,
                        "stage":         "S2_pair",
                        "similarity":    s2_sim,
                        "similarity_sd": t.user_attrs.get(f"{pair_id}_sd", 0.0),
                        "deviation":     _dev(full_params),
                        "sim_route":     route,
                        **full_params,
                        **_imp_cols(s2_imp),
                    }
                break  # rank-1 only
            if s2_row is None and cp:
                s2_row = {
                    **base,
                    "stage":         "S2_pair",
                    "similarity":    cp["similarity"],
                    "similarity_sd": cp["similarity_sd"],
                    "deviation":     _dev(cp["params"]),
                    "sim_route":     cp["route"],
                    **cp["params"],
                    **_imp_cols(s1_pair_imp_dict),
                }
            if s2_row:
                rows.append(s2_row)
        elif cp:
            rows.append({
                **base,
                "stage":         "S2_pair",
                "similarity":    cp["similarity"],
                "similarity_sd": cp["similarity_sd"],
                "deviation":     _dev(cp["params"]),
                "sim_route":     cp["route"],
                **cp["params"],
                **_imp_cols(s1_pair_imp_dict),
            })

    df = pd.DataFrame(rows)
    interleaved = [c for p in param_cols for c in (p, f"imp_{p}")]
    col_order = ["subjID", "env", "pair", "stage", "similarity", "similarity_sd",
                 "deviation", "sim_route"] + interleaved
    df = df[[c for c in col_order if c in df.columns]]
    df["stage"] = pd.Categorical(
        df["stage"], categories=["group", "S1_pair", "S2_pair"], ordered=True
    )
    return s1_study, df

# ---------------------------------------------------------------------------
# Entry point (SLURM array)
# ---------------------------------------------------------------------------
# ---- Local test toggle ----
LOCAL_TEST  = True        # set True to run locally without SLURM
TEST_SUBJ   = "11002"     # subjID to use when LOCAL_TEST = True
TEST_ENV    = "dsp1"      # env prefix to use when LOCAL_TEST = True
TEST_TRIALS = 300         # trial budget for local runs
# ---------------------------

if __name__ == "__main__":
    allTrajGrid = pd.read_pickle(DATA_DIR / "subjTrajNodes.pkl").reset_index(drop=True)
    subj_ids = sorted(allTrajGrid["SubjectNum"].unique())
    envs     = sorted(allTrajGrid["pairs"].str.extract(r"^(dsp\d+)")[0].dropna().unique())

    if LOCAL_TEST:
        subj_id, env_prefix = TEST_SUBJ, TEST_ENV
        N_TRIALS  = TEST_TRIALS
        PROCESSES = max(1, cpu_count())
        job_idx   = None
        print(f"\n[LOCAL TEST] subj={subj_id}  env={env_prefix}  n_trials={N_TRIALS}", flush=True)
    else:
        jobs = [(s, e) for s in subj_ids for e in envs]
        job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        subj_id, env_prefix = jobs[job_idx]
        N_TRIALS  = 500
        PROCESSES = max(1, cpu_count())
        print(
            f"\n[{job_idx + 1}/{len(jobs)}] subj={subj_id}  env={env_prefix}  "
            f"n_trials={N_TRIALS}  processes={PROCESSES}",
            flush=True,
        )

    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    study, results = fit_participant(
        subj_id=subj_id,
        env_prefix=env_prefix,
        n_trials=N_TRIALS,
        processes=PROCESSES,
        patience=50
    )

    if results is not None:
        out_path = DUMP_DIR / f"paramFit_{subj_id}_{env_prefix}.pkl"
        results.to_pickle(out_path)
        n_pairs = results["pair"].nunique()
        print(f"  Saved params for {n_pairs} pairs → {out_path}", flush=True)
        best = results[results["stage"] == "S1_pair"].nlargest(1, "similarity").iloc[0]
        print(f"  Best S1_pair similarity : {best['similarity']:.4f}  (pair={best['pair']})", flush=True)
        param_cols = [c for c in results.columns if c in PARAM_SPACE]
        print(f"  Its params:\n{best[param_cols].to_string()}", flush=True)
