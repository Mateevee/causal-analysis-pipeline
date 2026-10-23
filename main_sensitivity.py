"""
main_sensitivity.py -- sensitivity of the pipeline to its own cut-offs: the
five regime cut-offs (one decision per dataset) and the four Stage 7 cut-offs
(one decision per candidate pair). The protocol -- grids, joint draw,
robustness criterion -- is Config/sensitivity.py and is locked when a run
starts; the report is produced by generators/sensitivity_analysis.py (paper
repository) from <OUTPUT_DIR>. Every run uses the pipeline's own random seeds
(literals in its code): one seed, the one of the reported run.

What one run does (always all of it, in this order)
---
  control  the default configuration on every scenario. Its diagnostics give
           the margin table (distance of each dataset to each cut-off), and
           its output must reproduce the reported run.
  oat      every grid value of every cut-off, one cut-off at a time.
  joint    the Latin-hypercube configurations (all cut-offs at once).

How a configuration is run
---
Each configuration is a real call of the reference model's run_pipeline with
the cut-offs replaced. Cut-offs that are run_pipeline arguments go in as
arguments; the others are module-level values of the pipeline, set for the
duration of the call (_Hooks) -- the pipeline's code is not edited.

The expensive stage functions are memoized on disk (<OUTPUT_DIR>/cache), each
keyed by the scenario and by what the function's output depends on: the
branch taken (time axis yes/no for Stages 3-5; control set for Stage 6; gate
open/closed for Stage 8b) or the content of its input table (Stages 8a and
8a-bis). A configuration therefore recomputes a stage only when that stage's
input actually differs from one already computed, and the cut-off itself is
always compared by the pipeline's own code. Stages 8a and 8a-bis, which work
one target at a time, are also memoized per target.

Usage
---
No command-line arguments, by design -- edit the constants in the
"EXECUTION CONTROL" section below and run:

    python main_sensitivity.py

An interrupted run is continued by setting RESUME_DIR to its folder: a
(scenario, configuration) already in <OUTPUT_DIR>/runs is skipped.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import datetime
import gzip
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd

import main_heldout
from Config import execution as cfg_exec
from Config import sensitivity as proto
from Auxiliary.console_progress import PinnedProgress
from Data import (
    generate_tigramite_benchmark_scenarios,
    generate_synthetic_var_benchmark_scenarios,
    generate_lorenz96_benchmark_scenarios,
    generate_logistic_maps_benchmark_scenarios,
    generate_causaltime_benchmark_scenarios,
    load_alarm_benchmark,
    generate_netsim_benchmark_scenarios,
    load_with_fallback,
)
from Pipeline import (pivot, property_scanner, preprocessing, discovery,
                      dml_engine, predictive, coherence, SelectorFunc)
from Pipeline import contemporaneous as contemp
from Pipeline import pipeline as pipeline_mod
from Pipeline import structural as structural_mod

# =============================================================================
# EXECUTION CONTROL -- edit here.
# =============================================================================

OUTPUT_DIR = f"./Results_Sensitivity/Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Folder of an interrupted run to continue (None = start a new run in
# OUTPUT_DIR). The protocol and the pipeline's code must be the ones the run
# started with.
RESUME_DIR = None

# Substrings of scenario names; empty = every scenario. For testing only: the
# report needs every scenario.
SCENARIO_FILTER = []

BASE = next(m for m in cfg_exec.ALL_MODELS if m['name'] == proto.REFERENCE_MODEL)
REPO = os.path.dirname(os.path.abspath(__file__))
OUT = sys.__stdout__


def say(msg: str) -> None:
    print(msg, file=OUT, flush=True)


# =============================================================================
# PROGRESS
# =============================================================================

STEP_LABEL = {'control': 'control (default)', 'oat': 'one cut-off at a time',
              'joint': 'joint variation'}
STAGE_LABEL = {
    'pivot': 'Stage 0 (structuring)', 'scan': 'Stage 1 (active properties)',
    'prep': 'Stage 2 (pre-processing)', 'prefilter': 'Stages 3-5 (pre-filter)',
    'dml': 'Stage 6 (DML)', 'stage8a': 'Stage 8a (structural pruning)',
    'stage8a_target': 'Stage 8a (structural pruning)',
    'stage8abis_delta': 'Stage 8a-bis (predictive competition)',
    'stage8abis': 'Stage 8a-bis (predictive competition)',
    'stage8abis_target': 'Stage 8a-bis (predictive competition)',
    'past_r2_all': 'Stage 8 guards (past R2)', 'past_r2_own': 'Stage 8 guards (past R2)',
    'stage8b_links': 'Stage 8b (lag-0 skeleton)', 'stage8b_dml': 'Stage 8b (lag-0 DML)',
}
IDLE_STAGE = "stage: Stage 7 and cached stages"


def _hms(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}" if m >= 60 else f"{m}min{int(seconds % 60):02d}s"


class Progress:
    """Three-line footer pinned to the bottom of the terminal: where the run
    is overall, which configuration of which scenario is running, and which
    stage is being computed. Without a terminal (output redirected) the
    footer is off and a line is printed per configuration that had to
    compute something."""

    def __init__(self, n_scenarios: int, n_total: int, n_done: int):
        self.n_scenarios, self.n_total = n_scenarios, n_total
        self.n_done = self.n_done_at_start = n_done
        self.t0 = time.perf_counter()
        self.lines = ["", "", ""]
        with contextlib.redirect_stdout(OUT):
            self.footer = PinnedProgress(n_lines=3)
            self.footer.start()
        self.tty = self.footer._active

    def _draw(self) -> None:
        with contextlib.redirect_stdout(OUT):   # the pipeline's stdout is captured
            self.footer.update(self.lines)

    def configuration(self, i_scen: int, name: str, config: dict, i_cfg: int, n_cfg: int,
                      step_pos: str, t_scen: float) -> None:
        elapsed = time.perf_counter() - self.t0
        ran = self.n_done - self.n_done_at_start
        left = self.n_total - self.n_done
        eta = (f"~{_hms(elapsed / ran * left)} left at the pace of this session"
               if ran >= 20 else "estimating time left")
        self.lines = [
            f"[{self.n_done / max(self.n_total, 1):6.1%}] run {self.n_done + 1} of {self.n_total} | "
            f"scenario {i_scen} of {self.n_scenarios} | elapsed {_hms(elapsed)} | {eta}",
            f"{name} | {STEP_LABEL[config['phase']]} {step_pos} | configuration {i_cfg} of "
            f"{n_cfg} ({config['id']}) | scenario time {_hms(time.perf_counter() - t_scen)}",
            IDLE_STAGE,
        ]
        self._draw()

    def stage(self, name: str, detail: str = "") -> None:
        self.lines[2] = f"stage: computing {STAGE_LABEL.get(name, name)}{detail}"
        self._draw()

    def stage_done(self) -> None:
        self.lines[2] = IDLE_STAGE
        self._draw()

    def finished(self, name: str, record: dict) -> None:
        self.n_done += 1
        if not self.tty and (record['computed'] or record['seconds'] >= 5):
            stages = sorted({STAGE_LABEL.get(c, c) for c in record['computed']})
            say(f"    [{self.n_done / max(self.n_total, 1):6.1%}] {name} | {record['config']}: "
                f"{record['seconds']:.0f}s, computed {', '.join(stages) or 'nothing'}")

    def stop(self) -> None:
        with contextlib.redirect_stdout(OUT):
            self.footer.stop()


# =============================================================================
# SCENARIOS
# =============================================================================

def load_scenarios() -> list:
    """The scenarios of main.py (same arguments as its entry point) and of
    main_heldout.py."""
    dfs: list = []
    for (name_prefix, seed, n_points, n_entities, prob_edge,
         max_parents, max_lag, min_edges) in cfg_exec.SYNTHETIC_VAR_SCENARIOS:
        dfs += load_with_fallback(
            name_prefix, generate_synthetic_var_benchmark_scenarios,
            seeds=[seed], n_points=n_points, name_prefix=name_prefix,
            n_entities=n_entities, prob_edge=prob_edge, max_parents=max_parents,
            max_lag=max_lag, min_edges=min_edges)
    dfs += load_with_fallback("tigramite", generate_tigramite_benchmark_scenarios,
                              seeds=[11, 22, 33], n_points=300)
    dfs += load_with_fallback("lorenz96", generate_lorenz96_benchmark_scenarios,
                              Ks=[8, 10], n_points=300, seeds=[1, 2, 3])
    dfs += load_with_fallback("logistic_maps", generate_logistic_maps_benchmark_scenarios,
                              n_points=300, seeds=[1, 2, 3])
    dfs += load_with_fallback("causaltime", generate_causaltime_benchmark_scenarios,
                              domains=["medical", "pm25", "traffic"],
                              base_dir=cfg_exec.CAUSALTIME_DIR_PATH,
                              mode="separate_samples", max_samples=3)
    dfs += load_with_fallback("alarm", load_alarm_benchmark, n_samples=500)
    dfs += load_with_fallback("netsim", generate_netsim_benchmark_scenarios,
                              subjects=cfg_exec.NETSIM_SUBJECTS)
    dfs += main_heldout._load_scenarios()[0]
    seen, out = set(), []
    for d in dfs:
        if d['name'] not in seen:
            seen.add(d['name'])
            out.append(d)
    return out


# =============================================================================
# PROTOCOL AND CONFIGURATIONS
# =============================================================================

def code_fingerprint() -> str:
    """Hash of the pipeline's code and of its run configuration (line endings
    normalized): what must stay the same for the whole analysis."""
    h = hashlib.sha1()
    files = [os.path.join("Config", "execution.py")]
    for folder in ("Pipeline", "Data"):
        files += [os.path.join(folder, f) for f in sorted(os.listdir(os.path.join(REPO, folder)))
                  if f.endswith(".py")]
    for rel in files:
        with open(os.path.join(REPO, rel), "rb") as fh:
            h.update(rel.replace("\\", "/").encode())
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def code_default(th: dict):
    """The pipeline's default value for this cut-off."""
    if th['kind'] == 'config':
        override = BASE['config_override']
        if th['target'] in override:
            return override[th['target']]
        return inspect.signature(pipeline_mod.run_pipeline).parameters[th['target']].default
    if th['kind'] == 'time_axis':
        return inspect.signature(discovery._detect_time_axis).parameters[th['target']].default
    if th['kind'] == 'penalty':
        return getattr(BASE['config_override']['causal_penalty_fn'], th['target'])
    return getattr(SelectorFunc, th['target'])


def check_defaults() -> None:
    for th in proto.THRESHOLDS:
        if float(code_default(th)) != float(th['default']):
            raise SystemExit(f"[main_sensitivity] '{th['key']}': protocol default {th['default']} "
                             f"!= value in the code {code_default(th)}")
        if th['default'] not in th['grid']:
            raise SystemExit(f"[main_sensitivity] '{th['key']}': default not in its grid")


def defaults() -> dict:
    return {th['key']: float(th['default']) for th in proto.THRESHOLDS}


def build_configs() -> list:
    """[{'id', 'phase', 'values': {cut-off key: value}}]; 'default' first."""
    configs = [dict(id='default', phase='control', values=defaults())]
    for th in proto.THRESHOLDS:
        for v in th['grid']:
            if float(v) != float(th['default']):
                configs.append(dict(id=f"oat__{th['key']}__{v:g}", phase='oat',
                                    values={**defaults(), th['key']: float(v)}))
    rng = np.random.default_rng(proto.JOINT['seed'])
    n = proto.JOINT['n']
    columns = {}
    for th in proto.THRESHOLDS:          # Latin hypercube: one draw per stratum
        lo, hi = min(th['grid']), max(th['grid'])
        u = (rng.permutation(n) + rng.random(n)) / n
        columns[th['key']] = np.round(lo + u * (hi - lo), 4)
    for i in range(n):
        configs.append(dict(id=f"joint__{i:03d}", phase='joint',
                            values={k: float(col[i]) for k, col in columns.items()}))
    return configs


def lock_protocol(output_dir: str, configs: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    current = dict(protocol=json.loads(json.dumps(proto.protocol())),
                   code_fingerprint=code_fingerprint())
    path = os.path.join(output_dir, "protocol.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            stored = json.load(fh)
        for key, what in (("protocol", "protocol (Config/sensitivity.py)"),
                          ("code_fingerprint", "pipeline code (Pipeline/, Data/, Config/execution.py)")):
            if stored[key] != current[key]:
                raise SystemExit(f"[main_sensitivity] the {what} changed since this run "
                                 f"started; start a new OUTPUT_DIR.")
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({**current, "git_commit": git_commit(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                  fh, ensure_ascii=False, indent=2)
    keys = [th['key'] for th in proto.THRESHOLDS]
    with open(os.path.join(output_dir, "configs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "phase"] + keys)
        for c in configs:
            w.writerow([c['id'], c['phase']] + [c['values'][k] for k in keys])


# =============================================================================
# MEMOIZATION OF THE STAGE FUNCTIONS
# =============================================================================

def _sig(value) -> str:
    """Address-free text signature of an argument value."""
    if isinstance(value, (type(None), bool, int, float, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_sig(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_sig(v)}" for k, v in sorted(value.items())) + "}"
    return type(value).__name__ + _sig(getattr(value, "__dict__", {}))


def _kw_sig(kwargs: dict, drop=()) -> str:
    return _sig({k: v for k, v in kwargs.items() if k not in ('n_jobs', 'verbose') + tuple(drop)})


def frame_key(df: pd.DataFrame) -> str:
    """Content hash of a results table (object cells by their repr)."""
    h = hashlib.sha1()
    h.update(repr(list(df.index)).encode())
    for c in df.columns:
        col = df[c]
        if col.dtype == object:
            col = col.map(repr)
        h.update(str(c).encode())
        h.update(pd.util.hash_pandas_object(col, index=False).values.tobytes())
    return h.hexdigest()


class Memo:
    """Disk + memory cache of stage outputs for the scenario being run."""

    def __init__(self, root: str):
        self.root, self.scenario, self.mem, self.computed = root, None, {}, []
        self.progress = None

    def start(self, scenario: str) -> None:
        self.scenario, self.mem = scenario, {}
        os.makedirs(os.path.join(self.root, scenario), exist_ok=True)

    def _locate(self, name: str, key: str) -> tuple:
        digest = hashlib.sha1(f"{name}|{key}".encode()).hexdigest()[:20]
        return digest, os.path.join(self.root, self.scenario, f"{name}_{digest}.pkl")

    def has(self, name: str, key: str) -> bool:
        digest, path = self._locate(name, key)
        return digest in self.mem or os.path.exists(path)

    def get(self, name: str, key: str, compute):
        digest, path = self._locate(name, key)
        if digest not in self.mem:
            if not os.path.exists(path):
                self.computed.append(name)
                if self.progress and not name.endswith('_target'):
                    self.progress.stage(name)
                joblib.dump(compute(), path + ".tmp", compress=3)
                os.replace(path + ".tmp", path)
                if self.progress and not name.endswith('_target'):
                    self.progress.stage_done()
            # always the stored copy, so a fresh result and a cached one are
            # the same object downstream
            self.mem[digest] = joblib.load(path)
        return copy.deepcopy(self.mem[digest]), digest


def memo_parallel(memo: Memo, name: str, parallel, task_key):
    """Stand-in for joblib.Parallel in a stage that dispatches one task per
    target: a task already computed for this scenario (same target, same
    input, same parameters -- `task_key`) is read from the cache and only the
    others are dispatched."""

    class MemoParallel:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs

        def __call__(self, calls):
            calls = list(calls)                      # (function, args, kwargs)
            keys = [task_key(args[0]) for _, args, _ in calls]
            missing = [i for i, k in enumerate(keys) if not memo.has(name, k)]
            if missing and memo.progress:
                memo.progress.stage(name, f" -- {len(missing)} of {len(keys)} targets "
                                          f"(the others are cached)")
            if len(missing) > 1:
                fresh = parallel(*self.args, **self.kwargs)(calls[i] for i in missing)
            else:
                fresh = [calls[i][0](*calls[i][1], **calls[i][2]) for i in missing]
            computed = dict(zip(missing, fresh))
            return [memo.get(name, k, lambda i=i: computed[i])[0] for i, k in enumerate(keys)]

    return MemoParallel


def time_axis_statistic(processed: dict) -> tuple:
    """(median |ACF(1)| * sqrt(median n), median n): the series have a time
    axis when the first is at least the band constant. Mirrors
    discovery._detect_time_axis, whose decision it is checked against."""
    acf, sizes = [], []
    for series in processed.values():
        v = np.asarray(series, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) < 30:
            continue
        v = v - v.mean()
        den = float(np.dot(v, v))
        if den <= 0:
            continue
        acf.append(abs(float(np.dot(v[:-1], v[1:]) / den)))
        sizes.append(len(v))
    if not acf:
        return None, None
    n = int(np.median(sizes))
    return float(np.median(acf)) * float(np.sqrt(max(n, 1))), n


class _Hooks:
    """Sets the cut-offs that are not run_pipeline arguments and memoizes the
    stage functions, for one pipeline call; restores everything on exit.
    `trace` receives the diagnostics each cut-off is compared with."""

    def __init__(self, memo: Memo, values: dict, trace: dict):
        self.memo, self.values, self.trace, self._saved = memo, values, trace, []

    def _set(self, module, name, new):
        self._saved.append((module, name, getattr(module, name)))
        setattr(module, name, new)

    def __enter__(self):
        memo, trace, values = self.memo, self.trace, self.values
        band = values['time_axis_band']

        for th in proto.THRESHOLDS:
            if th['kind'] == 'selector':
                self._set(SelectorFunc, th['target'], values[th['key']])

        # Stages 0-2: no cut-off upstream of them.
        for module, name, tag in ((pivot, 'load_and_pivot', 'pivot'),
                                  (property_scanner, 'scan_entity_properties', 'scan'),
                                  (preprocessing, 'preprocess_all', 'prep')):
            orig = getattr(module, name)
            self._set(module, name, lambda *a, _o=orig, _t=tag, **k:
                      memo.get(_t, '', lambda: _o(*a, **k))[0])

        detect = discovery._detect_time_axis

        def has_time_axis(processed, *_a, **_k):
            decision = detect(processed, alpha_white_noise=band, verbose=False)
            stat, n = time_axis_statistic(processed)
            if stat is not None and (stat >= band) != decision:
                raise RuntimeError("time_axis_statistic disagrees with _detect_time_axis")
            trace.update(time_axis_stat=stat, time_axis_n=n, time_axis=bool(decision))
            return decision
        self._set(discovery, '_detect_time_axis', has_time_axis)
        self._set(contemp, '_detect_time_axis', has_time_axis)

        prefilter = discovery.filter_causal_candidates

        def filter_causal_candidates(processed, pairs, **kw):
            mode = kw.get('correct_lag_multiplicity_rescue')
            branch = has_time_axis(processed) if mode == 'auto' else bool(mode)
            out, trace['key_prefilter'] = memo.get(
                'prefilter', f"{branch}|{_kw_sig(kw)}",
                lambda: prefilter(processed, pairs, **kw))
            return out
        self._set(discovery, 'filter_causal_candidates', filter_causal_candidates)

        estimate = dml_engine.estimate_all_effects

        def estimate_all_effects(candidates, processed, all_keys, **kw):
            out, _ = memo.get('dml', f"{trace['key_prefilter']}|{_kw_sig(kw)}",
                              lambda: estimate(candidates, processed, all_keys, **kw))
            if kw.get('control_set') == 'parents':
                trace['outcome_r2'] = dml_engine.median_outcome_r2(out)
            trace['control_set'] = kw.get('control_set')
            return out
        self._set(dml_engine, 'estimate_all_effects', estimate_all_effects)

        assess = coherence.assess_evidence_coherence

        def assess_evidence_coherence(results, **kw):
            out = assess(results, **kw)
            trace.update(rho=out['rho'], rho_lower=out['rho_lower'], rho_upper=out['rho_upper'],
                         rho_n=out['n'], evidence_regime=out['regime'],
                         evidence_reason=out['reason'])
            return out
        self._set(coherence, 'assess_evidence_coherence', assess_evidence_coherence)

        structural = pipeline_mod.apply_structural_filter

        def apply_structural_filter(results, processed, **kw):
            return memo.get('stage8a', f"{frame_key(results)}|{_kw_sig(kw)}",
                            lambda: structural(results, processed, **kw))[0]
        self._set(pipeline_mod, 'apply_structural_filter', apply_structural_filter)

        # Inside Stages 8a and 8a-bis the work is one task per target, and a
        # change in the Stage 7 cut-offs alters the input of few targets:
        # the per-target tasks are memoized too. The series (`processed`,
        # and the arrays taken from it) are fixed for the scenario and stay
        # out of the key.
        self._set(structural_mod, 'Parallel', memo_parallel(
            memo, 'stage8a_target', structural_mod.Parallel,
            lambda a: f"{a[0]}|{a[1]}|{frame_key(a[2])}|{_sig(a[4:])}"))
        self._set(predictive, 'Parallel', memo_parallel(
            memo, 'stage8abis_target', predictive.Parallel,
            lambda a: f"{a[0]}|{_sig(a[2])}|{_sig(a[4:])}"))

        compete = predictive.predictive_competition

        def predictive_competition(results, processed, **kw):
            # The cut-off enters only through `delta BIC > min_delta_bic`
            # (predictive._two_populations), and delta BIC does not depend
            # on it: the output is keyed by the side of the cut-off.
            key = f"{frame_key(results)}|{_kw_sig(kw, drop=('min_delta_bic',))}"
            delta, _ = memo.get('stage8abis_delta', key,
                                lambda: compete(results, processed, **kw)[1]['delta_bic'])
            above = bool(np.isfinite(delta) and delta > kw['min_delta_bic'])
            (res, info), _ = memo.get('stage8abis', f"{key}|{above}",
                                      lambda: compete(results, processed, **kw))
            trace.update(delta_bic=info['delta_bic'], predictive_applied=bool(info['applied']),
                         predictive_n_significant=info['n_significant'])
            return res, info
        self._set(predictive, 'predictive_competition', predictive_competition)

        for name, tag in (('past_explained_variance', 'past_r2_all'),
                          ('own_past_explained_variance', 'past_r2_own')):
            orig = getattr(contemp, name)

            def past(processed, keys, max_lag, _o=orig, _t=tag):
                out, _ = memo.get(_t, str(max_lag), lambda: _o(processed, keys, max_lag))
                trace[_t] = out
                return out
            self._set(contemp, name, past)

        discover = contemp.discover_contemporaneous

        def discover_contemporaneous(processed, keys, **kw):
            # The gate statistic is past_explained_variance (same computation).
            cut = kw.get('max_past_r2')
            r2 = contemp.past_explained_variance(processed, keys, kw['max_lag'])
            closed = cut is not None and r2 is not None and r2 > cut
            links, _ = memo.get('stage8b_links', f"{closed}|{_kw_sig(kw, drop=('max_past_r2',))}",
                                lambda: discover(processed, keys, **kw))
            if closed and links:
                raise RuntimeError("Stage 8b gate: closed by past R2 but links were returned")
            trace['stage8b_gate'] = 'closed' if closed else 'open'
            return links
        self._set(contemp, 'discover_contemporaneous', discover_contemporaneous)

        lag0 = dml_engine.estimate_lag0_effects

        def estimate_lag0_effects(links, processed, all_keys, **kw):
            return memo.get('stage8b_dml', f"{_sig(links)}|{_kw_sig(kw)}",
                            lambda: lag0(links, processed, all_keys, **kw))[0]
        self._set(dml_engine, 'estimate_lag0_effects', estimate_lag0_effects)
        return self

    def __exit__(self, *exc):
        for module, name, old in reversed(self._saved):
            setattr(module, name, old)
        return False


# =============================================================================
# ONE RUN
# =============================================================================

def _plain(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    return v


def run_one(entry: dict, config: dict, memo: Memo, log_path: str = None) -> dict:
    values = config['values']
    penalty = copy.copy(BASE['config_override']['causal_penalty_fn'])
    cfg = {**cfg_exec.PIPELINE_CONFIG, **BASE['config_override'], 'causal_penalty_fn': penalty}
    for th in proto.THRESHOLDS:
        if th['kind'] == 'config':
            cfg[th['target']] = values[th['key']]
        elif th['kind'] == 'penalty':
            setattr(penalty, th['target'], values[th['key']])

    trace, buffer, start = {}, io.StringIO(), time.perf_counter()
    memo.computed = []
    record = dict(config=config['id'], status='ok')
    try:
        with _Hooks(memo, values, trace), contextlib.redirect_stdout(buffer):
            results = BASE['pipeline'](entry['df'], **cfg)
        pairs = []
        if results is not None and 'significant' in results.columns:
            sig = results[results['significant'] == True]  # noqa: E712
            pairs = sorted({(str(r.source_entity).strip(), str(r.target_entity).strip(),
                             int(r.lag) if pd.notna(r.lag) else -1)
                            for r in sig.itertuples()})
            for column in ('evidence_fallback', 'evidence_past_r2'):
                if column in results.columns and len(results):
                    trace[column] = results[column].iloc[0]
        record['pairs'] = pairs
        trace.update(support_floor=penalty.calibrated_threshold_,
                     support_floor_n=penalty.n_calibration_candidates_)
    except Exception as exc:
        record.update(status=f"error: {type(exc).__name__}: {exc}", pairs=[])
    trace.pop('key_prefilter', None)
    record.update(trace={k: _plain(v) for k, v in trace.items()},
                  computed=memo.computed, seconds=round(time.perf_counter() - start, 1))
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with gzip.open(log_path, "wt", encoding="utf-8") as fh:
            fh.write(buffer.getvalue())
    return record


# =============================================================================
# RUN
# =============================================================================

def run_configs(phases: tuple, output_dir: str) -> None:
    configs = build_configs()
    lock_protocol(output_dir, configs)
    todo = [c for c in configs if c['phase'] in phases]
    scenarios = [s for s in load_scenarios()
                 if not SCENARIO_FILTER or any(f in s['name'] for f in SCENARIO_FILTER)]
    for folder in ("runs", "gt"):
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # What is already done (a resumed run): {scenario: {configuration id}}.
    done = {}
    for entry in scenarios:
        runs_path = os.path.join(output_dir, "runs", f"{entry['name']}.jsonl")
        done[entry['name']] = set()
        if os.path.exists(runs_path):
            with open(runs_path, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r['status'] == 'ok':
                        done[entry['name']].add(r['config'])
    ids = {c['id'] for c in todo}
    n_total = len(scenarios) * len(todo)
    n_done = sum(len(d & ids) for d in done.values())
    say(f"\n[main_sensitivity] {' + '.join(STEP_LABEL[p] for p in phases)}: "
        f"{len(scenarios)} scenario(s) x {len(todo)} configuration(s) = "
        f"{n_total} runs, {n_done} already done -> {output_dir}")

    memo = Memo(os.path.join(output_dir, "cache"))
    progress = memo.progress = Progress(len(scenarios), n_total, n_done)
    step_size = {p: sum(c['phase'] == p for c in todo) for p in phases}
    try:
        for i, entry in enumerate(scenarios, 1):
            name = entry['name']
            if entry.get('gt') is not None:
                entry['gt'].to_csv(os.path.join(output_dir, "gt", f"{name}.csv"), index=False)
            runs_path = os.path.join(output_dir, "runs", f"{name}.jsonl")
            if ids <= done[name]:
                continue
            memo.start(name)
            t_scen, n_ran, n_err = time.perf_counter(), 0, 0
            step_pos = {p: 0 for p in phases}
            for j, config in enumerate(todo, 1):
                step_pos[config['phase']] += 1
                if config['id'] in done[name]:
                    continue
                progress.configuration(
                    i, name, config, j, len(todo),
                    f"{step_pos[config['phase']]} of {step_size[config['phase']]}", t_scen)
                log_path = None
                if config['id'] == 'default':
                    log_path = os.path.join(output_dir, "logs", f"{name}.log.gz")
                record = run_one(entry, config, memo, log_path)
                n_ran += 1
                n_err += record['status'] != 'ok'
                with open(runs_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.finished(name, record)
                if record['status'] != 'ok':
                    say(f"    {config['id']}: {record['status']}")
            say(f"  [{i}/{len(scenarios)}] {name}: {n_ran} configuration(s) in "
                f"{_hms(time.perf_counter() - t_scen)}, {n_err} error(s); "
                f"elapsed {_hms(time.perf_counter() - progress.t0)}")
    finally:
        progress.stop()


def main():
    check_defaults()
    output_dir = os.path.abspath(RESUME_DIR or OUTPUT_DIR)
    total_start_time = time.perf_counter()
    run_configs(('control', 'oat', 'joint'), output_dir)
    say(f"\n[main_sensitivity] done in {(time.perf_counter() - total_start_time) / 3600:.2f} h "
        f"-> {output_dir}")


if __name__ == '__main__':
    main()
