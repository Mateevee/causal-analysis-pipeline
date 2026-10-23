"""
pipeline_cuts_plus.py — Wrapper for the OFFICIAL CUTS+ neural architecture
================================================================================
Exposes run_pipeline_cuts_plus with the SAME call signature used by the
runner. Reuses from causal_common: _ngc_build_lagged, _ngc_ftest_pair and
_split_sample.

NOTE: this pipeline does NOT use _pcmci_load_and_pivot — it has its own
variant (_cuts_plus_load_and_pivot_with_mask, defined below) which also
returns the mask of originally observed values (needed for the mask-aware
supervision of the fine phase). If the pivoting logic in
_pcmci_load_and_pivot changes in the future, replicate the change here too.

Uses the OFFICIAL CUTS+ neural architecture (Cheng, Li, Xiao, Li, Suo, He
& Dai, "CUTS+: High-Dimensional Causal Discovery from Irregular
Time-Series", AAAI 2024), vendored from the authors' own repository,
https://github.com/jarrycyx/UNN (folder CUTS_Plus), MIT-licensed:

  - `GRUCell`, `MPNN`, `LocalConv1D`, `CUTS_Plus_Net` (classes below) are a
    line-for-line port of CUTS_Plus/model/cuts_plus_net.py, with the
    original's `einops.rearrange` calls replaced by equivalent native
    torch .permute()/.reshape() calls (to avoid adding `einops` as a new
    project dependency) — no other change to the architecture.
  - `_CutsPlusOfficialTrainer` (below) ports the alternating "data
    prediction" / "graph discovery" training loop of CUTS_Plus/cuts_plus.py
    (the `MultiCAD` class): differentiable graph search via a
    Gumbel-sigmoid-sampled adjacency (`torch.nn.functional.gumbel_softmax`,
    which is itself where the original's vendored `utils/gumbel_softmax.py`
    copies its implementation FROM — so this project calls PyTorch's own
    copy directly instead of re-vendoring it), an L1 sparsity penalty on the
    learned graph, the coarse-to-fine NODE-GROUPING schedule
    (`_maybe_update_grouping`, with the reference config's
    n_groups=32 / multiply_2_every_20 when that is an actual coarse
    grouping, N >= 64; below that the official code's
    `n_groups == n_nodes` branch is used, i.e. no grouping — see
    cuts_n_groups_start), and the
    "rate"-style progressive self-training fill policy for the
    originally-missing points. Per-run TensorBoard/figure logging and the
    original's own true-graph diagnostics (`utils/logger.py`,
    `utils/misc.py`) are not ported, since this project performs its own
    validation downstream (Validation/).

    MIT License — Copyright (c) jarrycyx / UNN contributors.
    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files, to deal in
    the software without restriction, subject to including this notice —
    see https://github.com/jarrycyx/UNN for the full license text.

Hyperparameter defaults below (hidden_ch=32, gru_layers=1, lr schedules,
lambda_s/tau annealing, causal_thres=0.5, the "rate_0.1_after_20/64" fill
policy) mirror the authors' own published example configuration for
Lorenz-96 (CUTS_Plus/opt/lorenz_example.yaml) unchanged, except where a
scenario-size-dependent quantity (total_epoch, batch_size, the fill
policy's "after" epoch) is re-expressed as a fraction of that reference
config so it scales sensibly across this paper's much wider range of T and
N than the authors' own single reference scenario (the domain-agnostic,
untuned-per-scenario convention already used everywhere else in this
pipeline).

CUTS+ has no native p-value. As for the other baselines in this runner
without one (VARLiNGAM, DYNOTEARS), a pair i -> j is "discovered" when the
learned edge score (G @ GT)[i, j] -- the RAW GT, as in the official code,
see "Operating point" below -- clears cuts_causal_threshold,
and its p-value comes from an OLS F-test of the tau_max lags of X_i
conditioned on the other discovered parents of j (_ngc_ftest_pair) — the
same post-selection inference already used elsewhere in this runner.
'significant' = discovered AND pvalue < alpha_discovery (after
cuts_correction; cuts_ftest_gate=False uses the native thresholded graph
alone -- see _posthoc_gate).

Operating point: the paper (Cheng et al., AAAI 2024) fixes no
binarization rule (an edge is dropped when its causal
probability m_ij is "penalized to zero (or below some certain
threshold)", value unspecified). The official code's only cut is in
`MultiCAD.train` -> `utils/misc.py::calc_and_log_metrics(Graph, ...,
threshold=0.5)`, where `Graph = G @ GT` uses the RAW GT (no sigmoid) — the
yaml's `causal_thres` key is never read. So the official rule is
GT > 0.5, reproduced here (cuts_causal_threshold, default 0.5, applied to
the raw G @ GT). In the paper's terms (CPG M = G @ sigmoid(Theta)) this is
m_ij > sigmoid(0.5) ~= 0.622, which is the probability every edge is
INITIALIZED at (the official GT init is the raw value 0.5): an edge is
kept when training raised its causal probability above its starting
value, and dropped when the sparsity penalty lowered it. Since G is a
hard one-hot group assignment, G @ GT is GT broadcast to full resolution.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as _F
from torch import nn
from typing import Optional, List, Tuple

from Auxiliary.causal_common import (
    _ngc_build_lagged,
    _ngc_ftest_pair,
    _split_sample,
    _posthoc_gate,
)


def _cuts_plus_load_and_pivot_with_mask(
    df: pd.DataFrame,
    date_col: str,
    entity_col: str,
    property_cols: Optional[List[str]],
) -> tuple:
    """
    Variant of _pcmci_load_and_pivot that also returns the mask of
    originally missing values (before any interpolation), needed for the
    mask-aware supervision of the official CUTS+ trainer (see
    run_pipeline_cuts_plus). Reimplements the pivoting so as not to
    alter the contract of _pcmci_load_and_pivot used by the other
    pipelines (PCMCI, DYNOTEARS) — if the pivoting logic changes in the
    future, replicate the change here too.

    Returns (pivoted_filled, observed_mask):
        pivoted_filled : wide DataFrame, gaps filled by zero-order hold (same
                          contract as _pcmci_load_and_pivot).
        observed_mask  : wide boolean DataFrame, same shape/columns as
                          pivoted_filled — True where the value is original
                          (not imputed), False where it was filled.
    """
    dfx = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(dfx[date_col]):
        dfx[date_col] = pd.to_datetime(dfx[date_col], dayfirst=True, errors='coerce')
    dfx = dfx.dropna(subset=[date_col]).sort_values([date_col, entity_col])
    if dfx.empty:
        raise ValueError(f"CUTS+: empty dataframe after converting '{date_col}' to datetime.")

    if property_cols is None:
        resolved_property_cols = [c for c in dfx.columns if c not in [date_col, entity_col]]
    else:
        resolved_property_cols = list(property_cols)
    if not resolved_property_cols:
        raise ValueError("CUTS+: resolved property_cols is empty.")

    for col in resolved_property_cols:
        if col not in dfx.columns:
            raise KeyError(f"CUTS+: property column '{col}' does not exist in the dataframe.")
        dfx[col] = pd.to_numeric(dfx[col], errors='coerce')

    pivoted_raw = dfx.pivot_table(
        index=date_col, columns=entity_col, values=resolved_property_cols, aggfunc='mean',
    )
    if pivoted_raw.shape[1] == 0:
        raise ValueError("CUTS+: pivot_table resulted in 0 columns (entity x property).")

    pivoted_raw.columns = [f"{w}__{f}" for f, w in pivoted_raw.columns]
    pivoted_raw = pivoted_raw.sort_index()
    pivoted_raw = pivoted_raw.dropna(axis=1, how='all')
    if pivoted_raw.shape[1] == 0:
        raise ValueError("CUTS+: all columns became empty after pivoting.")

    observed_mask = pivoted_raw.notna()   # True where there is real data (not imputed)

    # Zero-Order Hold (last observed value; leading gaps take the first
    # observation), the official pre-imputation (lorenz_example.yaml:
    # pre_impute.name = ZOH). Linear interpolation is NOT used: the CUTS
    # paper (App. A.3.2) shows it hampers the causal discovery, since an
    # interpolated point leaks the future sample into the predictor.
    pivoted_filled = pivoted_raw.ffill().bfill()
    pivoted_filled = pivoted_filled.dropna(axis=1, how='any')
    valid_cols = pivoted_filled.columns
    observed_mask = observed_mask[valid_cols]

    if pivoted_filled.shape[1] == 0:
        raise ValueError("CUTS+: all series became empty after interpolation.")

    return pivoted_filled, observed_mask


# =============================================================================
# Vendored (verbatim except einops -> native torch) from
# CUTS_Plus/model/cuts_plus_net.py — MIT License, jarrycyx/UNN. See file
# docstring above for provenance.
# =============================================================================

class GRUCell(nn.Module):

    def __init__(self, d_in, num_units, n_nodes, concat_h=False, activation='tanh'):
        super().__init__()
        self.activation_fn = getattr(torch, activation)
        mpnn_channel = d_in * n_nodes + num_units if concat_h else d_in * n_nodes
        self.forget_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.update_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.c_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)

    def forward(self, x, h, adj):
        r = torch.sigmoid(self.forget_gate(x, h, adj))
        u = torch.sigmoid(self.update_gate(x, h, adj))
        c = self.c_gate(x, r * h, adj)
        c = self.activation_fn(c)
        return u * h + (1. - u) * c


class MPNN(nn.Module):
    """graph[b, m, n] = edge weight from SOURCE node m to TARGET node n:
    x_messages[b,c,m,n] = x[b,c,m] * graph[b,m,n] is the message source m
    sends into target slot n; the 1x1 conv then mixes, per target node n,
    across the collapsed (channel, source) axis."""

    def __init__(self, c_in, c_out, concat_h=True):
        super().__init__()
        self.concat_h = concat_h
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x, h, graph):
        b, c, n = x.shape
        x_repeat = x[:, :, :, None].expand(-1, -1, -1, n)
        x_messages = torch.einsum('bcmn,bmn->bcmn', (x_repeat, graph))
        x_messages = x_messages.reshape(b, c * n, n)  # 'b c m n -> b (c m) n' (m and n both size n)
        if self.concat_h:
            out = self.mlp(torch.cat([x_messages, h], dim=1))
        else:
            out = self.mlp(x_messages)
        return out


class LocalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_nodes):
        super().__init__()
        self.out_channel = out_channels
        self.conv_list = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
            for _ in range(n_nodes)
        ])

    def forward(self, x):
        b, h, n = x.shape
        out = torch.zeros((b, self.out_channel, n)).to(x.device)
        for i in range(n):
            x_local_in = x[..., i].unsqueeze(-1)
            x_local_out = self.conv_list[i](x_local_in)
            out[..., i] = x_local_out.squeeze(-1)
        return out


class CUTS_Plus_Net(nn.Module):
    def __init__(self, n_nodes, in_ch=1, hidden_ch=32, n_layers=1,
                 shared_weights_decoder=False, concat_h=False):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.n_layers = n_layers

        self.conv_encoder1 = nn.Conv1d(in_channels=hidden_ch, out_channels=hidden_ch, kernel_size=1)
        self.conv_encoder2 = nn.Conv1d(in_channels=2 * hidden_ch, out_channels=hidden_ch, kernel_size=1)
        if shared_weights_decoder:
            self.decoder = nn.Sequential(
                nn.Conv1d(in_channels=2 * hidden_ch, out_channels=in_ch, kernel_size=1),
            )
        else:
            self.decoder = nn.Sequential(
                LocalConv1D(in_channels=2 * hidden_ch, out_channels=in_ch, kernel_size=1, n_nodes=n_nodes),
            )
        self.act = nn.LeakyReLU()

        self.cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GRUCell(d_in=in_ch if i == 0 else hidden_ch,
                                       num_units=hidden_ch, n_nodes=n_nodes, concat_h=concat_h))
        self.h0 = self.init_state(n_nodes)

    def init_state(self, n_nodes):
        h = []
        for _layer in range(self.n_layers):
            h.append(nn.parameter.Parameter(torch.zeros([self.hidden_ch, n_nodes])))
        return nn.ParameterList(h)

    def update_state(self, x, h, graph):
        rnn_in = x
        for layer in range(self.n_layers):
            rnn_in = h[layer] = self.cells[layer](rnn_in, h[layer], graph)
        return h

    def forward(self, x, mask, fwd_graph):
        # 'b n s c -> b c n s'
        x = x.permute(0, 3, 1, 2)
        bs, in_ch, n_nodes, steps = x.shape
        h = [h_.expand(bs, -1, -1) for h_ in self.h0.to(x.device)]

        pred = []
        for step in range(steps):
            x_now = x[..., step]
            h = self.update_state(x_now, h, fwd_graph)
            h_now = h[-1]
            x_repr = self.act(self.conv_encoder1(h_now))
            x_repr = self.act(self.conv_encoder2(torch.cat([x_repr, h_now], dim=1)))
            x_repr = torch.cat([x_repr, h_now], dim=1)
            x_hat2 = self.decoder(x_repr)
            pred.append(x_hat2)

        pred = torch.stack(pred, dim=-1)
        # 'b c n s -> b n s c'
        pred = pred.permute(0, 2, 3, 1)
        return pred[:, :, -1:]


# =============================================================================
# Ported (grouping/logging stripped, see file docstring) from
# CUTS_Plus/cuts_plus.py's `MultiCAD` class — MIT License, jarrycyx/UNN.
# =============================================================================

def _cuts_plus_generate_batch_starts(input_step: int, pred_step: int, t_length: int, rng) -> list:
    """Verbatim logic of the original's `generate_indices` with block_size
    == t_length (no coarse-to-fine block chunking — see file docstring).
    `rng` is the trainer's own seeded np.random.RandomState (the original
    draws from numpy's global generator, seeded once by its launcher)."""
    offsets = np.arange(input_step, t_length - pred_step + 1)
    starts = offsets.tolist()
    rng.shuffle(starts)
    return starts


def _cuts_plus_batch_generator(data, observ_mask, bs, n_nodes, input_step, pred_step, rng):
    """Native-torch port of the original's `batch_generater` (einops
    rearrange 't n d -> n t d' replaced by .permute(1, 0, 2))."""
    t_length = data.shape[0]
    starts = _cuts_plus_generate_batch_starts(input_step, pred_step, t_length, rng)
    n_batches = max(1, len(starts) // bs)
    for _b in range(n_batches):
        x = torch.zeros([bs, n_nodes, input_step, 1], device=data.device)
        y = torch.zeros([bs, n_nodes, pred_step, 1], device=data.device)
        t = torch.zeros([bs], dtype=torch.long, device=data.device)
        mask_x = torch.zeros([bs, n_nodes, input_step, 1], device=data.device)
        mask_y = torch.zeros([bs, n_nodes, pred_step, 1], device=data.device)
        for k in range(bs):
            data_t = starts.pop() if starts else rng.randint(input_step, t_length - pred_step + 1)
            x[k] = data[data_t - input_step: data_t, :].permute(1, 0, 2)
            y[k] = data[data_t: data_t + pred_step, :].permute(1, 0, 2)
            t[k] = data_t
            mask_x[k] = observ_mask[data_t - input_step: data_t, :].permute(1, 0, 2)
            mask_y[k] = observ_mask[data_t: data_t + pred_step, :].permute(1, 0, 2)
        yield x, y, t, mask_x, mask_y


class _CutsPlusOfficialTrainer:
    """
    Alternating "data prediction" (imputation) / "graph discovery"
    (differentiable Gumbel-sigmoid adjacency search) trainer, ported from
    CUTS_Plus/cuts_plus.py's `MultiCAD`, including the original's
    coarse-to-fine node-GROUPING schedule (`_maybe_update_grouping` below):
    early epochs constrain groups of SOURCE nodes to share one learned
    edge-probability row (`self.GT`, shape n_groups x n_nodes; `self.G`,
    shape n_nodes x n_groups, is the fixed hard group assignment), then the
    group count doubles periodically until it reaches n_nodes (full
    resolution, one row per individual source node). `n_groups_start`/
    `group_mul`/`group_every` default to the reference config's 32/2/20
    for N >= 64 and to no grouping below that (see run_pipeline_cuts_plus).

    `Graph = self.G @ sigmoid(self.GT)` (used for sampling during
    training; the returned/thresholded score is the raw G @ GT, as in the
    official code), shape (n_nodes, n_nodes), with
    `Graph[i, j]` = edge weight from source i to target j (MPNN convention,
    see class docstring above).
    """

    def __init__(self, n_nodes: int, hidden_ch: int, gru_layers: int,
                 shared_weights_decoder: bool, concat_h: bool,
                 lr_data_start: float, lr_data_end: float, weight_decay: float,
                 lr_graph_start: float, lr_graph_end: float,
                 lambda_s_start: float, lambda_s_end: float,
                 start_tau: float, end_tau: float,
                 n_groups_start: int, group_mul: int, group_every: int,
                 total_epoch: int, seed: int):
        torch.manual_seed(seed)
        # batch-order generator: local and seeded, so a run is reproducible
        # and numpy's global state is left untouched
        self.rng = np.random.RandomState(seed)
        self.n_nodes = n_nodes
        self.total_epoch = total_epoch

        self.model = CUTS_Plus_Net(n_nodes, in_ch=1, hidden_ch=hidden_ch, n_layers=gru_layers,
                                    shared_weights_decoder=shared_weights_decoder,
                                    concat_h=concat_h)
        self.mse = nn.MSELoss()
        self.data_opt = torch.optim.Adam(self.model.parameters(), lr=lr_data_start, weight_decay=weight_decay)
        data_gamma = (lr_data_end / lr_data_start) ** (1.0 / max(total_epoch, 1))
        self.data_sched = torch.optim.lr_scheduler.StepLR(self.data_opt, step_size=1, gamma=data_gamma)

        self.lr_graph_start = lr_graph_start
        self.lr_graph_end = lr_graph_end
        self.n_groups = min(max(1, n_groups_start), n_nodes)
        self.group_mul = max(2, group_mul)
        self.group_every = max(1, group_every)
        self.G = None
        self.GT = None
        self.graph_opt = None
        self.graph_sched = None

        self.tau = start_tau
        self.tau_gamma = (end_tau / start_tau) ** (1.0 / max(total_epoch, 1))
        self.lambda_s = lambda_s_start
        self.lambda_gamma = (lambda_s_end / lambda_s_start) ** (1.0 / max(total_epoch, 1))

    def _set_graph_optimizer(self, epoch: int):
        """Verbatim logic of the original's `set_graph_optimizer`: the LR
        picks up from where the geometric decay schedule would be at this
        epoch, rather than resetting to lr_graph_start on every group
        change (Adam's own per-parameter state IS reset here, since GT is a
        brand-new tensor with a new shape each time the grouping changes —
        matching the original, which also builds a fresh optimizer here)."""
        gamma = (self.lr_graph_end / self.lr_graph_start) ** (1.0 / max(self.total_epoch, 1))
        lr = self.lr_graph_start * (gamma ** epoch)
        self.graph_opt = torch.optim.Adam([self.GT], lr=lr)
        self.graph_sched = torch.optim.lr_scheduler.StepLR(self.graph_opt, step_size=1, gamma=gamma)

    def _maybe_update_grouping(self, epoch: int):
        """Verbatim port of `MultiCAD.train`'s group_policy block (the
        `group_policy is not None` branch — this project always uses a
        grouping schedule, so the `is None` early-exit of the original does
        not apply here)."""
        N = self.n_nodes
        if epoch % self.group_every == 0 and self.n_groups < N:
            if epoch != 0:
                self.n_groups = min(N, self.n_groups * self.group_mul)
            block = N // self.n_groups
            G = torch.zeros(N, self.n_groups)
            i = j = 0
            for i in range(self.n_groups):
                for j in range(block):
                    G[i * block + j, i] = 1
            for k in range(i * block + j, N):
                G[k, i] = 1
            self.G = G
            if self.GT is not None:
                # Verbatim quirk of the original: this "de-OR" formula
                # produces a probability-like value in [0, 1], which is
                # then assigned directly as the new raw (pre-sigmoid) GT —
                # the same probability/logit conflation as the 0.5 init.
                GT_init = torch.sigmoid(self.GT).detach().repeat_interleave(self.group_mul, 0)[:self.n_groups, :]
                GT_init = 1.0 - (1.0 - GT_init).clamp_min(1e-8) ** (1.0 / self.group_mul)
            else:
                # Verbatim quirk of the original: GT is initialized to the
                # raw value 0.5 (not a logit), so sigmoid(0.5) ~= 0.622 is
                # the actual starting edge probability, not exactly 0.5.
                GT_init = torch.ones(self.n_groups, N) * 0.5
            self.GT = nn.Parameter(GT_init)
            self._set_graph_optimizer(epoch)
        elif epoch == 0 and self.n_groups == N:
            self.G = torch.eye(N)
            self.GT = nn.Parameter(torch.ones(self.n_groups, N) * 0.5)
            self._set_graph_optimizer(epoch)

    def _current_graph(self) -> torch.Tensor:
        return self.G @ torch.sigmoid(self.GT)

    def _data_pred_step(self, x, y, mask_x, mask_y, batch_size):
        self.model.train()
        self.data_opt.zero_grad()
        Graph = self._current_graph()
        graph_sampled = torch.bernoulli(Graph[None].expand(batch_size, -1, -1))
        y_pred = self.model(x, mask_x, graph_sampled)
        loss = self.mse(y * mask_y, y_pred * mask_y) / mask_y.mean().clamp_min(1e-8)
        loss.backward()
        self.data_opt.step()
        return y_pred.detach(), float(loss.item())

    def _graph_discov_step(self, x, y, mask_x, mask_y, batch_size):
        self.graph_opt.zero_grad()
        Graph = self._current_graph()
        prob = Graph[None, :, :, None].expand(batch_size, -1, -1, -1)
        logits = torch.cat([prob, 1 - prob], dim=-1)
        graph_sampled = _F.gumbel_softmax(logits, tau=self.tau, hard=True)[:, :, :, 0]
        loss_sparsity = torch.linalg.norm(Graph.flatten(), ord=1) / (self.n_nodes ** 2)
        y_pred = self.model(x, mask_x, graph_sampled)
        loss_data = self.mse(y * mask_y, y_pred * mask_y) / mask_y.mean().clamp_min(1e-8)
        loss = self.lambda_s * loss_sparsity + loss_data
        loss.backward()
        self.graph_opt.step()
        return float(loss.item())

    def train(self, data: torch.Tensor, observ_mask: torch.Tensor,
              batch_size: int, input_step: int, pred_step: int,
              fill_rate: float, fill_after_epoch: int) -> np.ndarray:
        """
        data, observ_mask: (T, N, 1) CPU tensors.
        Returns the final (N, N) RAW score matrix G @ GT (no sigmoid),
        exactly what the official `MultiCAD.train` returns and thresholds
        (already expanded through the group assignment G, i.e. at full
        resolution regardless of how coarse the grouping was during
        training).
        """
        data = data.clone()
        data_pred = data.clone()  # observed points keep their true value, imputed points get the model's own prediction (updated each epoch)

        for epoch in range(self.total_epoch):
            self._maybe_update_grouping(epoch)

            # official: `if epoch_i + 1 > update_after` (starts AT epoch 20)
            if epoch >= fill_after_epoch:
                data = data * (1.0 - fill_rate) + data_pred * fill_rate

            batches = list(_cuts_plus_batch_generator(
                data, observ_mask, batch_size, self.n_nodes, input_step, pred_step, self.rng
            ))
            data_pred = data.clone()
            for x, y, t, mask_x, mask_y in batches:
                y_pred, _loss = self._data_pred_step(x, y, mask_x, mask_y, batch_size)
                data_pred[t] = (y_pred * (1 - mask_y) + y * mask_y).detach()[:, :, 0]
            self.data_sched.step()

            for x, y, t, mask_x, mask_y in batches:
                self._graph_discov_step(x, y, mask_x, mask_y, batch_size)
            self.graph_sched.step()

            self.tau *= self.tau_gamma
            self.lambda_s *= self.lambda_gamma

        return (self.G @ self.GT).detach().numpy()


def run_pipeline_cuts_plus(
    df: pd.DataFrame,
    date_col: str = "Data",
    entity_col: str = "Nome",
    property_cols: Optional[List[str]] = None,
    max_lag: int = 24,
    alpha_discovery: float = 0.20,
    cuts_tau_max: Optional[int] = None,
    cuts_hidden: int = 32,
    cuts_gru_layers: int = 1,
    cuts_shared_weights_decoder: bool = False,
    cuts_concat_h: bool = True,
    cuts_total_epoch: int = 64,
    cuts_batch_size: int = 128,
    cuts_lr_data_start: float = 1e-2,
    cuts_lr_data_end: float = 1e-3,
    cuts_weight_decay: float = 0.0,
    cuts_lr_graph_start: float = 1e-3,
    cuts_lr_graph_end: float = 1e-4,
    cuts_lambda_s_start: float = 1e-1,
    cuts_lambda_s_end: float = 1e-2,
    cuts_start_tau: float = 1.0,
    cuts_end_tau: float = 0.1,
    cuts_fill_rate: float = 0.1,
    cuts_fill_after_frac: float = 20.0 / 64.0,
    cuts_causal_threshold: float = 0.5,
    cuts_n_groups_start: Optional[int] = None,
    cuts_group_mul: int = 2,
    cuts_group_every: int = 20,
    cuts_seed: int = 0,
    cuts_sample_split: bool = False,
    cuts_ftest_gate: bool = True,
    cuts_correction: str = "none",
    n_jobs: int = 1,
    **_kwargs,
) -> pd.DataFrame:
    """
    OFFICIAL CUTS+ (Cheng et al., AAAI 2024) — drop-in in the runner:
    pipeline(df, **cfg) -> DataFrame with 'significant'.

    Method-specific parameters (prefix 'cuts_'), defaulting to the authors'
    own published Lorenz-96 example config (CUTS_Plus/opt/lorenz_example.yaml)
    except where noted:
        cuts_tau_max              : max lag tested by the POST-HOC F-test that
                                     assigns a p-value/dominant lag to each
                                     pair CUTS+ discovers (CUTS+ itself has no
                                     explicit per-lag structure — see file
                                     docstring). Default: min(max_lag, 10).
        cuts_hidden                : hidden channels of the GRU/MPNN net. 32.
        cuts_gru_layers            : recurrent layers. 1.
        cuts_shared_weights_decoder: shared vs. per-node decoder. False.
        cuts_concat_h              : concatenate hidden state into the MPNN
                                     message. True.
        cuts_total_epoch           : alternating data-pred/graph-discov epochs.
                                     64 (unchanged from the reference config;
                                     this project's series are much shorter
                                     than the authors' T=1000 Lorenz-96
                                     example, so 64 epochs remain cheap here).
        cuts_batch_size            : 128, capped automatically to the number
                                     of valid (input_step, pred_step) windows
                                     actually available in short series.
        cuts_lr_data_start/end     : Adam LR schedule (StepLR, geometric) for
                                     the data-prediction (imputation) network.
        cuts_weight_decay          : 0.0.
        cuts_lr_graph_start/end    : Adam LR schedule for the graph logits GT.
        cuts_lambda_s_start/end    : L1 sparsity penalty schedule on the
                                     learned graph.
        cuts_start_tau/end_tau     : Gumbel-softmax temperature annealing.
        cuts_fill_rate             : progressive self-training fill rate
                                     (the reference config's "rate_0.1").
        cuts_fill_after_frac       : fraction of cuts_total_epoch after which
                                     the fill policy activates — the reference
                                     config's "after_20" at total_epoch=64 is
                                     20/64; re-expressed as a fraction here so
                                     it scales with cuts_total_epoch if that is
                                     changed for a given scenario.
        cuts_causal_threshold      : ABSOLUTE threshold on the learned RAW
                                     score G @ GT (no sigmoid) to call a pair
                                     "discovered". 0.5 — the official code's
                                     calc_and_log_metrics(Graph,
                                     threshold=0.5) on the raw G @ GT.
        cuts_n_groups_start        : starting number of SOURCE-node groups
                                     for the coarse-to-fine schedule (see
                                     _CutsPlusOfficialTrainer). Default None
                                     -> 32 (the reference config's n_groups,
                                     set for N=128) when N >= 64, else N,
                                     i.e. NO grouping — the official
                                     `n_groups == n_nodes` branch. Reasons:
                                     (i) the official code cannot run
                                     n_groups > N (GT is never created);
                                     (ii) for 32 < N < 64 the paper's
                                     allocation |G_i| = floor(N / Ng) puts
                                     one series in each of the first 31
                                     groups, so there is nothing coarse to
                                     refine; (iii) every group split
                                     re-initializes the split rows below the
                                     initial value, which the fixed 0.5 cut
                                     is defined against.
        cuts_group_mul             : group-count multiplier each time the
                                     schedule advances. 2 (reference default).
        cuts_group_every           : epochs between grouping-schedule
                                     advances. Default 20 — the reference
                                     config's own value (group_policy:
                                     multiply_2_every_20). Pass None to divide
                                     cuts_total_epoch evenly across the
                                     (stages + 1) resolution levels instead.
        cuts_seed                  : random seed of the whole run (torch:
                                     weights, Bernoulli/Gumbel sampling;
                                     numpy: batch order). 0. Same seed ->
                                     same graph.
        cuts_sample_split          : if True, splits the series into two
                                     temporal halves — the first for CUTS+
                                     training (graph + imputation network),
                                     the second for the post-hoc F-test —
                                     eliminating double-dipping. Default False.
        cuts_ftest_gate            : True (default) requires the post-hoc
                                     F-test p-value (corrected with
                                     cuts_correction) to be < alpha; False
                                     keeps the NATIVE output — every pair
                                     whose graph score clears the threshold is
                                     significant, as in the paper (the
                                     p-value is still reported).
        cuts_correction            : 'none' (default) | 'bonferroni' |
                                     'fdr_bh', over the discovered pairs
                                     (see _posthoc_gate).
        n_jobs                     : unused (kept for interface compatibility
                                     with the runner; CUTS+'s alternating
                                     training loop is not
                                     per-target-parallelizable).

    Requires `torch` (already a project dependency via the DML cascade).
    Runs on CPU only (no GPU support).

    Flow:
        1) Loads and pivots the df, capturing the mask of originally
           observed values (_cuts_plus_load_and_pivot_with_mask).
        2) [If cuts_sample_split] splits (X, mask) into a training half
           (CUTS+'s own graph+imputation training) and an inference half
           (F-test).
        3) Trains the OFFICIAL CUTS+ architecture (alternating data-pred /
           graph-discov steps, Gumbel-sigmoid graph sampling, progressive
           self-training fill of originally-missing points) on the training
           half.
        4) Thresholds the learned raw G @ GT at cuts_causal_threshold to
           decide which (source, target) pairs are "discovered".
        5) For each discovered pair: OLS F-test conditioned on the other
           discovered parents -> (f_stat, pvalue, dom_lag), on the
           inference half when cuts_sample_split.
        6) Returns a DataFrame in the pipeline's standard format.
    """
    tau_max = cuts_tau_max if cuts_tau_max is not None else min(int(max_lag), 10)

    pivoted, observed_mask = _cuts_plus_load_and_pivot_with_mask(
        df, date_col=date_col, entity_col=entity_col, property_cols=property_cols
    )
    var_names = list(pivoted.columns)
    N = len(var_names)
    X = pivoted.values.astype(float)
    dev = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(dev < 1e-12, 1.0, dev)
    T = X.shape[0]
    mask_np = observed_mask.values.astype(float)

    if T <= tau_max + 5:
        raise ValueError(f"CUTS+: series too short (T={T}) for tau_max={tau_max}.")
    if N < 2:
        raise ValueError(f"CUTS+: only {N} valid series — need >= 2.")

    imputed_frac = 1.0 - mask_np.mean()

    # -- Data for the post-hoc F-test (grouped BY SOURCE VARIABLE; same
    #    convention as the other baselines in this runner) --
    X_lag_ftest = _ngc_build_lagged(X, tau_max)
    Y_ftest = X[tau_max:]

    if cuts_sample_split:
        cutoff = T // 2
        X_train, mask_train = X[:cutoff], mask_np[:cutoff]
        X_ftest_sel, Y_ftest_sel, _, X_ftest_inf, Y_ftest_inf, _ = _split_sample(X_lag_ftest, Y_ftest)
        if X_train.shape[0] <= tau_max + 5 or X_ftest_inf.shape[0] <= tau_max + 5:
            print(f"    [cuts+] WARNING: the split sample would be too short "
                  f"(T={T}) — ignoring cuts_sample_split for this scenario.")
            cuts_sample_split = False
            X_train, mask_train = X, mask_np
            X_ftest_inf, Y_ftest_inf = X_lag_ftest, Y_ftest
    else:
        X_train, mask_train = X, mask_np
        X_ftest_inf, Y_ftest_inf = X_lag_ftest, Y_ftest

    T_train = X_train.shape[0]
    input_step, pred_step = 1, 1
    n_valid_windows = max(1, T_train - input_step - pred_step + 1)
    batch_size = max(1, min(cuts_batch_size, n_valid_windows))
    fill_after_epoch = max(1, round(cuts_fill_after_frac * cuts_total_epoch))

    # -- Coarse-to-fine grouping schedule: reference n_groups=32 for N >= 64, else none (see docstring) --
    n_groups_start = cuts_n_groups_start if cuts_n_groups_start is not None else (32 if N >= 64 else N)
    n_groups_start = min(n_groups_start, N)
    n_stages = 0
    _ng = n_groups_start
    while _ng < N:
        _ng = min(N, _ng * cuts_group_mul)
        n_stages += 1
    group_every = cuts_group_every if cuts_group_every is not None else max(1, round(cuts_total_epoch / (n_stages + 1)))

    print(
        f"    [cuts+] N={N} vars, T={T} (train T={T_train}), tau_max={tau_max}, "
        f"{imputed_frac * 100:.1f}% of points were originally missing, "
        f"total_epoch={cuts_total_epoch}, batch_size={batch_size}, device=cpu, "
        f"grouping: {n_groups_start} -> {N} in {n_stages} stage(s) every {group_every} epochs, "
        f"sample_split={cuts_sample_split}, ftest_gate={cuts_ftest_gate}, correction={cuts_correction}, "
        f"official architecture (jarrycyx/UNN CUTS_Plus)"
    )

    data_t = torch.from_numpy(X_train[:, :, None]).float()
    mask_t = torch.from_numpy(mask_train[:, :, None]).float()

    trainer = _CutsPlusOfficialTrainer(
        n_nodes=N, hidden_ch=cuts_hidden, gru_layers=cuts_gru_layers,
        shared_weights_decoder=cuts_shared_weights_decoder, concat_h=cuts_concat_h,
        lr_data_start=cuts_lr_data_start, lr_data_end=cuts_lr_data_end, weight_decay=cuts_weight_decay,
        lr_graph_start=cuts_lr_graph_start, lr_graph_end=cuts_lr_graph_end,
        lambda_s_start=cuts_lambda_s_start, lambda_s_end=cuts_lambda_s_end,
        start_tau=cuts_start_tau, end_tau=cuts_end_tau,
        n_groups_start=n_groups_start, group_mul=cuts_group_mul, group_every=group_every,
        total_epoch=cuts_total_epoch, seed=cuts_seed,
    )
    graph_score = trainer.train(
        data_t, mask_t, batch_size=batch_size, input_step=input_step, pred_step=pred_step,
        fill_rate=cuts_fill_rate, fill_after_epoch=fill_after_epoch,
    )  # (N, N), raw G @ GT; graph_score[i, j] = score of source i -> target j

    discovered_by_target = {j: set() for j in range(N)}
    for i in range(N):
        for j in range(N):
            if i != j and graph_score[i, j] > cuts_causal_threshold:
                discovered_by_target[j].add(i)
    n_active = sum(len(v) for v in discovered_by_target.values())
    print(f"    [cuts+] official fit: {n_active} active edge(s) above cuts_causal_threshold={cuts_causal_threshold:g}")

    # -- F-test for p-value per discovered pair (on the inference half) -----
    rows = []
    for j in range(N):
        discovered = discovered_by_target[j]
        for i in range(N):
            if i == j or i not in discovered:
                continue
            cond = [p for p in discovered if p != i]
            f_val, pval, dom_lag = _ngc_ftest_pair(
                i, j, X_ftest_inf, Y_ftest_inf[:, j], N, tau_max, cond
            )

            src_entity, src_property = var_names[i].split("__", 1)
            dest_entity, dest_property = var_names[j].split("__", 1)
            if src_entity == dest_entity and src_property == dest_property:
                continue
            rows.append({
                "source_entity":    src_entity,
                "source_property":  src_property,
                "target_entity":   dest_entity,
                "target_property": dest_property,
                "lag":            dom_lag,
                "value":          f_val,
                "pvalue":        pval,
                "method":         "cuts_plus",
            })

    _posthoc_gate(rows, alpha_discovery, cuts_ftest_gate, cuts_correction)

    return pd.DataFrame(rows, columns=[
        "source_entity", "source_property", "target_entity", "target_property",
        "lag", "value", "pvalue", "pvalue_corrected", "significant", "method",
    ])
