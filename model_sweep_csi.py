# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_standardize(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.rank(axis=1, pct=True, na_option="keep")
    ranked = ranked - 0.5
    ranked = ranked.clip(-0.5, 0.5)
    return ranked.fillna(0.0)


@dataclass
class RunConfig:
    data_dir: Path
    output_dir: Path
    returns_file: str = "monthly_returns.pkl"
    lead_return: int = 1
    train_window: int = 120
    refit_interval: int = 12
    k_min: int = 1
    k_max: int = 6
    models: tuple[str, ...] = ("FF", "PCA", "IPCA", "CA0", "CA1", "CA2", "CA3")
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ipca_max_iter: int = 10
    ipca_tol: float = 1e-5
    ca_epochs: int = 15
    ca_lr: float = 1e-3
    ca_weight_decay: float = 1e-4
    ca_beta_hidden: tuple[int, ...] = (64, 32)
    ca_factor_hidden: tuple[int, ...] = (32,)
    ca_max_assets_train: int | None = 1500
    min_obs_per_asset: int = 24
    ff_q: float = 0.3
    ridge: float = 1e-5
    max_test_months: int | None = None
    use_cache: bool = False
    cache_file: str = "panel_cache_sweep.npz"


@dataclass
class Panel:
    returns: np.ndarray  # [T, N]
    chars: np.ndarray  # [T, N, L]
    dates: pd.DatetimeIndex
    assets: list[str]
    char_names: list[str]


def list_char_files(config: RunConfig) -> list[Path]:
    return sorted(p for p in config.data_dir.glob("*.pkl") if p.name != config.returns_file)


def load_panel(config: RunConfig) -> Panel:
    returns = pd.read_pickle(config.data_dir / config.returns_file)
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    if config.lead_return != 0:
        returns = returns.shift(-config.lead_return)

    char_files = list_char_files(config)
    if not char_files:
        raise RuntimeError(f"No characteristic files found in {config.data_dir}")

    dates = None
    cols_union: set[str] = set()
    for f in char_files:
        df = pd.read_pickle(f)
        df.index = pd.to_datetime(df.index)
        dates = df.index if dates is None else dates.intersection(df.index)
        cols_union.update(df.columns)

    dates = dates.intersection(returns.index).sort_values()
    assets = sorted(cols_union.intersection(returns.columns))

    t_len = len(dates)
    n_assets = len(assets)
    n_chars = len(char_files)
    x = np.empty((t_len, n_assets, n_chars), dtype=np.float32)
    names: list[str] = []

    for j, f in enumerate(char_files):
        df = pd.read_pickle(f)
        df = df.reindex(index=dates, columns=assets)
        df = rank_standardize(df)
        x[:, :, j] = df.to_numpy(dtype=np.float32)
        names.append(f.stem)

    r = returns.reindex(index=dates, columns=assets).to_numpy(dtype=np.float32)
    return Panel(returns=r, chars=x, dates=pd.DatetimeIndex(dates), assets=assets, char_names=names)


def load_or_build_panel(config: RunConfig) -> Panel:
    cache_path = config.output_dir / config.cache_file
    if config.use_cache and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return Panel(
            returns=data["returns"],
            chars=data["chars"],
            dates=pd.DatetimeIndex(pd.to_datetime(data["dates"])),
            assets=data["assets"].tolist(),
            char_names=data["char_names"].tolist(),
        )

    panel = load_panel(config)
    if config.use_cache:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            returns=panel.returns,
            chars=panel.chars,
            dates=np.array(panel.dates, dtype="datetime64[ns]"),
            assets=np.array(panel.assets),
            char_names=np.array(panel.char_names),
        )
    return panel


def solve_factor(beta: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    k = beta.shape[1]
    a = beta.T @ beta + ridge * np.eye(k, dtype=np.float64)
    b = beta.T @ y
    return np.linalg.solve(a, b)


def r2_from_sse(sse: float, sst: float) -> float:
    if sst <= 0:
        return np.nan
    return 1.0 - sse / sst


class FitPredictModel(Protocol):
    def fit(self, x_train: np.ndarray, r_train: np.ndarray) -> None: ...

    def predict_total(self, x_t: np.ndarray, r_t: np.ndarray) -> np.ndarray: ...

    def predict_predicted(self, x_t: np.ndarray) -> np.ndarray: ...


class FFModel:
    def __init__(self, k: int, char_names: list[str], cfg: RunConfig):
        self.k = k
        self.cfg = cfg
        self.beta: np.ndarray | None = None
        self.mu_f: np.ndarray | None = None
        self.signs: list[int] = []
        self.ff_indices = self._select_ff_indices(char_names)

    @staticmethod
    def _pick_first(candidates: list[str], name_to_idx: dict[str, int]) -> int:
        for c in candidates:
            if c in name_to_idx:
                return name_to_idx[c]
        raise KeyError(f"None of candidates found: {candidates}")

    def _select_ff_indices(self, char_names: list[str]) -> list[int]:
        name_to_idx = {n: i for i, n in enumerate(char_names)}
        idx_size = self._pick_first(["total_mv"], name_to_idx)
        idx_value = self._pick_first(
            ["book_value_to_total_mktcap_mrq", "book_value_plus_rdexp_to_total_mktcap_ttm"],
            name_to_idx,
        )
        idx_profit = self._pick_first(
            ["operating_profit_to_book_value_ttm", "net_profit_to_asset_ttm", "aqr_profitability"],
            name_to_idx,
        )
        idx_invest = self._pick_first(["asset_growth_mrq", "investment_to_asset"], name_to_idx)
        idx_mom = self._pick_first(["momentum_academic_252_21", "momentum_21"], name_to_idx)
        # Market + 5 style factors => max 6
        # signs apply to style factors only: SIZE(-), VALUE(+), PROFIT(+), INVEST(-), MOM(+)
        self.signs = [-1, 1, 1, -1, 1]
        return [idx_size, idx_value, idx_profit, idx_invest, idx_mom]

    def _single_month_f(self, x_t: np.ndarray, r_t: np.ndarray) -> np.ndarray:
        mask_r = ~np.isnan(r_t)
        y = r_t[mask_r].astype(np.float64)
        out = np.full(self.k, np.nan, dtype=np.float64)
        if y.size < 10:
            return out

        out[0] = y.mean()  # market factor
        q = self.cfg.ff_q
        for j in range(1, self.k):
            idx_char = self.ff_indices[j - 1]
            sgn = self.signs[j - 1]
            z = x_t[mask_r, idx_char].astype(np.float64)
            lo = np.quantile(z, q)
            hi = np.quantile(z, 1.0 - q)
            lo_mask = z <= lo
            hi_mask = z >= hi
            if lo_mask.sum() < 3 or hi_mask.sum() < 3:
                out[j] = np.nan
                continue
            long_ret = y[hi_mask].mean()
            short_ret = y[lo_mask].mean()
            hm = long_ret - short_ret
            out[j] = hm if sgn > 0 else -hm
        return out

    def fit(self, x_train: np.ndarray, r_train: np.ndarray) -> None:
        t_len, n_assets, _ = x_train.shape
        f_train = np.full((t_len, self.k), np.nan, dtype=np.float64)
        for t in range(t_len):
            f_train[t] = self._single_month_f(x_train[t], r_train[t])

        beta = np.full((n_assets, self.k), np.nan, dtype=np.float64)
        for i in range(n_assets):
            y = r_train[:, i]
            mask = ~np.isnan(y) & ~np.isnan(f_train).any(axis=1)
            if mask.sum() < max(self.cfg.min_obs_per_asset, self.k + 5):
                continue
            f = f_train[mask]
            yy = y[mask].astype(np.float64)
            try:
                beta[i] = np.linalg.lstsq(f, yy, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

        self.beta = beta
        self.mu_f = np.nanmean(f_train, axis=0)

    def predict_total(self, x_t: np.ndarray, r_t: np.ndarray) -> np.ndarray:
        f_t = self._single_month_f(x_t, r_t)
        pred = np.full_like(r_t, np.nan, dtype=np.float32)
        if self.beta is None or np.isnan(f_t).any():
            return pred
        beta = self.beta
        mask = ~np.isnan(r_t) & ~np.isnan(beta).any(axis=1)
        if mask.sum() == 0:
            return pred
        pred[mask] = (beta[mask] @ f_t).astype(np.float32)
        return pred

    def predict_predicted(self, x_t: np.ndarray) -> np.ndarray:
        pred = np.full(x_t.shape[0], np.nan, dtype=np.float32)
        if self.beta is None or self.mu_f is None or np.isnan(self.mu_f).any():
            return pred
        beta = self.beta
        mask = ~np.isnan(beta).any(axis=1)
        if mask.sum() == 0:
            return pred
        pred[mask] = (beta[mask] @ self.mu_f).astype(np.float32)
        return pred


class PCAModel:
    def __init__(self, k: int, cfg: RunConfig):
        self.k = k
        self.cfg = cfg
        self.beta: np.ndarray | None = None
        self.mu_f: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, r_train: np.ndarray) -> None:
        del x_train
        r_fill = np.nan_to_num(r_train.astype(np.float64), nan=0.0)
        _, _, vt = np.linalg.svd(r_fill, full_matrices=False)
        beta = vt[: self.k].T  # [N, K]

        # Normalize columns.
        norms = np.linalg.norm(beta, axis=0, keepdims=True) + 1e-12
        beta = beta / norms

        factors = np.full((r_train.shape[0], self.k), np.nan, dtype=np.float64)
        for t in range(r_train.shape[0]):
            y = r_train[t]
            mask = ~np.isnan(y)
            if mask.sum() < self.k + 2:
                continue
            b = beta[mask]
            yy = y[mask].astype(np.float64)
            factors[t] = solve_factor(b, yy, self.cfg.ridge)

        self.beta = beta
        self.mu_f = np.nanmean(factors, axis=0)

    def predict_total(self, x_t: np.ndarray, r_t: np.ndarray) -> np.ndarray:
        del x_t
        pred = np.full_like(r_t, np.nan, dtype=np.float32)
        if self.beta is None:
            return pred
        mask = ~np.isnan(r_t)
        if mask.sum() < self.k + 2:
            return pred
        b = self.beta[mask]
        yy = r_t[mask].astype(np.float64)
        f_t = solve_factor(b, yy, self.cfg.ridge)
        pred[mask] = (b @ f_t).astype(np.float32)
        return pred

    def predict_predicted(self, x_t: np.ndarray) -> np.ndarray:
        del x_t
        pred = np.full(self.beta.shape[0], np.nan, dtype=np.float32) if self.beta is not None else np.array([], dtype=np.float32)
        if self.beta is None or self.mu_f is None or np.isnan(self.mu_f).any():
            return pred
        pred[:] = (self.beta @ self.mu_f).astype(np.float32)
        return pred


class IPCAModel:
    def __init__(self, k: int, cfg: RunConfig):
        self.k = k
        self.cfg = cfg
        self.gamma: np.ndarray | None = None  # [L, K]
        self.mu_f: np.ndarray | None = None  # [K]

    def _init_gamma(self, x_train: np.ndarray, r_train: np.ndarray) -> np.ndarray:
        t_len, _, l = x_train.shape
        managed = np.full((t_len, l), 0.0, dtype=np.float64)
        for t in range(t_len):
            y = r_train[t]
            mask = ~np.isnan(y)
            if mask.sum() < 10:
                continue
            xt = x_train[t, mask].astype(np.float64)
            yt = y[mask].astype(np.float64)
            managed[t] = (xt.T @ yt) / max(mask.sum(), 1)
        _, _, vt = np.linalg.svd(managed, full_matrices=False)
        gamma = vt[: self.k].T
        q, _ = np.linalg.qr(gamma)
        return q[:, : self.k]

    def fit(self, x_train: np.ndarray, r_train: np.ndarray) -> None:
        t_len, _, l = x_train.shape
        gamma = self._init_gamma(x_train, r_train)
        f_hist = np.zeros((t_len, self.k), dtype=np.float64)

        for _ in range(self.cfg.ipca_max_iter):
            # Step 1: estimate factors with current gamma.
            for t in range(t_len):
                y = r_train[t]
                mask = ~np.isnan(y)
                if mask.sum() < self.k + 2:
                    f_hist[t] = np.nan
                    continue
                xt = x_train[t, mask].astype(np.float64)
                yt = y[mask].astype(np.float64)
                beta = xt @ gamma
                f_hist[t] = solve_factor(beta, yt, self.cfg.ridge)

            # Step 2: estimate gamma with fixed factors by normal equations.
            lk = l * self.k
            a = np.zeros((lk, lk), dtype=np.float64)
            b = np.zeros(lk, dtype=np.float64)
            valid_t = 0
            for t in range(t_len):
                f_t = f_hist[t]
                if np.isnan(f_t).any():
                    continue
                y = r_train[t]
                mask = ~np.isnan(y)
                if mask.sum() < self.k + 2:
                    continue
                xt = x_train[t, mask].astype(np.float64)
                yt = y[mask].astype(np.float64)
                xtx = xt.T @ xt
                xtr = xt.T @ yt
                ff = np.outer(f_t, f_t)
                a += np.kron(ff, xtx)
                b += np.kron(f_t, xtr)
                valid_t += 1
            if valid_t == 0:
                break
            a += self.cfg.ridge * np.eye(lk, dtype=np.float64)
            try:
                z = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                z = np.linalg.lstsq(a, b, rcond=None)[0]
            gamma_new = z.reshape((l, self.k), order="F")

            # Identification: orthogonalize gamma and rotate factors.
            q, r = np.linalg.qr(gamma_new)
            gamma_new = q[:, : self.k]
            f_hist = f_hist @ r[: self.k, : self.k].T

            diff = np.linalg.norm(gamma_new - gamma) / (np.linalg.norm(gamma) + 1e-12)
            gamma = gamma_new
            if diff < self.cfg.ipca_tol:
                break

        self.gamma = gamma
        self.mu_f = np.nanmean(f_hist, axis=0)

    def predict_total(self, x_t: np.ndarray, r_t: np.ndarray) -> np.ndarray:
        pred = np.full_like(r_t, np.nan, dtype=np.float32)
        if self.gamma is None:
            return pred
        mask = ~np.isnan(r_t)
        if mask.sum() < self.k + 2:
            return pred
        xt = x_t[mask].astype(np.float64)
        yt = r_t[mask].astype(np.float64)
        beta = xt @ self.gamma
        f_t = solve_factor(beta, yt, self.cfg.ridge)
        pred[mask] = (beta @ f_t).astype(np.float32)
        return pred

    def predict_predicted(self, x_t: np.ndarray) -> np.ndarray:
        pred = np.full(x_t.shape[0], np.nan, dtype=np.float32)
        if self.gamma is None or self.mu_f is None or np.isnan(self.mu_f).any():
            return pred
        beta = x_t.astype(np.float64) @ self.gamma
        pred[:] = (beta @ self.mu_f).astype(np.float32)
        return pred


def make_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ConditionalAutoencoder(nn.Module):
    def __init__(self, n_chars: int, k: int, beta_hidden: tuple[int, ...], factor_hidden: tuple[int, ...]):
        super().__init__()
        self.beta_net = make_mlp(n_chars, beta_hidden, k)
        self.factor_net = make_mlp(n_chars, factor_hidden, k)

    def latent_factor(self, x_t: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
        n_obs = x_t.shape[0]
        m_t = (x_t.T @ r_t) / max(n_obs, 1)
        return self.factor_net(m_t.unsqueeze(0)).squeeze(0)

    def forward(self, x_t: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
        beta = self.beta_net(x_t)
        f_t = self.latent_factor(x_t, r_t)
        return (beta * f_t).sum(dim=1)


class CAModel:
    def __init__(self, variant: str, k: int, n_chars: int, cfg: RunConfig):
        self.variant = variant.upper()
        self.k = k
        self.n_chars = n_chars
        self.cfg = cfg
        self.model: ConditionalAutoencoder | None = None
        self.mu_f: np.ndarray | None = None

    def _hidden(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        beta_hidden: tuple[int, ...] = ()
        factor_hidden: tuple[int, ...] = ()
        if self.variant in {"CA1", "CA3"}:
            beta_hidden = self.cfg.ca_beta_hidden
        if self.variant in {"CA2", "CA3"}:
            factor_hidden = self.cfg.ca_factor_hidden
        return beta_hidden, factor_hidden

    def _extract_f(self, x_t: np.ndarray, r_t: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(self.k, np.nan, dtype=np.float32)
        device = torch.device(self.cfg.device)
        mask = ~np.isnan(r_t)
        if mask.sum() < self.k + 2:
            return np.full(self.k, np.nan, dtype=np.float32)
        xv = torch.from_numpy(x_t[mask]).to(device)
        yv = torch.from_numpy(r_t[mask]).to(device)
        with torch.no_grad():
            f = self.model.latent_factor(xv, yv).cpu().numpy().astype(np.float32)
        return f

    def fit(self, x_train: np.ndarray, r_train: np.ndarray) -> None:
        device = torch.device(self.cfg.device)
        beta_hidden, factor_hidden = self._hidden()
        model = ConditionalAutoencoder(self.n_chars, self.k, beta_hidden, factor_hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.ca_lr, weight_decay=self.cfg.ca_weight_decay)
        loss_fn = nn.MSELoss()

        idx = list(range(r_train.shape[0]))
        for _ in range(self.cfg.ca_epochs):
            np.random.shuffle(idx)
            for t in idx:
                y = r_train[t]
                mask = ~np.isnan(y)
                if mask.sum() < self.k + 2:
                    continue
                if self.cfg.ca_max_assets_train is not None and mask.sum() > self.cfg.ca_max_assets_train:
                    valid_idx = np.flatnonzero(mask)
                    choose = np.random.choice(valid_idx, size=self.cfg.ca_max_assets_train, replace=False)
                    xv_np = x_train[t, choose]
                    yv_np = y[choose]
                else:
                    xv_np = x_train[t, mask]
                    yv_np = y[mask]

                xv = torch.from_numpy(xv_np).to(device)
                yv = torch.from_numpy(yv_np).to(device)
                pred = model(xv, yv)
                loss = loss_fn(pred, yv)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.model = model

        # Average latent factors on training sample for predicted R^2.
        f_hist = np.full((r_train.shape[0], self.k), np.nan, dtype=np.float32)
        for t in range(r_train.shape[0]):
            f_hist[t] = self._extract_f(x_train[t], r_train[t])
        self.mu_f = np.nanmean(f_hist, axis=0)

    def predict_total(self, x_t: np.ndarray, r_t: np.ndarray) -> np.ndarray:
        pred = np.full_like(r_t, np.nan, dtype=np.float32)
        if self.model is None:
            return pred
        device = torch.device(self.cfg.device)
        mask = ~np.isnan(r_t)
        if mask.sum() < self.k + 2:
            return pred
        xv = torch.from_numpy(x_t[mask]).to(device)
        yv = torch.from_numpy(r_t[mask]).to(device)
        with torch.no_grad():
            yh = self.model(xv, yv).cpu().numpy().astype(np.float32)
        pred[mask] = yh
        return pred

    def predict_predicted(self, x_t: np.ndarray) -> np.ndarray:
        pred = np.full(x_t.shape[0], np.nan, dtype=np.float32)
        if self.model is None or self.mu_f is None or np.isnan(self.mu_f).any():
            return pred
        device = torch.device(self.cfg.device)
        xv = torch.from_numpy(x_t).to(device)
        f = torch.from_numpy(self.mu_f.astype(np.float32)).to(device)
        with torch.no_grad():
            beta = self.model.beta_net(xv)
            yh = (beta * f).sum(dim=1).cpu().numpy().astype(np.float32)
        pred[:] = yh
        return pred


def evaluate_rolling(
    model_name: str,
    k: int,
    panel: Panel,
    cfg: RunConfig,
) -> dict[str, float | int | str]:
    r = panel.returns
    x = panel.chars
    t_len = r.shape[0]
    start = cfg.train_window
    end = t_len
    if cfg.max_test_months is not None:
        end = min(t_len, start + cfg.max_test_months)

    sse_total = 0.0
    sst_total = 0.0
    sse_pred = 0.0
    sst_pred = 0.0
    obs_total = 0
    obs_pred = 0
    refits = 0
    model: FitPredictModel | None = None

    for t in range(start, end):
        if model is None or (t - start) % cfg.refit_interval == 0:
            refits += 1
            x_train = x[t - cfg.train_window : t]
            r_train = r[t - cfg.train_window : t]
            if model_name == "FF":
                model = FFModel(k, panel.char_names, cfg)
            elif model_name == "PCA":
                model = PCAModel(k, cfg)
            elif model_name == "IPCA":
                model = IPCAModel(k, cfg)
            elif model_name in {"CA0", "CA1", "CA2", "CA3"}:
                model = CAModel(model_name, k, x.shape[2], cfg)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            model.fit(x_train, r_train)

        actual = r[t]
        pred_total = model.predict_total(x[t], actual)
        pred_pred = model.predict_predicted(x[t])

        m_total = ~np.isnan(actual) & ~np.isnan(pred_total)
        if m_total.sum() > 0:
            err = actual[m_total] - pred_total[m_total]
            sse_total += float(np.sum(err.astype(np.float64) ** 2))
            sst_total += float(np.sum(actual[m_total].astype(np.float64) ** 2))
            obs_total += int(m_total.sum())

        m_pred = ~np.isnan(actual) & ~np.isnan(pred_pred)
        if m_pred.sum() > 0:
            err = actual[m_pred] - pred_pred[m_pred]
            sse_pred += float(np.sum(err.astype(np.float64) ** 2))
            sst_pred += float(np.sum(actual[m_pred].astype(np.float64) ** 2))
            obs_pred += int(m_pred.sum())

    return {
        "model": model_name,
        "K": k,
        "total_r2": r2_from_sse(sse_total, sst_total),
        "predicted_r2": r2_from_sse(sse_pred, sst_pred),
        "obs_total": obs_total,
        "obs_predicted": obs_pred,
        "refits": refits,
        "test_months": max(end - start, 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FF/PCA/IPCA/CA0-3 model sweep on CSI characteristics.")
    default_data_dir = Path(__file__).resolve().parents[2] / "数据" / "features500"
    default_out = Path(__file__).resolve().parent / "output_sweep"

    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--refit-interval", type=int, default=12)
    parser.add_argument("--lead-return", type=int, default=1)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--models", type=str, default="FF,PCA,IPCA,CA0,CA1,CA2,CA3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--ipca-max-iter", type=int, default=10)
    parser.add_argument("--ipca-tol", type=float, default=1e-5)
    parser.add_argument("--ca-epochs", type=int, default=15)
    parser.add_argument("--ca-lr", type=float, default=1e-3)
    parser.add_argument("--ca-weight-decay", type=float, default=1e-4)
    parser.add_argument("--ca-max-assets-train", type=int, default=1500)
    parser.add_argument("--min-obs-per-asset", type=int, default=24)
    parser.add_argument("--ff-q", type=float, default=0.3)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--max-test-months", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = tuple(m.strip().upper() for m in args.models.split(",") if m.strip())
    cfg = RunConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lead_return=args.lead_return,
        train_window=args.train_window,
        refit_interval=args.refit_interval,
        k_min=args.k_min,
        k_max=args.k_max,
        models=models,
        seed=args.seed,
        device=args.device,
        ipca_max_iter=args.ipca_max_iter,
        ipca_tol=args.ipca_tol,
        ca_epochs=args.ca_epochs,
        ca_lr=args.ca_lr,
        ca_weight_decay=args.ca_weight_decay,
        ca_max_assets_train=args.ca_max_assets_train,
        min_obs_per_asset=args.min_obs_per_asset,
        ff_q=args.ff_q,
        ridge=args.ridge,
        max_test_months=args.max_test_months,
        use_cache=args.use_cache,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    panel = load_or_build_panel(cfg)
    print(
        f"Panel loaded: T={panel.returns.shape[0]}, N={panel.returns.shape[1]}, "
        f"L={panel.chars.shape[2]}, start={panel.dates.min().date()}, end={panel.dates.max().date()}"
    )

    results: list[dict[str, float | int | str]] = []
    for model in cfg.models:
        for k in range(cfg.k_min, cfg.k_max + 1):
            print(f"Running {model} with K={k} ...")
            row = evaluate_rolling(model, k, panel, cfg)
            results.append(row)
            print(
                f"  total_r2={row['total_r2']:.6f}, predicted_r2={row['predicted_r2']:.6f}, "
                f"obs_total={row['obs_total']}, obs_pred={row['obs_predicted']}"
            )

    out = pd.DataFrame(results).sort_values(["model", "K"]).reset_index(drop=True)
    out.to_csv(cfg.output_dir / "model_r2_summary.csv", index=False)

    pivot_total = out.pivot(index="model", columns="K", values="total_r2")
    pivot_pred = out.pivot(index="model", columns="K", values="predicted_r2")
    pivot_total.to_csv(cfg.output_dir / "model_r2_total_pivot.csv")
    pivot_pred.to_csv(cfg.output_dir / "model_r2_predicted_pivot.csv")

    print("Saved:")
    print(cfg.output_dir / "model_r2_summary.csv")
    print(cfg.output_dir / "model_r2_total_pivot.csv")
    print(cfg.output_dir / "model_r2_predicted_pivot.csv")


if __name__ == "__main__":
    main()
