# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from torch import nn


@dataclass
class Config:
    data_dir: Path = Path(r"D:\量化\paper\数据\features500")
    output_dir: Path = Path(__file__).resolve().parent / "output"
    returns_file: str = "monthly_returns.pkl"
    lead_return: int = 1
    train_window: int = 120
    refit_interval: int = 12
    n_factors: int = 5
    beta_hidden: tuple[int, ...] = (64, 32)
    factor_hidden: tuple[int, ...] = (32,)
    model_type: str = "CA3"  # CA0, CA1, CA2, CA3
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_value_weight: bool = True
    mv_factor_name: str = "total_mv.pkl"
    use_cache: bool = False
    cache_file: str = "panel_cache.npz"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_factor_files(config: Config) -> list[Path]:
    files = sorted(p for p in config.data_dir.glob("*.pkl") if p.name != config.returns_file)
    return files


def rank_standardize(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.rank(axis=1, pct=True, na_option="keep")
    ranked = ranked - 0.5
    ranked = ranked.clip(-0.5, 0.5)
    return ranked.fillna(0.0)


def load_panel(config: Config):
    returns = pd.read_pickle(config.data_dir / config.returns_file)
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    if config.lead_return != 0:
        returns = returns.shift(-config.lead_return)

    factor_files = list_factor_files(config)

    dates = None
    cols_union: set[str] = set()
    for f in factor_files:
        df = pd.read_pickle(f)
        df.index = pd.to_datetime(df.index)
        cols_union.update(df.columns)
        dates = df.index if dates is None else dates.intersection(df.index)

    if dates is None:
        raise RuntimeError("No factor files found.")

    dates = dates.intersection(returns.index)
    dates = dates.sort_values()
    assets = sorted(cols_union.intersection(returns.columns))

    mv = None
    if config.use_value_weight:
        mv_path = config.data_dir / config.mv_factor_name
        if mv_path.exists():
            mv_df = pd.read_pickle(mv_path)
            mv_df = mv_df.reindex(index=dates, columns=assets)
            mv = mv_df.to_numpy(dtype=np.float32)

    t_len = len(dates)
    n_assets = len(assets)
    n_factors = len(factor_files)
    x = np.empty((t_len, n_assets, n_factors), dtype=np.float32)
    factor_names: list[str] = []

    for j, f in enumerate(factor_files):
        df = pd.read_pickle(f)
        df = df.reindex(index=dates, columns=assets)
        df = rank_standardize(df)
        x[:, :, j] = df.to_numpy(dtype=np.float32)
        factor_names.append(f.stem)

    r = returns.reindex(index=dates, columns=assets).to_numpy(dtype=np.float32)
    return r, x, dates, assets, factor_names, mv


def load_or_build_panel(config: Config):
    cache_path = config.output_dir / config.cache_file
    if config.use_cache and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        r = data["r"]
        x = data["x"]
        dates = pd.to_datetime(data["dates"])
        assets = data["assets"].tolist()
        factor_names = data["factor_names"].tolist()
        mv = data["mv"] if "mv" in data else None
        return r, x, dates, assets, factor_names, mv

    r, x, dates, assets, factor_names, mv = load_panel(config)
    if config.use_cache:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            r=r,
            x=x,
            dates=np.array(dates, dtype="datetime64[ns]"),
            assets=np.array(assets),
            factor_names=np.array(factor_names),
            mv=mv,
        )
    return r, x, dates, assets, factor_names, mv


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
    def __init__(self, n_chars: int, n_factors: int, beta_hidden, factor_hidden):
        super().__init__()
        self.beta_net = make_mlp(n_chars, beta_hidden, n_factors)
        self.factor_net = make_mlp(n_chars, factor_hidden, n_factors)

    def forward(self, x_t: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
        beta = self.beta_net(x_t)  # (N, K)
        n_obs = x_t.shape[0]
        m_t = (x_t.T @ r_t) / max(n_obs, 1)
        f_t = self.factor_net(m_t.unsqueeze(0)).squeeze(0)  # (K,)
        r_hat = (beta * f_t).sum(dim=1)
        return r_hat


def build_model(config: Config, n_chars: int) -> ConditionalAutoencoder:
    model_type = config.model_type.upper()
    if model_type not in {"CA0", "CA1", "CA2", "CA3"}:
        raise ValueError("model_type must be one of CA0, CA1, CA2, CA3")

    beta_hidden = ()
    factor_hidden = ()
    if model_type in {"CA1", "CA3"}:
        beta_hidden = config.beta_hidden
    if model_type in {"CA2", "CA3"}:
        factor_hidden = config.factor_hidden

    return ConditionalAutoencoder(n_chars, config.n_factors, beta_hidden, factor_hidden)


def train_model(r: np.ndarray, x: np.ndarray, train_idx: list[int], config: Config) -> ConditionalAutoencoder:
    device = torch.device(config.device)
    model = build_model(config, x.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(config.epochs):
        np.random.shuffle(train_idx)
        for t in train_idx:
            r_t = torch.from_numpy(r[t]).to(device)
            x_t = torch.from_numpy(x[t]).to(device)
            mask = ~torch.isnan(r_t)
            if mask.sum() < 10:
                continue
            r_t = r_t[mask]
            x_t = x_t[mask]
            pred = model(x_t, r_t)
            loss = loss_fn(pred, r_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def predict_month(model: ConditionalAutoencoder, x_t: np.ndarray, r_t: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    r_tensor = torch.from_numpy(r_t).to(device)
    x_tensor = torch.from_numpy(x_t).to(device)
    mask = ~torch.isnan(r_tensor)
    if mask.sum() < 1:
        return np.full_like(r_t, np.nan)
    r_masked = r_tensor[mask]
    x_masked = x_tensor[mask]
    pred = model(x_masked, r_masked).cpu().numpy()
    out = np.full_like(r_t, np.nan, dtype=np.float32)
    out[mask.cpu().numpy()] = pred
    return out


def compute_oos_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    sse = np.sum((actual[mask] - predicted[mask]) ** 2)
    sst = np.sum(actual[mask] ** 2)
    if sst == 0:
        return np.nan
    return 1.0 - sse / sst


def rolling_fit_predict(r: np.ndarray, x: np.ndarray, dates: pd.Index, config: Config):
    t_len = len(dates)
    start = config.train_window
    preds = np.full_like(r, np.nan, dtype=np.float32)
    r2_rows = []
    model = None
    device = torch.device(config.device)

    for t in range(start, t_len):
        if model is None or (t - start) % config.refit_interval == 0:
            train_idx = list(range(t - config.train_window, t))
            model = train_model(r, x, train_idx, config)

        preds[t] = predict_month(model, x[t], r[t], str(device))
        r2 = compute_oos_r2(r[t], preds[t])
        r2_rows.append((dates[t], r2))

    r2_df = pd.DataFrame(r2_rows, columns=["date", "r2"]).set_index("date")
    return preds, r2_df


def decile_portfolios(preds: np.ndarray, r: np.ndarray, dates: pd.Index, mv: np.ndarray | None, n=10):
    decile_returns = {k: [] for k in range(n)}
    ls_returns = []
    used_dates = []

    for t, dt in enumerate(dates):
        score = preds[t]
        ret = r[t]
        mask = ~np.isnan(score) & ~np.isnan(ret)
        if mask.sum() < n:
            continue
        score_m = score[mask]
        ret_m = ret[mask]
        if mv is not None:
            mv_m = mv[t][mask]
        else:
            mv_m = None

        try:
            bins = pd.qcut(score_m, n, labels=False, duplicates="drop")
        except ValueError:
            continue

        used_dates.append(dt)
        for k in range(n):
            idx = bins == k
            if idx.sum() == 0:
                decile_returns[k].append(np.nan)
                continue
            if mv_m is not None:
                w = mv_m[idx].copy()
                w = np.where(np.isnan(w), 0.0, w)
                if w.sum() == 0:
                    w = np.ones_like(w)
                w = w / w.sum()
            else:
                w = np.ones(idx.sum()) / idx.sum()
            decile_returns[k].append(np.sum(w * ret_m[idx]))

        ls_returns.append(decile_returns[n - 1][-1] - decile_returns[0][-1])

    decile_df = pd.DataFrame(decile_returns, index=used_dates)
    ls_series = pd.Series(ls_returns, index=used_dates, name="LS")
    return decile_df, ls_series


def newey_west_tstat(x: pd.Series, lags: int = 6) -> tuple[float, float]:
    x = x.dropna()
    if x.empty:
        return np.nan, np.nan
    model = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return model.params[0], model.tvalues[0]


def summarize_long_short(ls: pd.Series) -> pd.DataFrame:
    mean = ls.mean()
    vol = ls.std()
    ann_mean = mean * 12
    ann_vol = vol * math.sqrt(12)
    sharpe = ann_mean / ann_vol if ann_vol != 0 else np.nan
    nw_mean, nw_t = newey_west_tstat(ls)
    return pd.DataFrame(
        {
            "mean": [mean],
            "vol": [vol],
            "ann_mean": [ann_mean],
            "ann_vol": [ann_vol],
            "sharpe": [sharpe],
            "nw_mean": [nw_mean],
            "nw_tstat": [nw_t],
        }
    )


def plot_cumulative(series: pd.Series, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    cum = (1 + series.fillna(0)).cumprod()
    plt.plot(cum.index, cum.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_r2(r2: pd.Series, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    plt.plot(r2.index, r2.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("OOS R2")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    config = Config()
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    r, x, dates, assets, factor_names, mv = load_or_build_panel(config)

    preds, r2_df = rolling_fit_predict(r, x, dates, config)

    decile_df, ls_series = decile_portfolios(preds, r, dates, mv)
    stats_df = summarize_long_short(ls_series)

    r2_df.to_csv(config.output_dir / "oos_r2.csv")
    decile_df.to_csv(config.output_dir / "decile_returns.csv")
    ls_series.to_csv(config.output_dir / "long_short.csv")
    stats_df.to_csv(config.output_dir / "long_short_stats.csv", index=False)

    plot_cumulative(ls_series, "Long-Short Cumulative Return", config.output_dir / "long_short_cum.png")
    plot_r2(r2_df["r2"], "OOS R2 (Monthly)", config.output_dir / "oos_r2.png")

    print("Saved outputs to", config.output_dir)


if __name__ == "__main__":
    main()
