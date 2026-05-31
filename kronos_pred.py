#!/usr/bin/env python3
"""
Use local Kronos-style PyTorch model to predict next K-line from qlib data on CPU.

Example:
python kronos_qlib_predict.py \
  --provider-uri ~/.qlib/qlib_data/cn_data \
  --instrument sh600519 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --model-path ./kronos_model.pt \
  --window 64 \
  --horizon 1 \
  --out ./pred.csv
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kronos K-line prediction with qlib data (CPU).")
    parser.add_argument("--provider-uri", default="/kronos_demo/qlib_data", help="qlib data directory, e.g. ~/.qlib/qlib_data/cn_data")
    parser.add_argument("--region", default="cn", choices=["cn", "us"], help="qlib region")
    parser.add_argument("--instrument", required=True, help="single instrument, e.g. sh600519")
    parser.add_argument("--start", required=True, help="start date, e.g. 2023-01-01")
    parser.add_argument("--end", required=True, help="end date, e.g. 2024-12-31")
    parser.add_argument("--model-path", default="/kronos_demo/model", help="local torch model path (.pt/.pth)")
    parser.add_argument(
        "--tokenizer-path",
        default="/tokenizer",
        help="Kronos tokenizer local path or Hugging Face repo id",
    )
    parser.add_argument("--window", type=int, default=64, help="input sequence length")
    parser.add_argument("--horizon", type=int, default=10, help="prediction horizon (next N bars)")
    parser.add_argument("--batch-size", type=int, default=128, help="inference batch size")
    parser.add_argument("--seed", type=int, default=42, help="global random seed for reproducibility")
    parser.add_argument("--out", default="kronos_pred.csv", help="output csv path")
    parser.add_argument("--chart-out", default="kronos_pred.png", help="output candlestick chart path")
    parser.add_argument("--tune", action="store_true", help="run rolling backtest grid search instead of single forecast")
    parser.add_argument("--tune-out", default="kronos_tune_scores.csv", help="grid search result csv path")
    parser.add_argument("--tune-stride", type=int, default=5, help="rolling backtest step size in bars")
    parser.add_argument("--tune-max-windows", type=int, default=120, help="max rolling windows to evaluate (latest N)")
    parser.add_argument("--grid-window", default="64,128,256", help="comma-separated window candidates")
    parser.add_argument("--grid-temp", default="1.0,0.9,0.7", help="comma-separated temperature candidates")
    parser.add_argument("--grid-top-p", default="0.95,0.9,0.8", help="comma-separated top-p candidates")
    parser.add_argument("--grid-sample-count", default="1,5", help="comma-separated sample_count candidates")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_float_list(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def init_qlib(provider_uri: str, region: str) -> None:
    try:
        import qlib
        from qlib.config import REG_CN, REG_US
    except Exception as e:
        raise RuntimeError(
            "Cannot import Microsoft pyqlib. Please install with: "
            "`python -m pip install pyqlib`"
        ) from e

    if not hasattr(qlib, "init"):
        mod_file = getattr(qlib, "__file__", "unknown")
        raise RuntimeError(
            "Imported `qlib` package is not Microsoft `pyqlib` "
            f"(loaded from: {mod_file}).\n"
            "Fix:\n"
            "1) python -m pip uninstall -y qlib\n"
            "2) python -m pip install -U pyqlib\n"
            "3) rerun this script in the same python env"
        )

    reg_map = {"cn": REG_CN, "us": REG_US}
    qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=reg_map[region])


def load_ohlcv(instrument: str, start: str, end: str) -> pd.DataFrame:
    from qlib.data import D

    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]
    names = ["open", "high", "low", "close", "volume", "factor"]
    df = D.features([instrument], fields, start_time=start, end_time=end, freq="day")
    if df.empty:
        raise ValueError("No qlib data found. Check instrument/date/provider-uri.")
    df = df.reset_index().rename(columns={"datetime": "date", "instrument": "symbol"})
    df = df[["date", "symbol"] + fields].rename(columns=dict(zip(fields, names)))
    df = df.sort_values("date").dropna().reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Too few rows after dropna; need more history.")
    if "factor" not in df.columns:
        df["factor"] = 1.0
    df["factor"] = df["factor"].replace(0, np.nan).fillna(method="ffill").fillna(1.0)
    return df


@dataclass
class NormState:
    mu: np.ndarray
    sigma: np.ndarray


def zscore_fit(x: np.ndarray) -> NormState:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return NormState(mu=mu, sigma=sigma)


def zscore_apply(x: np.ndarray, state: NormState) -> np.ndarray:
    return (x - state.mu) / state.sigma


def build_windows(
    feat: np.ndarray,
    dates: Sequence[pd.Timestamp],
    window: int,
    horizon: int,
) -> tuple[np.ndarray, List[pd.Timestamp]]:
    xs: List[np.ndarray] = []
    pred_dates: List[pd.Timestamp] = []
    max_i = len(feat) - window - horizon + 1
    for i in range(max_i):
        xs.append(feat[i : i + window])
        pred_dates.append(dates[i + window + horizon - 1])
    if not xs:
        raise ValueError("Not enough rows for requested window+horizon.")
    return np.stack(xs, axis=0), pred_dates


def is_kronos_dir(model_path: str) -> bool:
    model_file = Path(model_path).expanduser()
    return model_file.is_dir() and (model_file / "config.json").exists() and any(model_file.glob("*.safetensors"))


def load_kronos_components(model_path: str, tokenizer_path: str):
    model_file = Path(model_path).expanduser()
    if not is_kronos_dir(str(model_file)):
        raise FileNotFoundError(
            f"Kronos model directory is incomplete: {model_file}\n"
            "Expected at least `config.json` and `*.safetensors`."
        )

    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    except Exception as e:
        raise RuntimeError(
            "Failed to import `Kronos`, `KronosTokenizer`, or `KronosPredictor` from `model`.\n"
            "Please install or clone the official Kronos project so that\n"
            "`from model import Kronos, KronosTokenizer, KronosPredictor` works.\n"
            "Reference: https://github.com/shiyu-coder/Kronos"
        ) from e

    print(f"[Info] Loading official Kronos model from directory: {model_file}")
    model = Kronos.from_pretrained(str(model_file))
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    return model, tokenizer, predictor


def load_local_model(model_path: str) -> torch.nn.Module:
    model_file = Path(model_path).expanduser()
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_file}\n"
            "Please pass an existing file (.pt/.pth/.ckpt/.bin) or a directory containing one."
        )

    # 1) Kronos official Hugging Face/local checkpoint.
    if model_file.is_dir():
        config_json = model_file / "config.json"
        safetensors = list(model_file.glob("*.safetensors"))
        if config_json.exists() and safetensors:
            try:
                from model import Kronos
            except Exception as e:
                raise RuntimeError(
                    "Detected Kronos checkpoint directory, but failed to import "
                    "`Kronos` from `model`.\n"
                    "This model does not load with `transformers.AutoModel`.\n"
                    "Please install or place the official Kronos source code in your environment,\n"
                    "so that `from model import Kronos` works.\n"
                    "Reference: https://github.com/shiyu-coder/Kronos"
                ) from e

            print(f"[Info] Loading official Kronos model from directory: {model_file}")
            model = Kronos.from_pretrained(str(model_file))
            model.to("cpu").eval()
            return model

        # 2) Fallback: plain torch checkpoint files in directory
        candidates: List[Path] = []
        preferred_names = [
            "model.pt",
            "model.pth",
            "checkpoint.pt",
            "checkpoint.pth",
            "pytorch_model.bin",
        ]
        for name in preferred_names:
            p = model_file / name
            if p.exists():
                candidates.append(p)
        if not candidates:
            for ext in ("*.pt", "*.pth", "*.ckpt", "*.bin"):
                candidates.extend(sorted(model_file.glob(ext)))
        if not candidates:
            raise FileNotFoundError(
                f"No model checkpoint file found under directory: {model_file}\n"
                "Expected one of: *.pt, *.pth, *.ckpt, *.bin or Kronos `model.safetensors` with config.json"
            )
        model_file = candidates[0]
        print(f"[Info] Using model checkpoint: {model_file}")

    # 3) Direct file load (PyTorch checkpoint)
    obj = torch.load(str(model_file), map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        raise RuntimeError(
            "Checkpoint contains only state_dict. "
            "Please save/load with full model object or implement your model class in this script."
        )
    else:
        raise TypeError(f"Unsupported model object type: {type(obj)}")
    model.to("cpu").eval()
    return model


def make_future_dates(last_date: pd.Timestamp, horizon: int) -> pd.Series:
    future = pd.date_range(start=last_date, periods=horizon + 1, freq="B")[1:]
    return pd.Series(future)


def unpack_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        return output[0]
    if isinstance(output, dict):
        for key in ("pred", "prediction", "y_hat", "logits", "output"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    raise TypeError(f"Cannot parse model output type: {type(output)}")


def predict_batches(model: torch.nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).float().to("cpu")
            raw = model(xb)
            y = unpack_output(raw).detach().cpu().numpy()
            outs.append(y)
    return np.concatenate(outs, axis=0)


def to_1step_ohlcv(pred: np.ndarray, last_window: np.ndarray, state: NormState) -> np.ndarray:
    """
    Convert model output to one-step OHLCV prediction in original scale.
    Supported output shape examples:
      - [N, 5]              -> direct one-step OHLCV
      - [N, 1, 5]           -> take step 0
      - [N, H, 5]           -> take last horizon step
      - [N, window, 5]      -> take last timestep
    """
    y = pred
    if y.ndim == 3:
        y = y[:, -1, :]
    elif y.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported pred ndim={y.ndim}, shape={y.shape}")

    if y.shape[1] != 5:
        # If model only predicts close, keep other fields as last observed.
        if y.shape[1] == 1:
            out = last_window[:, -1, :].copy()
            out[:, 3] = y[:, 0]
            y = out
        else:
            raise ValueError(f"Unsupported pred feature size={y.shape[1]}, expected 1 or 5.")

    return y * state.sigma + state.mu


def predict_with_kronos(
    df: pd.DataFrame,
    predictor,
    window: int,
    horizon: int,
    t: float = 0.7,
    top_p: float = 0.8,
    sample_count: int = 5,
) -> pd.DataFrame:
    x_df = df.tail(window).copy()
    x_timestamp = pd.to_datetime(x_df["date"])
    y_timestamp = make_future_dates(x_timestamp.iloc[-1], horizon)

    input_cols = ["open", "high", "low", "close"]
    if "volume" in x_df.columns:
        input_cols.append("volume")

    pred_df = predictor.predict(
        df=x_df[input_cols],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=horizon,
        T=t,
        top_p=top_p,
        sample_count=sample_count,
    )
    pred_df = pred_df.reset_index(drop=True)
    pred_df.insert(0, "date", y_timestamp.values)
    return pred_df


def rolling_kronos_backtest(
    df: pd.DataFrame,
    restored_df: pd.DataFrame,
    predictor,
    window: int,
    horizon: int,
    t: float,
    top_p: float,
    sample_count: int,
    stride: int,
    max_windows: int,
    seed: int,
) -> dict:
    eval_points = list(range(window, len(df) - horizon + 1, max(1, stride)))
    if max_windows > 0 and len(eval_points) > max_windows:
        eval_points = eval_points[-max_windows:]
    if not eval_points:
        raise ValueError("No rolling windows available for current window/horizon.")

    abs_err: List[float] = []
    sq_err: List[float] = []
    ape: List[float] = []

    for end_idx in eval_points:
        set_global_seed(seed)
        x_df = df.iloc[end_idx - window : end_idx].copy()
        y_df = df.iloc[end_idx : end_idx + horizon].copy()
        y_ts = pd.to_datetime(y_df["date"])
        x_ts = pd.to_datetime(x_df["date"])
        input_cols = ["open", "high", "low", "close"]
        if "volume" in x_df.columns:
            input_cols.append("volume")

        pred = predictor.predict(
            df=x_df[input_cols],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=horizon,
            T=t,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        ).reset_index(drop=True)
        pred = normalize_prediction_columns(pred)
        pred = apply_factor_to_prediction_prices(pred, float(x_df["factor"].iloc[-1]))
        actual = restored_df.iloc[end_idx : end_idx + horizon].reset_index(drop=True)

        y_pred = pred["pred_close"].values.astype(float)
        y_true = actual["close"].values.astype(float)
        err = y_pred - y_true
        abs_err.extend(np.abs(err).tolist())
        sq_err.extend((err ** 2).tolist())
        denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
        ape.extend((np.abs(err) / denom).tolist())

    mae = float(np.nanmean(abs_err))
    rmse = float(np.sqrt(np.nanmean(sq_err)))
    mape = float(np.nanmean(ape) * 100.0)
    return {
        "window": window,
        "T": t,
        "top_p": top_p,
        "sample_count": sample_count,
        "mae_close": mae,
        "rmse_close": rmse,
        "mape_close_pct": mape,
        "eval_windows": len(eval_points),
    }


def run_kronos_grid_search(df: pd.DataFrame, restored_df: pd.DataFrame, predictor, args: argparse.Namespace) -> pd.DataFrame:
    windows = parse_int_list(args.grid_window)
    temps = parse_float_list(args.grid_temp)
    top_ps = parse_float_list(args.grid_top_p)
    sample_counts = parse_int_list(args.grid_sample_count)
    rows: List[dict] = []
    for w in windows:
        for t in temps:
            for p in top_ps:
                for sc in sample_counts:
                    try:
                        row = rolling_kronos_backtest(
                            df=df,
                            restored_df=restored_df,
                            predictor=predictor,
                            window=w,
                            horizon=args.horizon,
                            t=t,
                            top_p=p,
                            sample_count=sc,
                            stride=args.tune_stride,
                            max_windows=args.tune_max_windows,
                            seed=args.seed,
                        )
                        row["status"] = "ok"
                    except Exception as e:
                        row = {
                            "window": w,
                            "T": t,
                            "top_p": p,
                            "sample_count": sc,
                            "mae_close": np.nan,
                            "rmse_close": np.nan,
                            "mape_close_pct": np.nan,
                            "eval_windows": 0,
                            "status": f"error: {e}",
                        }
                    rows.append(row)
    score_df = pd.DataFrame(rows)
    score_df = score_df.sort_values(["rmse_close", "mae_close"], na_position="last").reset_index(drop=True)
    return score_df


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in {"open", "high", "low", "close", "volume"}:
            rename_map[col] = f"pred_{lower}"
    out = df.rename(columns=rename_map).copy()
    required = ["pred_open", "pred_high", "pred_low", "pred_close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Prediction result missing required columns: {missing}")
    return out


def render_candlestick_chart(history_df: pd.DataFrame, pred_df: pd.DataFrame, chart_path: str, instrument: str) -> None:
    history = history_df[["date", "open", "high", "low", "close"]].copy()
    history["type"] = "history"

    pred = pred_df[["date", "pred_open", "pred_high", "pred_low", "pred_close"]].copy()
    pred = pred.rename(
        columns={
            "pred_open": "open",
            "pred_high": "high",
            "pred_low": "low",
            "pred_close": "close",
        }
    )
    pred["type"] = "prediction"

    plot_df = pd.concat([history, pred], ignore_index=True)
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.6

    for idx, row in plot_df.reset_index(drop=True).iterrows():
        up = row["close"] >= row["open"]
        if row["type"] == "prediction":
            color = "#1f77b4" if up else "#9467bd"
            alpha = 0.85
        else:
            color = "#d62728" if up else "#2ca02c"
            alpha = 0.65

        ax.vlines(idx, row["low"], row["high"], color=color, linewidth=1.2, alpha=alpha)
        body_low = min(row["open"], row["close"])
        body_height = max(abs(row["close"] - row["open"]), 1e-8)
        rect = plt.Rectangle(
            (idx - width / 2, body_low),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)

    split_idx = len(history) - 0.5
    ax.axvline(split_idx, color="gray", linestyle="--", linewidth=1.0)
    ax.text(split_idx + 0.1, plot_df["high"].max(), "prediction starts", fontsize=9, color="gray")

    ax.set_title(f"{instrument} future {len(pred)} trading days candlestick forecast")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["date"].dt.strftime("%Y-%m-%d"), rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close(fig)


def apply_factor_to_prediction_prices(pred_df: pd.DataFrame, factor_value: float) -> pd.DataFrame:
    out = pred_df.copy()
    if factor_value == 0:
        factor_value = 1.0
    for col in ("pred_open", "pred_high", "pred_low", "pred_close"):
        if col in out.columns:
            out[col] = out[col] / factor_value
    out["used_factor"] = factor_value
    return out


def restore_history_prices_with_factor(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] / out["factor"]
    return out


def load_actual_ohlc_for_dates(instrument: str, dates: pd.Series) -> pd.DataFrame:
    from qlib.data import D

    if dates.empty:
        return pd.DataFrame(columns=["date", "actual_open", "actual_high", "actual_low", "actual_close"])

    dt = pd.to_datetime(dates).dropna().sort_values().unique()
    if len(dt) == 0:
        return pd.DataFrame(columns=["date", "actual_open", "actual_high", "actual_low", "actual_close"])

    start_time = pd.Timestamp(dt.min()).strftime("%Y-%m-%d")
    end_time = pd.Timestamp(dt.max()).strftime("%Y-%m-%d")
    fields = ["$open", "$high", "$low", "$close", "$factor"]
    names = ["open", "high", "low", "close", "factor"]
    raw = D.features([instrument], fields, start_time=start_time, end_time=end_time, freq="day")
    if raw.empty:
        return pd.DataFrame(columns=["date", "actual_open", "actual_high", "actual_low", "actual_close"])

    act = raw.reset_index().rename(columns={"datetime": "date"})
    act = act[["date"] + fields].rename(columns=dict(zip(fields, names)))
    act["date"] = pd.to_datetime(act["date"])
    act["factor"] = act["factor"].replace(0, np.nan).ffill().fillna(1.0)
    for c in ("open", "high", "low", "close"):
        act[c] = act[c] / act["factor"]

    target = pd.DataFrame({"date": pd.to_datetime(dt)})
    act = target.merge(act[["date", "open", "high", "low", "close"]], on="date", how="left")
    act = act.rename(
        columns={
            "open": "actual_open",
            "high": "actual_high",
            "low": "actual_low",
            "close": "actual_close",
        }
    )
    return act


def _draw_candles(ax, df: pd.DataFrame, up_color: str, down_color: str, alpha: float = 0.8) -> None:
    width = 0.6
    for idx, row in df.reset_index(drop=True).iterrows():
        up = row["close"] >= row["open"]
        color = up_color if up else down_color
        ax.vlines(idx, row["low"], row["high"], color=color, linewidth=1.2, alpha=alpha)
        body_low = min(row["open"], row["close"])
        body_height = max(abs(row["close"] - row["open"]), 1e-8)
        ax.add_patch(
            plt.Rectangle((idx - width / 2, body_low), width, body_height, facecolor=color, edgecolor=color, alpha=alpha)
        )


def render_unified_chart(
    history_df: pd.DataFrame, pred_df: pd.DataFrame, actual_df: pd.DataFrame, chart_path: str, instrument: str
) -> bool:
    history = history_df[["date", "open", "high", "low", "close"]].copy()
    history["date"] = pd.to_datetime(history["date"])
    pred = pred_df[["date", "pred_open", "pred_high", "pred_low", "pred_close"]].rename(
        columns={"pred_open": "open", "pred_high": "high", "pred_low": "low", "pred_close": "close"}
    )
    pred["date"] = pd.to_datetime(pred["date"])
    top_df = pd.concat([history, pred], ignore_index=True)

    merged = pred_df.merge(actual_df, on="date", how="left")
    merged = merged.dropna(subset=["actual_open", "actual_high", "actual_low", "actual_close"]).copy()
    has_actual = not merged.empty

    if has_actual:
        actual = merged[["date", "actual_open", "actual_high", "actual_low", "actual_close"]].rename(
            columns={"actual_open": "open", "actual_high": "high", "actual_low": "low", "actual_close": "close"}
        )
        actual["date"] = pd.to_datetime(actual["date"])
        bottom_df = pd.concat([history, actual], ignore_index=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
        axes = [ax1]

    _draw_candles(axes[0], top_df, up_color="#1f77b4", down_color="#9467bd", alpha=0.85)
    split_top = len(history) - 0.5
    axes[0].axvline(split_top, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"{instrument} Predicted Candles (history + forecast)")
    axes[0].set_ylabel("Price")
    axes[0].set_xticks(range(len(top_df)))
    axes[0].set_xticklabels(top_df["date"].dt.strftime("%Y-%m-%d"), rotation=45, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)

    if has_actual:
        _draw_candles(axes[1], bottom_df, up_color="#d62728", down_color="#2ca02c", alpha=0.8)
        split_bottom = len(history) - 0.5
        axes[1].axvline(split_bottom, color="gray", linestyle="--", linewidth=1.0)
        axes[1].set_title(f"{instrument} Actual Candles (history + realized)")
        axes[1].set_ylabel("Price")
        axes[1].set_xlabel("Date")
        axes[1].set_xticks(range(len(bottom_df)))
        axes[1].set_xticklabels(bottom_df["date"].dt.strftime("%Y-%m-%d"), rotation=45, ha="right")
        axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    else:
        axes[0].set_xlabel("Date")

    fig.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close(fig)
    return has_actual


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    init_qlib(provider_uri=args.provider_uri, region=args.region)
    df = load_ohlcv(args.instrument, args.start, args.end)
    last_factor = float(df["factor"].iloc[-1]) if "factor" in df.columns else 1.0
    restored_df = restore_history_prices_with_factor(df)
    chart_history = restored_df.tail(min(args.window, 60)).copy()

    if is_kronos_dir(args.model_path):
        _, _, predictor = load_kronos_components(args.model_path, args.tokenizer_path)
        if args.tune:
            score_df = run_kronos_grid_search(df, restored_df, predictor, args)
            tune_out = Path(args.tune_out).expanduser()
            score_df.to_csv(tune_out, index=False)
            print(f"Saved tuning results: {tune_out}")
            ok = score_df[score_df["status"] == "ok"]
            if ok.empty:
                print("No valid parameter set found. Check tuning ranges and data length.")
            else:
                best = ok.iloc[0]
                print("Best parameters by RMSE(close):")
                print(best[["window", "T", "top_p", "sample_count", "mae_close", "rmse_close", "mape_close_pct", "eval_windows"]].to_string())
            return

        out_df = predict_with_kronos(df, predictor, args.window, args.horizon)
        out_df = normalize_prediction_columns(out_df)
        out_df = apply_factor_to_prediction_prices(out_df, last_factor)
        out_df.insert(1, "symbol", args.instrument)
        out_path = Path(args.out).expanduser()
        chart_path = Path(args.chart_out).expanduser()
        out_df.to_csv(out_path, index=False)
        actual_df = load_actual_ohlc_for_dates(args.instrument, out_df["date"])
        has_compare = render_unified_chart(chart_history, out_df, actual_df, str(chart_path), args.instrument)
        print(f"Saved predictions: {out_path}")
        print(f"Saved chart: {chart_path}")
        if not has_compare:
            print("No realized future candles found; chart contains prediction panel only.")
        print(out_df.tail(5).to_string(index=False))
        return

    feat = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
    norm_state = zscore_fit(feat)
    feat_z = zscore_apply(feat, norm_state).astype(np.float32)
    x, pred_dates = build_windows(
        feat=feat_z,
        dates=pd.to_datetime(df["date"]).tolist(),
        window=args.window,
        horizon=args.horizon,
    )

    model = load_local_model(args.model_path)
    pred_norm = predict_batches(model, x, args.batch_size)
    pred_raw = to_1step_ohlcv(pred_norm, x, norm_state)

    out_df = pd.DataFrame(
        {
            "date": pred_dates,
            "symbol": args.instrument,
            "pred_open": pred_raw[:, 0],
            "pred_high": pred_raw[:, 1],
            "pred_low": pred_raw[:, 2],
            "pred_close": pred_raw[:, 3],
            "pred_volume": pred_raw[:, 4],
        }
    )
    out_df = apply_factor_to_prediction_prices(out_df, last_factor)

    out_path = Path(args.out).expanduser()
    chart_path = Path(args.chart_out).expanduser()
    out_df.to_csv(out_path, index=False)
    out_tail = out_df.tail(args.horizon).copy()
    actual_df = load_actual_ohlc_for_dates(args.instrument, out_tail["date"])
    has_compare = render_unified_chart(chart_history, out_tail, actual_df, str(chart_path), args.instrument)
    print(f"Saved predictions: {out_path}")
    print(f"Saved chart: {chart_path}")
    if not has_compare:
        print("No realized future candles found; chart contains prediction panel only.")
    print(out_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
