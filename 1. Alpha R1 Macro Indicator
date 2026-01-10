"""
DISCLAIMER:
This code is provided for educational and research purposes only.
It does not constitute financial advice, investment advice,
trading advice, or a recommendation to buy or sell any security.

Agora Lycos Trading Lab makes no guarantees regarding accuracy,
performance, or profitability. Use at your own risk.
Past performance is not indicative of future results.

Alpha-R1 Macro Indicator (Lite)
- Pulls market proxies (SPY, VIX, HYG, IEF, TLT, UUP, USO)
- Computes trend/risk signals
- Classifies a daily regime + factor gating (active/inactive factor families)


Requirements:
  pip install pandas numpy yfinance

Notes:
- Needs internet access at runtime (yfinance pulls data from Yahoo Finance).
- This is a stable "macro/regime layer" you can later feed into a stock scanner.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Config
# -----------------------------
TICKERS = {
    "SPY": "SPY",        # US equities proxy
    "VIX": "^VIX",       # volatility proxy
    "HYG": "HYG",        # high yield credit proxy
    "IEF": "IEF",        # intermediate treasuries proxy
    "TLT": "TLT",        # long treasuries proxy
    "UUP": "UUP",        # USD proxy
    "USO": "USO",        # oil proxy
}

LOOKBACK_DAYS = 252  # ~1Y trading days
END = None           # default: today
START = None         # let yfinance decide based on period; we'll use "period"


# -----------------------------
# Utilities
# -----------------------------
def _safe_last(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def _slope(series: pd.Series, window: int) -> float:
    """
    Simple normalized slope via linear regression on last `window` points.
    Returns slope per day in "z-ish" space (scaled by std of y).
    """
    s = series.dropna().tail(window)
    if len(s) < max(10, window // 2):
        return float("nan")
    y = s.values.astype(float)
    x = np.arange(len(y), dtype=float)
    # linear regression slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return float("nan")
    m = np.sum((x - x_mean) * (y - y_mean)) / denom
    y_std = np.std(y) if np.std(y) > 1e-9 else 1.0
    return float(m / y_std)


def _zscore(series: pd.Series, window: int) -> float:
    s = series.dropna().tail(window)
    if len(s) < max(10, window // 2):
        return float("nan")
    mu = s.mean()
    sd = s.std()
    if sd is None or sd < 1e-9:
        return float("nan")
    return float((s.iloc[-1] - mu) / sd)


def _sigmoid(x: float) -> float:
    if math.isnan(x):
        return 0.5
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class MacroSignals:
    spy_trend: float
    vix_level_z: float
    vix_trend: float
    credit_risk_trend: float
    rates_risk_trend: float
    usd_trend: float
    oil_trend: float


# -----------------------------
# Data
# -----------------------------
def fetch_prices(tickers: Dict[str, str], period: str = "1y") -> pd.DataFrame:
    """
    Returns adjusted close prices as a DataFrame with columns = keys in `tickers`.
    """
    raw = yf.download(list(tickers.values()), period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("No data returned. Check internet access / Yahoo availability.")
    # yfinance returns multiindex columns for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw.rename(columns={"Close": list(tickers.values())[0]})[["Close"]].copy()

    # Map Yahoo symbols back to our friendly names
    inv = {v: k for k, v in tickers.items()}
    px = px.rename(columns=inv)

    # Ensure we have all columns (some may be missing)
    for k in tickers.keys():
        if k not in px.columns:
            px[k] = np.nan

    return px.sort_index()


# -----------------------------
# Signal engineering
# -----------------------------
def compute_macro_signals(px: pd.DataFrame) -> Tuple[MacroSignals, pd.DataFrame]:
    """
    Builds core macro signals + returns a diagnostic DataFrame.
    """
    # Core series
    spy = px["SPY"]
    vix = px["VIX"]
    hyg = px["HYG"]
    ief = px["IEF"]
    tlt = px["TLT"]
    uup = px["UUP"]
    uso = px["USO"]

    # Proxy spreads / risk measures:
    # 1) Credit risk appetite: HYG / IEF (risk-on credit tends to outperform safe-ish duration)
    credit_ratio = (hyg / ief).replace([np.inf, -np.inf], np.nan)

    # 2) Rates risk / duration stress proxy: IEF / TLT (rising rates often hurt long duration more)
    # If IEF/TLT trends up, long bonds underperform -> "rates risk" rising (simplified).
    rates_ratio = (ief / tlt).replace([np.inf, -np.inf], np.nan)

    # Compute signals
    sig = MacroSignals(
        spy_trend=_slope(np.log(spy), window=60),                # equity trend (2-3 months)
        vix_level_z=_zscore(vix, window=126),                    # vix relative level (~6 months)
        vix_trend=_slope(np.log(vix), window=20),                # vix trend (1 month)
        credit_risk_trend=_slope(np.log(credit_ratio), window=60),
        rates_risk_trend=_slope(np.log(rates_ratio), window=60),
        usd_trend=_slope(np.log(uup), window=60),
        oil_trend=_slope(np.log(uso), window=60),
    )

    diag = pd.DataFrame(
        {
            "spy_trend": [sig.spy_trend],
            "vix_level_z": [sig.vix_level_z],
            "vix_trend": [sig.vix_trend],
            "credit_risk_trend": [sig.credit_risk_trend],
            "rates_risk_trend": [sig.rates_risk_trend],
            "usd_trend": [sig.usd_trend],
            "oil_trend": [sig.oil_trend],
        },
        index=[px.index.max()],
    )
    return sig, diag


# -----------------------------
# Regime model + factor gating
# -----------------------------
def classify_regime(sig: MacroSignals) -> Tuple[str, float, Dict[str, float]]:
    """
    Returns (regime_label, confidence, component_scores).
    Confidence is derived from how decisive the composite score is.
    """
    # Risk-on score components (all mapped to 0..1 via sigmoid)
    # Interpretations:
    # - spy_trend up => risk-on
    # - vix_level_z high => risk-off (so we invert)
    # - vix_trend up => risk-off (invert)
    # - credit_risk_trend up => risk-on (credit appetite)
    # - rates_risk_trend up => can be risk-off for equities (invert lightly)
    # - usd_trend up => often tight financial conditions (invert lightly)
    # - oil_trend up => inflation risk (invert lightly)
    c_spy = _sigmoid(1.2 * (sig.spy_trend if not math.isnan(sig.spy_trend) else 0.0))
    c_vix_level = 1.0 - _sigmoid(0.9 * (sig.vix_level_z if not math.isnan(sig.vix_level_z) else 0.0))
    c_vix_trend = 1.0 - _sigmoid(1.0 * (sig.vix_trend if not math.isnan(sig.vix_trend) else 0.0))
    c_credit = _sigmoid(1.0 * (sig.credit_risk_trend if not math.isnan(sig.credit_risk_trend) else 0.0))
    c_rates = 1.0 - _sigmoid(0.6 * (sig.rates_risk_trend if not math.isnan(sig.rates_risk_trend) else 0.0))
    c_usd = 1.0 - _sigmoid(0.5 * (sig.usd_trend if not math.isnan(sig.usd_trend) else 0.0))
    c_oil = 1.0 - _sigmoid(0.5 * (sig.oil_trend if not math.isnan(sig.oil_trend) else 0.0))

    # Weighted composite (you can tune weights later)
    weights = {
        "spy": 0.25,
        "vix_level": 0.20,
        "vix_trend": 0.15,
        "credit": 0.20,
        "rates": 0.08,
        "usd": 0.06,
        "oil": 0.06,
    }

    comp = {
        "spy": c_spy,
        "vix_level": c_vix_level,
        "vix_trend": c_vix_trend,
        "credit": c_credit,
        "rates": c_rates,
        "usd": c_usd,
        "oil": c_oil,
    }

    risk_on_score = sum(weights[k] * comp[k] for k in weights.keys())

    # Regime thresholds
    if risk_on_score >= 0.62:
        regime = "Risk-On"
    elif risk_on_score <= 0.42:
        regime = "Risk-Off"
    else:
        regime = "Transition"

    # Confidence: distance from 0.52 midpoint, scaled to 0..1
    # more decisive score => higher confidence
    confidence = min(1.0, max(0.0, abs(risk_on_score - 0.52) / 0.30))
    return regime, confidence, {"risk_on_score": risk_on_score, **comp}


def gate_factors(regime: str, sig: MacroSignals) -> Tuple[List[str], List[str]]:
    """
    Factor families, not individual factors yet.
    We'll activate/deactivate families based on regime + a couple overrides.
    """
    # Base regime mapping
    if regime == "Risk-On":
        active = ["Momentum", "Growth", "Size", "Cyclical"]
        inactive = ["LowVol", "DefensiveYield", "DeepValue"]
    elif regime == "Risk-Off":
        active = ["Quality", "LowVol", "DefensiveYield"]
        inactive = ["Size", "HighBetaMomentum", "Cyclical"]
    else:  # Transition
        active = ["Quality", "Momentum"]  # barbell
        inactive = ["Size", "DeepValue"]  # usually the chop zone hurts these

    # Overrides (simple “shock” logic)
    # If oil is ripping up, tilt away from long-duration growth
    if not math.isnan(sig.oil_trend) and sig.oil_trend > 0.25:
        if "Growth" in active:
            active.remove("Growth")
        if "InflationHedge" not in active:
            active.append("InflationHedge")

    # If VIX is sharply rising, de-risk momentum
    if not math.isnan(sig.vix_trend) and sig.vix_trend > 0.30:
        if "Momentum" in active:
            active.remove("Momentum")
        if "LowVol" not in active:
            active.append("LowVol")

    # Deduplicate
    active = sorted(list(dict.fromkeys(active)))
    inactive = sorted(list(dict.fromkeys(inactive)))
    return active, inactive


# -----------------------------
# Runner
# -----------------------------
def run_alpha_r1_macro(period: str = "1y", save_csv: bool = True) -> Dict:
    px = fetch_prices(TICKERS, period=period)
    sig, diag = compute_macro_signals(px)

    regime, confidence, scores = classify_regime(sig)
    active, inactive = gate_factors(regime, sig)

    asof = str(px.index.max().date())

    result = {
        "asof": asof,
        "regime": regime,
        "confidence": round(confidence, 3),
        "active_factors": active,
        "inactive_factors": inactive,
        "scores": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()},
    }

    print("\n=== ALPHA-R1 MACRO INDICATOR (LITE) ===")
    print(f"As of:       {asof}")
    print(f"Regime:      {regime}")
    print(f"Confidence:  {result['confidence']}")
    print("\nActive factors:")
    for f in active:
        print(f"  ✔ {f}")
    print("\nInactive factors:")
    for f in inactive:
        print(f"  ✖ {f}")

    # Diagnostics
    print("\n--- Diagnostics (raw signal values) ---")
    print(diag.T.rename(columns={diag.index[0]: "value"}).to_string())

    print("\n--- Diagnostics (component scores 0..1) ---")
    score_df = pd.DataFrame({k: [v] for k, v in scores.items()}, index=[asof])
    print(score_df.T.rename(columns={asof: "value"}).to_string())

    if save_csv:
        out = pd.concat([diag, score_df], axis=1)
        out.insert(0, "regime", regime)
        out.insert(1, "confidence", result["confidence"])
        out.to_csv("alpha_r1_macro_output.csv", index=True)
        print("\nSaved: alpha_r1_macro_output.csv")

    return result


if __name__ == "__main__":
    run_alpha_r1_macro(period="1y", save_csv=True)
