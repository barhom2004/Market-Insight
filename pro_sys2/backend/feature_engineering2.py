import pandas as pd
import numpy as np

def compute_all_features(df, is_new_row=False, threshold=0.001, horizon=3):
    """
    حساب جميع الفيتشرز الخاصة بالتداول.
    """

    if is_new_row:
        working_df = df.tail(250).copy()
    else:
        working_df = df.copy()

    # =========================
    # 🎯 Target
    # =========================
    if not is_new_row:
        future_return = (
            working_df['Close'].shift(-horizon) - working_df['Close']
        ) / working_df['Close']

        working_df['Target'] = np.where(
            future_return > threshold, 1,
            np.where(
                future_return < -threshold, 0,
                np.nan
            )
        )

    # =========================
    # EMA
    # =========================
    for span in [20, 50, 200]:
        ema_col = f"EMA{span}"

        working_df[ema_col] = working_df["Close"].ewm(
            span=span,
            adjust=False,
            min_periods=span
        ).mean()

        working_df[f"Dist_EMA{span}"] = (
            working_df["Close"] - working_df[ema_col]
        ) / working_df[ema_col]

    # =========================
    # RSI
    # =========================
    window = 14
    delta = working_df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    working_df["RSI"] = 100 - (100 / (1 + rs))

    # =========================
    # MACD
    # =========================
    ema12 = working_df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = working_df["Close"].ewm(span=26, adjust=False).mean()

    working_df["MACD"] = ema12 - ema26
    working_df["MACD_Signal"] = working_df["MACD"].ewm(span=9, adjust=False).mean()
    working_df["MACD_Hist"] = working_df["MACD"] - working_df["MACD_Signal"]

    # =========================
    # Stochastic
    # =========================
    low_min = working_df["Low"].rolling(14).min()
    high_max = working_df["High"].rolling(14).max()

    working_df["Stoch_K"] = 100 * (
        working_df["Close"] - low_min
    ) / (high_max - low_min + 1e-10)

    working_df["Stoch_D"] = working_df["Stoch_K"].rolling(3).mean()

    # =========================
    # ATR
    # =========================
    prev_close = working_df["Close"].shift(1)

    tr = pd.concat([
        working_df["High"] - working_df["Low"],
        (working_df["High"] - prev_close).abs(),
        (working_df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    working_df["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    working_df["ATR_Ratio"] = working_df["ATR"] / working_df["Close"]

    # =========================
    # Returns
    # =========================
    working_df["Return"] = np.log(
        working_df["Close"] / working_df["Close"].shift(1)
    )

    working_df["Down_Move"] = (
        working_df["Close"] < working_df["Close"].shift(1)
    ).astype(int)

    working_df["Lower_Shadow"] = (
        (working_df["Close"] - working_df["Low"]) /
        (working_df["High"] - working_df["Low"] + 1e-10)
    )

    # =========================
    # Time Features
    # =========================
    if isinstance(working_df.index, pd.DatetimeIndex):
        hour = working_df.index.hour
        day = working_df.index.dayofweek

        working_df["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
        working_df["Hour_cos"] = np.cos(2 * np.pi * hour / 24)

        working_df["Day_sin"] = np.sin(2 * np.pi * day / 7)
        working_df["Day_cos"] = np.cos(2 * np.pi * day / 7)

    # =========================
    # تنظيف البيانات
    # =========================
    if not is_new_row:
        working_df = working_df.iloc[:-horizon]
        return working_df.dropna()

    # تحديث صف واحد
    if is_new_row:
        idx = df.index[-1]

        for col in working_df.columns:
            if col not in ["Open", "High", "Low", "Close", "Volume"]:
                df.loc[idx, col] = working_df[col].iloc[-1]

        return df
