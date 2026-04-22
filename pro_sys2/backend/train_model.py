# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import StandardScaler
#
# from feature_engineering2 import compute_all_features
#
# # =========================
# # تحميل البيانات
# # =========================
# df = pd.read_csv("test2.csv")
#
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df.set_index('timestamp', inplace=True)
#
# # =========================
# # Feature Engineering + Target
# # =========================
# df = compute_all_features(
#     df,
#     is_new_row=False,
#     threshold=0.001,
#     horizon=3
# )
#
# print("📊 Target Distribution:")
# print(df['Target'].value_counts(normalize=True))
#
# # =========================
# # تجهيز البيانات
# # =========================
# X = df.drop(columns=['Target'])
# y = df['Target']
#
# # =========================
# # Scaling
# # =========================
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # =========================
# # تقسيم البيانات (Time Series)
# # =========================
# split = int(len(X_scaled) * 0.8)
#
# X_train, X_test = X_scaled[:split], X_scaled[split:]
# y_train, y_test = y[:split], y[split:]
#
# # =========================
# # بناء المودل (XGBoost)
# # =========================
# model = XGBClassifier(
#     n_estimators=500,
#     max_depth=6,
#     learning_rate=0.03,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     random_state=42,
#     eval_metric='logloss'
# )
#
# # =========================
# # التدريب
# # =========================
# model.fit(X_train, y_train)
#
# # =========================
# # التوقع (Probabilities)
# # =========================
# probs = model.predict_proba(X_test)
#
# sell_probs = probs[:, 0]
# buy_probs = probs[:, 1]
#
# preds = []
#
# for s, b in zip(sell_probs, buy_probs):
#     if b > 0.6:
#         preds.append(1)   # Buy
#     elif s > 0.55:
#         preds.append(0)   # Sell
#     else:
#         preds.append(1)   # fallback
#
# preds = pd.Series(preds)
#
# # =========================
# # التقييم
# # =========================
# print("\n📊 Classification Report:")
# print(classification_report(y_test, preds))


import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from feature_engineering2 import compute_all_features


def train_xgb_model(df, threshold=0.001, horizon=3):
    """
    تدريب مودل XGBoost باستخدام DataFrame جاهز

    Parameters:
    -----------
    df : DataFrame يحتوي:
        ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']

    Returns:
    --------
    preds : التوقعات على test
    model : المودل المدرب
    scaler : scaler المستخدم
    """

    # =========================
    # تجهيز البيانات
    # =========================
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # =========================
    # Feature Engineering
    # =========================
    df = compute_all_features(
        df,
        is_new_row=False,
        threshold=threshold,
        horizon=horizon
    )

    print("📊 Target Distribution:")
    print(df['Target'].value_counts(normalize=True))

    # =========================
    # Split X / y
    # =========================
    X = df.drop(columns=['Target'])
    y = df['Target']

    # =========================
    # Scaling
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # Time Split
    # =========================
    split = int(len(X_scaled) * 0.8)

    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    # =========================
    # Model
    # =========================
    model = XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric='logloss'
    )

    # =========================
    # Training
    # =========================
    model.fit(X_train, y_train)

    # =========================
    # Prediction
    # =========================
    probs = model.predict_proba(X_test)

    sell_probs = probs[:, 0]
    buy_probs = probs[:, 1]

    preds = []

    for s, b in zip(sell_probs, buy_probs):
        if b > 0.65:
            preds.append(1)
        elif s > 0.65:
            preds.append(0)
        else:
            # Use the higher probability when neither threshold is met
            preds.append(1 if b > s else 0)

    preds = pd.Series(preds, index=y_test.index)

    # =========================
    # Evaluation
    # =========================
    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds))

    return preds, model, scaler