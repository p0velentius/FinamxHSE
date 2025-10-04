import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from pandas.tseries.offsets import BDay

import time
start_time_wall = time.time()

# best_alpha = 300

# --- 1. Загружаем препроцессор и модель ---
preprocessor = joblib.load('preprocessor.joblib')

# Выбираем нужный вариант
PER_H_ALPHA = True  # или False — в зависимости от того, какой режим ты сохранял

if PER_H_ALPHA:
    model_data = joblib.load('final_ridge_models_h.joblib')
    models_h = model_data["models_h"]
    best_alpha_h = model_data["best_alpha_h"]
    all_tickers = model_data["all_tickers"]
    tic2pos = model_data["tic2pos"]
    BASE_FEATURES = model_data["BASE_FEATURES"]
    K = model_data["K"]
    H = model_data["H"]
else:
    model_data = joblib.load('final_ridge_model.joblib')
    final_model = model_data["model"]
    best_alpha = model_data["best_alpha"]
    all_tickers = model_data["all_tickers"]
    tic2pos = model_data["tic2pos"]
    BASE_FEATURES = model_data["BASE_FEATURES"]
    K = model_data["K"]
    H = model_data["H"]

print("Модель и препроцессор успешно загружены")


# Загрузка новых данных
test_candles = pd.read_csv('candles_2.csv')

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет только информативные и некоррелированные технические фичи.
    Вход: df с колонками ['open', 'close', 'high', 'low', 'volume', 'begin', 'ticker']
    Выход: df с колонками: [
      'open', 'close', 'high', 'low', 'volume', 'begin', 'ticker',
      'log_return', 'log_return_lag1', 'log_return_lag2',
      'close_over_ema20', 'macd', 'macd_signal',
      'rolling_vol_10', 'atr_14',
      'candle_direction',
      'volume_ratio',
      'dow_sin', 'dow_cos'
    ]
    """
    df = df.copy()
    df['begin'] = pd.to_datetime(df['begin'])
    df = df.sort_values(['ticker', 'begin']).reset_index(drop=True)

    def compute_features(group):
        g = group.copy().reset_index(drop=True)

        # --- 1. Лог-доходности и лаги ---
        g['log_return'] = np.log(g['close'] / g['close'].shift(1))
        g['log_return_lag1'] = g['log_return'].shift(1)
        g['log_return_lag2'] = g['log_return'].shift(2)
        g['log_return_lag3'] = g['log_return'].shift(3)
        g['log_return_lag4'] = g['log_return'].shift(4)
        g['log_return_lag5'] = g['log_return'].shift(5)

        # --- 2. Доходности на окнах ---
        # 5-дневная доходность от t-10 до t-5
        g['log_return_window_5'] = np.log(g['close'].shift(5)) - np.log(g['close'].shift(10))

        # 10-дневная доходность от t-20 до t-10
        g['log_return_window_10'] = np.log(g['close'].shift(10)) - np.log(g['close'].shift(20))


        # --- 2. Нормированное отклонение от тренда ---
        ema_20 = g['close'].ewm(span=20, adjust=False).mean()
        g['close_over_ema20'] = g['close'] / ema_20

        # --- 3. MACD и сигнальная линия ---
        ema_12 = g['close'].ewm(span=12, adjust=False).mean()
        ema_26 = g['close'].ewm(span=26, adjust=False).mean()
        g['macd'] = ema_12 - ema_26
        g['macd_signal'] = g['macd'].ewm(span=9, adjust=False).mean()

        # --- 4. Волатильность (скользящая std лог-доходности) ---
        g['rolling_vol_10'] = g['log_return'].rolling(window=10, min_periods=2).std()

        # --- 5. ATR (Average True Range) ---
        tr1 = g['high'] - g['low']
        tr2 = abs(g['high'] - g['close'].shift(1))
        tr3 = abs(g['low'] - g['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g['atr_14'] = tr.rolling(window=14, min_periods=1).mean()

        # --- 6. Направление свечи (бинарное) ---
        g['candle_direction'] = (g['close'] > g['open']).astype(int)

        # --- 7. Относительный объём (аномалия) ---
        volume_ma_10 = g['volume'].rolling(window=10, min_periods=1).mean()
        g['volume_ratio'] = g['volume'] / volume_ma_10

        # --- 8. Циклическое кодирование дня недели ---
        dow = g['begin'].dt.dayofweek
        g['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        g['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        return g

    df_out = df.groupby('ticker', group_keys=False).apply(compute_features).reset_index(drop=True)
    return df_out

test_candles = add_technical_features(test_candles)

# -------------------
# Конфигурация
# -------------------
SEED = 0
np.random.seed(SEED)

TARGET_DAY = pd.Timestamp('2025-09-08')  # дата t0 (последний день в данных)
H = 20          # горизонт будущих дней
K = 20          # длина окна прошлого
VAL_Q = 0.8     # доля вал внутри "allowed" окон для подбора alpha
HALF_LIFE_BD = 60  # полупериод временного декай (в рабочих днях)
PER_H_ALPHA = True  # True = отдельный alpha и модель для каждого горизонта

# -------------------
# 0) ДАННЫЕ
# -------------------
df = test_candles.copy()
df['begin'] = pd.to_datetime(df['begin'])
df = df.sort_values(['ticker','begin']).reset_index(drop=True)

# цена для доходности
price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
df.loc[df[price_col] == 0, price_col] = np.nan
df[price_col] = df.groupby('ticker', group_keys=False)[price_col].apply(lambda s: s.ffill().bfill())

# однодневная форвард-доходность r_{t+1} = P_{t+1}/P_t - 1
df['target_return_1d'] = df.groupby('ticker')[price_col].shift(-1) / df[price_col] - 1

# признаки (подстройте под свои реальные колонки)
BASE_FEATURES = [
    'open','close','high','low','volume',
    'log_return','log_return_lag1','log_return_lag2','log_return_lag3','log_return_lag4','log_return_lag5',
    'log_return_window_5','log_return_window_10',
    'close_over_ema20','macd','macd_signal',
    'rolling_vol_10','atr_14','candle_direction','volume_ratio',
    'dow_sin','dow_cos',
]
RET_COL = 'target_return_1d'
for c in BASE_FEATURES:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# контроль: последняя дата по каждому тикеру
last_by_tic = df.groupby('ticker')['begin'].max()
if not (last_by_tic == TARGET_DAY).all():
    bad = last_by_tic[last_by_tic != TARGET_DAY]
    raise ValueError(f"Не у всех тикеров последняя дата = {TARGET_DAY.date()}. Несовпадения:\n{bad}")

# -------------------
# 1) ЭМБАРГО И «РАЗРЕШЁННОЕ» ПРОШЛОЕ
# -------------------
final_cutoff_t_end = TARGET_DAY - BDay(H)   # окна только с t_end ≤ t0 − H

# -------------------
# 2) ПРЕПРОЦЕССОР (fit только на прошлом: begin < t0 − H)
# -------------------
pre_fit_mask = df['begin'] < final_cutoff_t_end
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
num_pipe.fit(df.loc[pre_fit_mask, BASE_FEATURES])

X_scaled_all = pd.DataFrame(
    num_pipe.transform(df[BASE_FEATURES]),
    columns=BASE_FEATURES, index=df.index
)

# -------------------
# 3) ОКНА (K) + ТАРГЕТЫ (H)
# -------------------
def build_windowed_dataset(df_sorted, X_scaled, feature_cols, ret_col, K=20, H=20):
    X_list, Y_list, meta = [], [], []
    F = len(feature_cols)
    for tic, dft in df_sorted.groupby('ticker', sort=False):
        idx = dft.index.to_numpy()
        X_tic = X_scaled.loc[idx, feature_cols].to_numpy()
        r1_tic = df_sorted.loc[idx, ret_col].to_numpy()
        for tpos in range(K-1, len(idx)-H):
            X_win = X_tic[tpos-K+1:tpos+1, :]      # [K,F] (t-19..t)
            x_vec = X_win.reshape(K*F)
            y_vec = r1_tic[tpos+1:tpos+1+H]        # r_{t+1..t+20}
            if not (np.isfinite(x_vec).all() and np.isfinite(y_vec).all()):
                continue
            X_list.append(x_vec.astype(np.float32))
            Y_list.append(y_vec.astype(np.float32))
            meta.append((tic, df_sorted.loc[idx[tpos], 'begin']))
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    meta_df = pd.DataFrame(meta, columns=['ticker','t_end'])
    return X, Y, meta_df

X_all, Y_all, meta_all = build_windowed_dataset(df, X_scaled_all, BASE_FEATURES, RET_COL, K=K, H=H)

# ----- оставляем ТОЛЬКО «разрешённые» окна: t_end ≤ t0 − H -----
allowed_mask = meta_all['t_end'] <= final_cutoff_t_end
X_allowed, Y_allowed = X_all[allowed_mask], Y_all[allowed_mask]
meta_allowed = meta_all.loc[allowed_mask].reset_index(drop=True)

print("Allowed windows:",
      "X_allowed", X_allowed.shape, "Y_allowed", Y_allowed.shape)

# -------------------
# 4) One-Hot тикера
# -------------------
all_tickers = sorted(df['ticker'].unique().tolist())
tic2pos = {t:i for i,t in enumerate(all_tickers)}

def ohe_from_series(ticker_series: pd.Series) -> np.ndarray:
    M = np.zeros((len(ticker_series), len(all_tickers)), dtype=np.float32)
    ts = ticker_series.to_numpy()
    for i, t in enumerate(ts):
        M[i, tic2pos[t]] = 1.0
    return M

OHE_allowed = ohe_from_series(meta_allowed['ticker'])
X_allowed_ext = np.hstack([X_allowed, OHE_allowed])  # окно + OHE

# -------------------
# 5) Временной декай (sample weights) — без утечки
# -------------------
t_cut = final_cutoff_t_end.normalize()
delta_bd = np.array(
    [np.busday_count(d.date(), t_cut.date()) for d in meta_allowed['t_end']],
    dtype=np.int32
)
w_allowed = (0.5) ** (delta_bd / HALF_LIFE_BD)

# -------------------
# 8) SUBMISSION на t0: окно K, оканчивающееся t0, для каждого тикера
# -------------------
def make_window_for_date_ext(df_all, X_scaled_all, feature_cols, ticker, t_date, K=20):
    dft = df_all[df_all['ticker']==ticker].sort_values('begin')
    if dft['begin'].iloc[-1] != t_date:
        raise ValueError(f"{ticker}: последняя дата {dft['begin'].iloc[-1].date()} != {t_date.date()}")
    idx = dft.index.to_numpy()
    last_idx = idx[-K:]
    if len(last_idx) < K:
        raise ValueError(f"{ticker}: недостаточно строк для окна K={K}")
    X_win = X_scaled_all.loc[last_idx, feature_cols].to_numpy()
    x_vec = X_win.reshape(K*len(feature_cols)).astype(np.float32)
    # OHE тикера
    ohe = np.zeros(len(all_tickers), dtype=np.float32)
    ohe[tic2pos[ticker]] = 1.0
    return np.concatenate([x_vec, ohe], axis=0)  # (K*F + n_tickers,)

rows = []
for tic in all_tickers:
    x_vec_ext = make_window_for_date_ext(df, X_scaled_all, BASE_FEATURES, tic, TARGET_DAY, K=K)
    if PER_H_ALPHA:
        y_hat = np.array([models_h[h].predict(x_vec_ext.reshape(1,-1))[0] for h in range(H)], dtype=float)
    else:
        y_hat = final_model.predict(x_vec_ext.reshape(1,-1))[0]
    rows.append({"ticker": tic, **{f"p{i+1}": float(y_hat[i]) for i in range(H)}})

submission = pd.DataFrame(rows).sort_values('ticker').reset_index(drop=True)
# контроль качества файла
assert submission.shape[1] == (1 + H), f"submission columns mismatch: {submission.shape}"
assert submission.isna().sum().sum() == 0, "NaN в submission!"


def transform_row(row):
    # копируем, чтобы не портить исходные данные
    new_row = row.copy()
    for j in range(2, 21):
        new_row.iloc[j] = (new_row.iloc[j - 1] + 1) * (new_row.iloc[j] + 1) - 1
    return new_row


submission = submission.apply(transform_row, axis=1)

submission.to_csv("submission.csv", index=False)
print("submission.csv создан. Форма:", submission.shape)
print(submission.head())

end_time_wall = time.time()
print(f"Elapsed wall time (time.time()): {end_time_wall - start_time_wall} seconds")
