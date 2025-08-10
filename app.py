import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pandas_ta as ta
import plotly.graph_objects as go

try:
    from streamlit_autorefresh import st_autorefresh
    AUTORELOAD_AVAILABLE = True
except Exception:
    AUTORELOAD_AVAILABLE = False

# Configuración de símbolos para CoinGecko
SYMBOLS = {
    'Bitcoin': 'bitcoin',
    'Ethereum': 'ethereum',
    'XRP': 'ripple',
    'Tron': 'tron'
}

@st.cache_data(ttl=600)
def fetch_ohlcv_coingecko(coin_name: str, vs_currency='usd', days=30, interval='daily'):
    coin_id = SYMBOLS.get(coin_name)
    if coin_id is None:
        st.error(f"Moneda {coin_name} no soportada en CoinGecko.")
        return None

    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': interval
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        prices = data['prices']          # [[timestamp, price], ...]
        volumes = data['total_volumes']  # [[timestamp, volume], ...]
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
        df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
        df = pd.merge(df_prices, df_volumes, on='timestamp')
        # CoinGecko no da OHLC, solo precio puntual. Repetimos precio en open, high, low, close
        df['open'] = df['price']
        df['high'] = df['price']
        df['low'] = df['price']
        df['close'] = df['price']
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Error obteniendo datos de CoinGecko: {e}")
        return None

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # RSI 14
    out['rsi14'] = ta.rsi(out['close'], length=14)

    # SMA 20, 50, 200
    out['sma20'] = ta.sma(out['close'], length=20)
    out['sma50'] = ta.sma(out['close'], length=50)
    out['sma200'] = ta.sma(out['close'], length=200)

    # EMA 9, 21, 50
    out['ema9'] = ta.ema(out['close'], length=9)
    out['ema21'] = ta.ema(out['close'], length=21)
    out['ema50'] = ta.ema(out['close'], length=50)

    # MACD
    macd = ta.macd(out['close'])
    out['macd'] = macd['MACD_12_26_9']
    out['macd_signal'] = macd['MACDs_12_26_9']
    out['macd_hist'] = macd['MACDh_12_26_9']

    # Bollinger Bands
    bbands = ta.bbands(out['close'], length=20, std=2)
    out['bb_h'] = bbands['BBU_20_2.0']
    out['bb_l'] = bbands['BBL_20_2.0']

    # ATR 14
    out['atr14'] = ta.atr(out['high'], out['low'], out['close'], length=14)

    # OBV
    out['obv'] = ta.obv(out['close'], out['volume'])

    # VWAP aproximado
    tp = (out['high'] + out['low'] + out['close']) / 3
    out['vwap'] = (tp * out['volume']).cumsum() / out['volume'].cumsum()

    return out

def generate_signal(latest: pd.Series) -> str:
    close = latest['close']
    ema21 = latest['ema21']
    ema50 = latest['ema50']
    macd_hist = latest['macd_hist']
    rsi = latest['rsi14']
    vol = latest['volume']

    bull = (close > ema21) and (ema21 > ema50) and (macd_hist > 0) and (rsi < 75) and (vol > 0)
    bear = (close < ema21) and (ema21 < ema50) and (macd_hist < 0) and (rsi > 25)

    if bull and rsi < 60:
        return 'BUY — tendencia alcista con momentum. Confirmar con volumen y gestión de riesgo.'
    if bull and rsi >= 60:
        return 'CAUTION — sobrecompra cercana (RSI alto). Esperar retroceso o reducir tamaño.'
    if bear:
        return 'SELL — tendencia bajista. Evitar entradas largas o preparar stop.'
    return 'HOLD — sin señal clara. Esperar confirmación (price/volume).'

def daily_highlights(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    last_24h = df.last('1D')
    change_24h = (latest['close'] / last_24h['close'].iloc[0] - 1) * 100 if len(last_24h) > 1 else np.nan
    vol_spike = latest['volume'] > 2 * (df['volume'].rolling(window=20, min_periods=1).mean().iloc[-1])
    highlight = {
        'price': float(latest['close']),
        'change_24h_pct': float(change_24h) if not np.isnan(change_24h) else None,
        'volume': float(latest['volume']),
        'volume_spike': bool(vol_spike),
        'rsi': float(latest['rsi14'])
    }
    return highlight

st.set_page_config(layout='wide', page_title='Crypto Dashboard (CoinGecko)')
st.title('Crypto Dashboard — usando CoinGecko (BTC, ETH, XRP, TRX)')

with st.sidebar:
    st.header('Opciones')
    coin_name = st.selectbox('Selecciona moneda', list(SYMBOLS.keys()), index=0)
    days = st.slider('Últimos días a analizar', min_value=7, max_value=90, value=30, step=1)
    interval = st.selectbox('Intervalo de velas', ['daily', 'hourly'], index=0)
    autoreload = st.checkbox('Auto-refresh cada X segundos', value=True if AUTORELOAD_AVAILABLE else False)
    if AUTORELOAD_AVAILABLE and autoreload:
        interval_seconds = st.number_input('Intervalo refresco (s)', min_value=10, max_value=600, value=60, step=10)
    else:
        interval_seconds = None
    st.markdown('---')
    st.markdown('Datos vía CoinGecko API pública, sin bloqueo geográfico.')

if AUTORELOAD_AVAILABLE and autoreload and interval_seconds:
    st_autorefresh(interval=interval_seconds * 1000, limit=None, key=f"autorefresh_{coin_name}")

with st.spinner(f'Obteniendo datos {coin_name} ({interval})...'):
    df = fetch_ohlcv_coingecko(coin_name, vs_currency='usd', days=days, interval=interval)

if df is None or df.empty:
    st.error('No se pudieron obtener datos. Revisa tu conexión o intenta más tarde.')
    st.stop()

df_ind = compute_indicators(df)
latest = df_ind.iloc[-1]
signal = generate_signal(latest)
highlights = daily_highlights(df_ind)

col1, col2 = st.columns([3,1])

with col1:
    st.subheader(f'{coin_name} — Intervalo {interval} — Últimos {days} días')
    fig = go.Figure(data=[go.Candlestick(
        x=df_ind.index,
        open=df_ind['open'],
        high=df_ind['high'],
        low=df_ind['low'],
        close=df_ind['close'],
        name='Velas'
    )])
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['sma20'], mode='lines', name='SMA20'))
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['ema21'], mode='lines', name='EMA21'))
    fig.update_layout(height=600, margin={'t':30, 'b':10})
    st.plotly_chart(fig, use_container_width=True)

    st.line_chart(df_ind[['volume', 'obv']].tail(200))

with col2:
    st.metric('Último precio', f"{latest['close']:.6f}")
    st.metric('RSI (14)', f"{latest['rsi14']:.2f}")
    st.metric('MACD hist', f"{latest['macd_hist']:.6f}")

    st.markdown('### Señal compuesta')
    st.info(signal)

    st.markdown('### Highlights (última vela)')
    st.write(highlights)

st.markdown('---')
st.header('Recomendaciones operativas (reglas simples — educativas)')
st.markdown('''
- BUY si: precio > EMA21, EMA21 > EMA50, MACD_hist > 0, RSI < 70 y volumen reciente > promedio.
- SELL si: precio < EMA21, EMA21 < EMA50 y MACD_hist < 0.
- CAUTION si: RSI alto (> 70) o volumen anómalo.
Estas reglas son heurísticas, siempre gestiona el riesgo y usa stop-loss.
''')

st.markdown('---')
st.header('Próximas mejoras posibles')
st.markdown('''
- Añadir análisis de derivados (open interest, funding rate) usando otras APIs.
- Incorporar métricas on-chain (Etherscan, TronGrid, XRPL).
- Alertas automáticas (Telegram, email) basadas en señales.
- Backtesting de reglas.
''')
