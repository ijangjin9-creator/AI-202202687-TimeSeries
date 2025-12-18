import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    CSV 파일에서 데이터를 로드합니다.
    """
    df = pd.read_csv(file_path)
    return df

def load_bitcoin_data(start_date, end_date, ticker='BTC-USD'):
    """
    yfinance를 사용하여 비트코인 시계열 데이터를 로드합니다.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    # yfinance가 대문자로 컬럼명을 반환하므로 소문자로 변경
    df.columns = [col.lower() for col in df.columns]
    return df

def plot_close_price(df, title='Stock Close Price'):
    """
    종가 그래프를 그립니다.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(df['close'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()

def create_features(df):
    """
    간단한 기술적 분석 지표를 추가하는 함수 (기본 버전)
    """
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df