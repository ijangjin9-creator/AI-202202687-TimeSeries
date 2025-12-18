import torch
import torch.nn as nn
from datetime import datetime
import ta
import pandas as pd
import numpy as np
from utils import load_bitcoin_data, plot_close_price

print("🚀 Step 1: 데이터 로딩 및 특성 공학 시작...")
# 1-1. yfinance를 통해 실제 비트코인 데이터 불러오기
end_date = datetime.now().strftime('%Y-%m-%d')
btc_raw = load_bitcoin_data(start_date='2018-01-01', end_date=end_date)
# plot_close_price call disabled for non-interactive run

# 1-2. 고급 기술적 분석 지표 추가 함수 정의
def create_features_enhanced(df_orig):
    df = df_orig.copy()
    # 볼린저 밴드
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_mid'] = indicator_bb.bollinger_mavg()
    df['bb_high'] = indicator_bb.bollinger_hband()
    df['bb_low'] = indicator_bb.bollinger_lband()
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    
    # Target 생성 (다음날 종가가 오늘보다 높으면 1, 아니면 0)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 결측치 제거 및 최종 데이터프레임 반환
    df.dropna(inplace=True)
    return df

# 1-3. 함수를 적용하여 최종 피처 데이터셋 생성
btc_features = create_features_enhanced(btc_raw)
print("✅ 특성 공학 완료. 생성된 피처:")
print(btc_features.info())
   
print("🚀 Step 2: 딥러닝 모델 클래스 정의...")
class MyTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout_prob=0.3):
        super(MyTradingModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.batch_norm(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
print("✅ 모델 클래스 MyTradingModel 정의 완료.")
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

print("🚀 Step 3: 모델 학습 및 예측 시작...")
# 3-1. 데이터 전처리 (스케일링, 시퀀스 생성, 분할)
feature_cols = [c for c in btc_features.columns if c != 'target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(btc_features[feature_cols])
y_target = btc_features['target'].values

def create_sequences(X, y, seq_length=30):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 30
X_seq, y_seq = create_sequences(X_scaled, y_target, SEQ_LENGTH)

split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# 3-2. 모델 학습
input_size = X_train.shape[2]
model = MyTradingModel(input_size=input_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🔥 모델 학습 시작...")
model.train()
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# 3-3. 예측
print("🔍 테스트 데이터 예측 중...")
model.eval()
with torch.no_grad():
    test_preds = model(torch.FloatTensor(X_test).to(device))
    predictions_prob = test_preds.cpu().numpy().flatten()

# 3-4. 시뮬레이션을 위한 최종 데이터 준비
test_dates = btc_features.index[split + SEQ_LENGTH:]
test_prices = btc_features['close'][split + SEQ_LENGTH:]
print("✅ 예측 및 최종 데이터 준비 완료.")
import matplotlib.pyplot as plt
print("🚀 Step 4: 백테스팅 및 결과 시각화 시작...")
# 4-1. 트레이딩 전략 함수
def my_advanced_strategy(predictions_prob, actual_prices, initial_cash=1000000, transaction_fee=0.001):
    cash = initial_cash
    holdings = 0
    trade_log = []
    portfolio_values = []
    for i in range(len(predictions_prob)):
        prob = predictions_prob[i]
        price = actual_prices.iloc[i]
        if prob >= 0.55 and cash > 0:
            holdings = (cash / (1 + transaction_fee)) / price
            cash = 0
            trade_log.append({'Date': actual_prices.index[i], 'Action': 'BUY', 'Price': price, 'Prob': prob})
        elif prob <= 0.45 and holdings > 0:
            cash = (holdings * price) * (1 - transaction_fee)
            holdings = 0
            trade_log.append({'Date': actual_prices.index[i], 'Action': 'SELL', 'Price': price, 'Prob': prob})
        portfolio_values.append(cash + holdings * price)
    return {'final_value': portfolio_values[-1], 'values': pd.Series(portfolio_values, index=actual_prices.index), 'log': pd.DataFrame(trade_log)}

# 4-2. 전략 실행
strategy_results = my_advanced_strategy(predictions_prob, test_prices)

# 4-3. Buy & Hold 전략과 비교
buy_hold_value = (1000000 / test_prices.iloc[0]) * test_prices.iloc[-1]
model_return = (strategy_results['final_value'] / 1000000 - 1) * 100
buy_hold_return = (buy_hold_value / 1000000 - 1) * 100

print("\n--- 최종 결과 ---")
print(f"💰 LSTM 모델 전략 최종 가치: {strategy_results['final_value']:,.0f} 원 (수익률: {model_return:.2f}%)")
print(f"📈 단순 보유 전략 최종 가치: {buy_hold_value:,.0f} 원 (수익률: {buy_hold_return:.2f}%)")

print("📈 시각화 결과는 별도의 이미지 파일로 생성되거나 GUI 환경에서 표시됩니다.")

print("\n--- 거래 기록 ---")
trade_log_df = strategy_results['log']
if not trade_log_df.empty:
    print(trade_log_df.set_index('Date'))
else:
    print("거래가 발생하지 않았습니다.")
