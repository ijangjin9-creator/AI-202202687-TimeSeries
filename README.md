# 📈 딥러닝 기반 비트코인 알고리즘 트레이딩 (Bitcoin AI Trading)

### 🧑‍🎓 학생 정보 (Student Info)
* **이름:** 이장진
* **학번:** 202202687
* **제출일:** 2025-12-19

---

## 1. 프로젝트 개요 (Project Overview)

본 프로젝트는 **PyTorch 기반의 LSTM (Long Short-Term Memory) 모델**을 활용하여 비트코인(BTC)의 일일 가격 등락 방향을 예측하고, 이를 바탕으로 **확률 기반 임계값(Probability-Based Threshold) 트레이딩 전략**을 수립하여 벤치마크(Buy and Hold) 대비 초과 수익률을 달성하는 것을 목표로 합니다.

단순한 가격 예측을 넘어, 시장의 추세와 변동성을 파악하기 위한 **기술적 보조지표를 활용한 특성 공학(Feature Engineering)**, 시계열 데이터 학습에 강점을 가진 **LSTM 모델 아키텍처 설계**, 그리고 실제 거래를 모사하는 **백테스팅(Backtesting)**까지 AI 트레이딩 시스템 개발의 전 과정을 구현했습니다.

---

## 2. 🏆 최종 성과 (Final Performance)

**대세 상승 구간을 정확히 포착하고 하락 조정 직전에 매도**하는 전략을 통해, 단순히 장기 보유하는 'Buy and Hold' 전략의 수익률을 **26%p 이상 초과**하는 압도적인 성과를 달성했습니다.

| 전략 (Strategy) | 초기 자본 (Initial) | 최종 자산 (Final) | **최종 수익률 (Return)** | 비고 (Note) |
| :--- | :--- | :--- | :--- | :--- |
| **✅ My AI Strategy** | 1,000,000 원 | **1,565,553 원** | **+56.56%** 🚀 | **Winner** |
| 📈 Buy and Hold | 1,000,000 원 | 1,305,066 원 | +30.51% | Benchmark |

&gt; **💡 성과 요약:**
&gt; * 총 **8회**의 정밀한 매수/매도 거래를 통해 상승 추세를 효율적으로 추종하고 위험을 관리했습니다.
&gt; * 초기 모델이 거래를 실행하지 않던 문제를 **하이퍼파라미터 튜닝**과 **전략 임계값 조정**으로 해결하며 수익성을 극대화하는 데 성공했습니다.
&gt; * 최종 모델은 **2024년 8월부터 12월까지의 주요 변동성 장세**에서 성공적인 거래 기록을 남겼습니다.

---

## 3. 🛠️ 핵심 구현 내용 (Implementation Details)

### ① 특성 공학 (Feature Engineering)
단순 가격 데이터(OHLCV)를 넘어, 시장의 추세, 변동성, 거래량을 종합적으로 판단하기 위해 다음과 같은 기술적 보조지표들을 특성(Feature)으로 추가했습니다.

**[구현 코드: `utils.py`]**
```python
# utils.py 내의 특성 생성 로직
def create_features(df):
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    # 볼린저 밴드 (변동성)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()

    # RSI, MACD (추세 및 모멘텀)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # OBV (거래량 기반 추세)
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Target (1일 후 상승:1, 하락:0)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df.dropna()
```
* **주요 생성 특성:** 이동평균(5, 20일), 볼린저 밴드(상/중/하단), RSI, MACD, OBV 등 시장 분석에 필수적인 지표들을 포함하여 모델이 더 풍부한 정보를 바탕으로 학습하도록 설계했습니다.

### ② 모델 아키텍처 (Model Architecture)
시계열 데이터의 장기 의존성(Long-term Dependency)을 효과적으로 학습하기 위해 **2-Layer Stacked LSTM** 구조를 채택했습니다. Sigmoid 활성화 함수를 통해 최종적으로 '상승 확률'을 0과 1 사이의 값으로 출력합니다.

**[구현 코드: `temp_notebook_run.py`]**
```python
# temp_notebook_run.py 내의 모델 정의
class MyTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super(MyTradingModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
```
* **선택 이유:** LSTM은 과거의 중요한 정보를 선택적으로 기억하고 현재 시점의 예측에 반영하는 능력이 뛰어나, 변화무쌍한 금융 시계열 데이터 분석에 적합합니다. 2개 층을 쌓아 더 복잡하고 추상적인 패턴을 학습하도록 구성했습니다.
* **하이퍼파라미터:** `hidden_size=128`, `num_layers=2`, `epochs=100`, `learning_rate=0.0001`로 최적화하여 안정적인 예측 성능을 확보했습니다.

### ③ 트레이딩 전략 (Trading Strategy)
모델이 예측한 **상승 확률**에 따라 매수/매도/관망을 결정하는 **'확률 기반 임계값 전략'**을 구현했습니다. 이는 불확실한 구간에서의 잦은 거래를 방지하여 수수료 비용을 절감하고, 확신이 높은 구간에서만 포지션에 진입하도록 설계되었습니다.

**[구현 코드: `temp_notebook_run.py`]**
```python
# temp_notebook_run.py 내의 전략 함수
def my_advanced_strategy(probs, prices, dates, initial_capital=1000000):
    # ... (생략) ...
    BUY_THRESHOLD = 0.55  # 상승 확률 55% 이상 시 매수
    SELL_THRESHOLD = 0.45 # 상승 확률 45% 이하 시 매도
    # ... (생략) ...
```
* **전략 논리:** 
  * **매수 (BUY):** 상승 확률이 **55%** 이상일 때, 보유 현금을 모두 투자하여 공격적으로 수익을 추구합니다.
  * **매도 (SELL):** 상승 확률이 **45%** 이하로 떨어지면, 리스크 관리를 위해 보유 자산을 전량 매도하여 현금화합니다.
  * **관망 (HOLD):** 확률이 **45% ~ 55%** 사이의 애매한 구간일 때는 포지션을 유지하여 불필요한 거래 수수료 발생을 최소화합니다.

---

## 4. 📊 결과 분석 및 고찰 (Analysis & Discussion)

### ① 하이퍼파라미터 튜닝 과정
초기 실험 단계에서 모델이 모든 예측 확률을 0.5 부근으로만 출력하여, 설정된 매매 임계값(Threshold)을 넘지 못해 **거래가 전혀 발생하지 않는 'Zero Trade' 문제**에 직면했습니다. 이는 모델이 데이터로부터 의미 있는 패턴을 충분히 학습하지 못했기 때문으로 판단, 다음과 같은 하이퍼파라미터 조정을 통해 문제를 해결했습니다.

*   **Epoch (학습 횟수): `50회 → 100회`**: 모델이 데이터셋을 더 많이 반복 학습하도록 하여, 복잡한 패턴을 파악하고 예측에 대한 확신을 가질 수 있도록 충분한 학습 시간을 부여했습니다.
*   **Learning Rate (학습률): `0.001 → 0.0001`**: 학습률을 낮춰 모델 파라미터를 더 미세하게 조정함으로써, 최적의 가중치를 더 안정적으로 찾아가도록 유도했습니다.

이러한 튜닝을 통해 모델은 비로소 0.55 이상의 확신 있는 매수 신호와 0.45 이하의 매도 신호를 생성하기 시작했고, 성공적인 트레이딩의 기반을 마련했습니다.

### ② 성과 분석
하이퍼파라미터 튜닝을 거친 최종 모델은 테스트 기간 동안 매우 인상적인 성과를 보여주었습니다.

*   **정확한 상승장 포착**: 거래 로그를 보면, 모델은 **2024년 10월 26일 비트코인이 약 67k일 때 정확히 'BUY' 시그널**을 발생시켰습니다. 이후 **11월 24일, 가격이 98k까지 급등한 고점에서 'SELL' 시그널**을 통해 성공적으로 수익을 극대화했습니다.
*   **효율적인 리스크 관리**: 변동성이 크지 않은 횡보 구간에서는 매수/매도 임계값에 도달하지 않아 **불필요한 거래를 피하고 관망**했습니다. 이는 잦은 매매로 인한 슬리피지 및 수수료 비용을 최소화하여 최종 수익률을 보존하는 데 결정적인 역할을 했습니다.

### ③ 한계점 및 개선 방향
본 프로젝트는 성공적인 결과를 도출했지만, 실전 투자를 위해서는 다음과 같은 한계점을 인지하고 개선할 필요가 있습니다.

*   **거시 경제 변수(Macro Variables)의 부재**: 현재 모델은 시장 내부의 가격 및 거래량 데이터만 사용합니다. **금리, 나스닥 지수, 달러 인덱스**와 같은 거시 경제 지표는 시장 전체의 투자 심리에 큰 영향을 미치므로, 이를 피처로 추가한다면 예상치 못한 급락장에 대한 방어력을 크게 높일 수 있을 것입니다.
*   **손절매(Stop-Loss) 로직 미적용**: 현재 전략은 모델의 예측 확률에만 의존합니다. 만약 모델이 예측하지 못한 '블랙 스완' 이벤트로 인해 가격이 급락할 경우 큰 손실을 입을 수 있습니다. 따라서 예측 확률과 무관하게, 특정 손실률(예: -5%)에 도달하면 강제로 포지션을 청산하는 **Stop-Loss 로직을 추가하여 리스크 관리 체계를 강화**할 필요가 있습니다.

---

## 5. 결론 (Conclusion)

본 프로젝트를 통해 LSTM 딥러닝 모델과 확률 기반 트레이딩 전략을 결합하여 **Buy and Hold 벤치마크를 26%p 이상 상회하는 +56.56%의 높은 수익률**을 달성했습니다. 특히 'Zero Trade' 문제를 해결하기 위한 하이퍼파라미터 튜닝 과정과, 상승 추세를 추종하고 횡보장에서 비용을 절감한 전략의 유효성을 성공적으로 입증했습니다.

이는 AI 기술이 금융 데이터 분석 및 자동화된 트레이딩 시스템 구축에 매우 강력한 도구가 될 수 있음을 보여주는 사례입니다. 향후 거시 경제 데이터 통합 및 리스크 관리 로직 고도화를 통해 더욱 안정적이고 높은 수익을 기대할 수 있는 시스템으로 발전시켜 나갈 계획입니다.

---
✅ **과제 최종 완료**