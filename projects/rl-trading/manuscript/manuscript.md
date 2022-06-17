# Intraday Trading via Deep Reinforcement Learning and Technical Indicators

## Abstract

Trading based on Reinforcement Learning (RL) has sparked a lot of attention. However, in previous research, RL 
in the day-trade task has been found to suffer by noisy financial signals on a short time scale and a costly 
search of a continuous-valued action space. Based on the use of Technical Indicators (TI) and 
Deep Reinforcement Learning (DRL), this work introduces an end-to-end RL intraday trading agent. We 
show that our model outperforms other popular strategies based on TI 
using cryptocurrency data and a portfolio of multiple assets.

## Introduction

Financial trading is an online decision-making process (Deng et al., 2016). Previous works (Moody and Saffell, 
1998; Moody and Saffell, 2001; Dempster and Leemans, 2006) demonstrated the RL agent’s 
promising profitability in trading activities. However, traditional RL algorithms face challenges for the intraday 
trading problem in three aspects: 

- Short-term financial movement is often accompanied by noisy oscillations. 
- The computational complexity for making decisions in daily continuous-value price range is high. 
- The early stop of orders when applying the intraday strategy. 

In a typical scenario, the settlement of orders involves two hyperparameters: 

- Target Profit (TP)

- Stop Loss (SL) 

TP refers to the price closing the activating order and collect the profit if the price moves as expected. 
SL denotes the price to terminate the transaction and avoid a further loss if the price moves towards a loss direction. These two 
hyperparameters are defined as a fixed threshold. If the price crosses these two-preset levels, the order will be closed deterministically. 

Focusing on the mentioned challenges, we propose a DRL model, named TIRL-Trader. Our model directly generates the trading policy using TI 
instead of using fixed TP and SL. TIRL-Trader comprises two neural networks with different functions: 

- A Long-short Term Memory (LSTM) networks for extracting the temporal feature in financial time series.
- A policy generator network (PGN) for generating the distribution of actions (policy) in each state. 

Experiments on cryptocurrency financial datasets and comparisons with popular rule-based trading systems have been conducted. 
Our TIRL-Trader outperforms some state-of-the-art baselines in the profitability evaluated by the cumulative return and the 
risk-adjusted return (Sharpe ratio), showing adaptability in the unseen market environment. The generated policy of 
TIRL-Trader also provides an explainable profit-and-loss order control strategy.

Our main contributions could be summarized as:

- We propose a novel end-to-end daytrade model that directly learns the optimal trading strategy, thus solving the 
early stop in an implicit stop-loss and target-profit setting.
- We constraint the RL agent’s action space via the utilization of TI.
- We achieve better profitability and robustness compared to state-of-the-art rule-based strategies.

## Related Work

Our work is in line with trading based on DRL and strategies that utilize TI. Therefore, we shortly review past studies.

### Deep Reinforcement Learning in Trading

Algorithmic trading has been widely studied in its different subareas, including risk control (Pichler et al., 2021), portfolio 
optimization (Giudici et al., 2020), and trading strategy (Marques and Gomes, 2010; Vella and Ng, 2015; Chen et al., 2021).
Nowadays, the AI-based trading, especially, the RL approach, attracts the interest in both academia and 
industry. Moody and Saffell (2001) proposed a direct RL algorithm to trade and performed a comprehensive comparison 
between the Q-learning with the policy gradient. Huang et al. (2016) further propose a robust trading agent based on the 
Deep-Q networks (DQN). Deng et al. (2016) utilized the fuzzy logic with a deep learning model to extract the financial 
feature from noisy time series, which achieved state-of-the-art performance in a single asset trading. Xiong et al. 
(2018) used the Deep Deterministic Policy Gradient (DDPG) based on the standard actor-critic framework for 
stock trading. The experiments demonstrated their profitability over the baselines including the min-variance portfolio 
allocation method and the technical approach based on the Dow Jones Industrial Average (DJIA) index. Wang et al. (2019) 
used the RL algorithm to construct the winner and loser portfolio and traded in the buy-winner-sell-loser strategy. 
However, the intraday trading for RL agents remains less addressed, which is mainly because of the complexity 
in designing an action space for frequent trading strategies.

### Technical Indicators in Trading

Trading in practice involves analyzing different charts and making decisions based on patterns and indicators. Regardless of 
a trader's proficiency, it is accepted among industry practitioners that TI play a pivotal role in market analysis. The financial 
market is quite dynamic, current affairs and concurrent events also heavily influence the market situation. The TI 
provide useful information about market trends and help maximize the returns. The provided information and the corresponding 
TI can be categorized as follows:

- Trend. 

They indicate the trend of the market or the direction in which the market is moving. Typically, the trend indicators 
are oscillators, i.e they tend to move between high and low values.

- Momentum. 

They indicate the strength of the trend and also signal whether there is any likelihood of reversal. Relative Strength Index (RSI) 
is one momentum indicator, used for indicating the top and bottom price.

- Volume. 

They indicate how the volume changes with time. They also indicate the number of assets that are being bought 
and sold over time. When the price changes, volume indicates how strong the move is.

- Volatility. 

They indicate how much the price is changing in the given period. High volatility indicates large price moves 
while lower volatility indicates price stability.

In the current work we utilize the following TI:

- Moving Averages

Moving averages is a frequently used intraday trading indicator. It provides information about the momentum of the market, trends 
in the market, the reversal of trends, and the stop loss points.

- Bollinger Bands

Bollinger bands indicate the volatility in the market. Bollinger bands are of 3 types: a middle band which is a 20-day simple moving 
average, a +2 standard deviation upper band and a -2 lower deviation lower band.

- Relative Strength Index (RSI)

RSI is a momentum indicator. It is a single line ranging from 0 to 100 which indicates when the stock is overbought or oversold in the market.

- Commodity Channel Index

Commodity Channel Index identifies new trends in the market. It has values of 0, +100, and -100.

- Stochastic Oscillator

The stochastic oscillator is one of the momentum indicators. The oscillator compares the closing price of a stock to a range of prices over a 
period of time.

## TIRL-Trader

The architecture of the TIRL-Trader has two main components:

### LSTM Network

LSTM networks show promising performance in the sequential feature learning, as its structural adaptability (Gers et al., 2000). We use the 
LSTM networks to extract the temporal features of the financial series. We apply the same look-back window in (Wang et al., 2019) with size W to 
split the input sequence () from the completed series (), i.e., agent evaluates the market status by the time period with size W. Hence, the 
input matrix of LSTM could be noted as ().

The input vectors is constituted by: 

- Opening, highest, lowest and closing prices as well as transaction volume for each trading day.

- Per minute data of the following values:

    - Prices

    - Moving Averages

    - Bollinger Bands

    - RSI

    - Commodity Channel Index

    - Stochastic Oscillator (KDJ)

The above features are included multiple times with different parametrization. Additionally, indices are included that are extracted from the above features and 
correspond to popular strategies. More details are provided in the sections below.

### Policy Generator Networks

Given the learned feature vector, PGN directly produces the output policy, i.e., the probability of setting an order in each time interval. The 
output policy at is calculated as follows:

()

In timestep t, model takes action () by sampling from the policy () comprised of only long positions.

We train the TIRL-Trader with standard RL techniques. The key idea is to maintain a loop with the successive steps: 

- The agent is aware of the environment through the state vector ().

- The agent applies an action.

- The agent adjust its behavior to receive more reward until it has completed its learning goal (Sutton and Barto, 2018). 

The objective is to maximize the expectation of reward (). Gradient descent is used during the optimization phase. To avoid the local 
minimum problem caused by the multiple postive-reward actions, we use the state-dependent threshold method (Sutton and Barto, 2018).

The overall summary of TIRL-Trader architecture is shown in the next figure:

()

## Experiments

We conduct the empirical evaluation for our TIRL-Trader in 

### Dataset

### Models

### Performance

## Analysis

## Conclusions and Future Work