# Regime-Switching Price Simulator

This project simulates stock price evolution using **Geometric Brownian Motion (GBM)** enhanced with two behavioral volatility models:

1. **Basic Regime Switching** — a naïve model where market regimes (calm vs. panic) are randomly determined each day with no memory
2. **Markov Chain Regime Switching** — a more realistic model where market regimes follow a first-order Markov process, allowing volatility to persist across time

---

## Purpose

Markets don’t behave in a vacuum. Periods of **calm** and **panic** often cluster together, especially during crises. This simulator explores the consequences of modeling volatility in two ways:

- One that **ignores psychological persistence**  
- One that **captures it explicitly** using Markovian logic

By comparing the two, we highlight how assumptions about behavior and memory influence simulated price dynamics, risk profiles, and the emergence of extreme events.

---

## Key Concepts

- **Geometric Brownian Motion (GBM):** A classic model for simulating stock prices, assuming continuous compounding and randomness
- **Volatility Regimes:** Two states — *calm* (low volatility) and *panic* (high volatility)
- **Memoryless Switching:** Each day has an independent chance of entering panic (e.g., 5%)
- **Markov Chain Switching:** Regime transitions depend on the previous day's state, making panic more likely to persist

---

## Simulation Details

| Parameter             | Value / Description                     |
|----------------------|------------------------------------------|
| Initial price         | $100                                     |
| Expected return (μ)   | 7% annualized                            |
| Calm volatility (σ₁)  | 20% annualized                           |
| Panic volatility (σ₂) | 50% annualized                           |
| Time horizon (T)      | 1 year                                   |
| Time steps (N)        | 252 (approx. trading days in a year)     |
| Simulations           | 500 independent price paths              |

---

## Outputs

- Side-by-side price path plots (Basic vs. Markov)
- Regime tracking (for Markov model)
- Final portfolio value distributions for both models
- Visualization of how persistence in volatility affects:
  - Spread of outcomes
  - Tail risk
  - Volatility clustering

---

## Why It Matters

This project demonstrates how **modeling behavioral persistence** — even with simple tools like a Markov chain — can **dramatically change simulated outcomes**. It's an invitation to move beyond textbook assumptions and start capturing more realistic market psychology.

---

## Run It Yourself

1. Clone the repo
2. Install requirements:
   ```bash
   pip install numpy matplotlib seaborn
