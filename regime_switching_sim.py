import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Parameters
S0 = 100            # Initial stock price
mu = 0.07           # Expected return
sigma_calm = 0.2    # Calm volatility (20%)
sigma_panic = 0.5   # Panic volatility (50%)
T = 1.0             # 1 year
N = 252             # Daily steps
n_simulations = 500

# Scenario 1: Basic regime-switching
panic_prob = 0.05   # 5% chance of panic at each step

# Scenario 2: Markov chain switching
prob_calm_to_panic = 0.05
prob_panic_to_calm = 0.2


#  Function: Basic regime-switching (memoryless)
def simulate_basic_regime_switching(S0, mu, sigma_calm, sigma_panic, panic_prob, T, N, n_simulations):
    dt = T / N
    paths = np.zeros((N + 1, n_simulations))
    paths[0] = S0

    for t in range(1, N + 1):
        # Random regime (no memory)
        is_panic = np.random.rand() < panic_prob
        sigma = sigma_panic if is_panic else sigma_calm

        Z = np.random.standard_normal(n_simulations)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return paths


#  Function: Markov chain regime-switching
def simulate_markov_regime_switching(S0, mu, sigma_calm, sigma_panic, 
                                     prob_calm_to_panic, prob_panic_to_calm, 
                                     T, N, n_simulations):
    dt = T / N
    paths = np.zeros((N + 1, n_simulations))
    paths[0] = S0

    regime_states = np.zeros(N + 1)  # Track the regime (0 = calm, 1 = panic)
    regime = 0  # Start in calm

    for t in range(1, N + 1):
        if regime == 0:  # calm
            if np.random.rand() < prob_calm_to_panic:
                regime = 1
        else:  # panic
            if np.random.rand() < prob_panic_to_calm:
                regime = 0

        regime_states[t] = regime
        sigma = sigma_panic if regime == 1 else sigma_calm

        Z = np.random.standard_normal(n_simulations)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return paths, regime_states


#  Run both simulations
paths_basic = simulate_basic_regime_switching(
    S0, mu, sigma_calm, sigma_panic, panic_prob, T, N, n_simulations
)

paths_markov, regime_states = simulate_markov_regime_switching(
    S0, mu, sigma_calm, sigma_panic, prob_calm_to_panic, prob_panic_to_calm,
    T, N, n_simulations
)

#  Plot: Side by Side Comparison
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plot basic
axs[0].plot(paths_basic, linewidth=0.6, alpha=0.6)
axs[0].set_title('Basic Regime-Switching (Memoryless)', fontsize=13)
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Stock Price')

# Plot Markov
axs[1].plot(paths_markov, linewidth=0.6, alpha=0.6)
axs[1].set_title('Markov Chain Regime-Switching (Sticky Volatility)', fontsize=13)
axs[1].set_xlabel('Time Step')

plt.suptitle('Regime-Switching GBM: Easy vs. Realistic Model', fontsize=15, fontweight='bold', color='hotpink')
plt.show()


#  Plot: Regime Tracking for Markov Chain (just to visualize one path's regime)
plt.figure(figsize=(12, 2))
plt.plot(regime_states, drawstyle='steps-pre', color='red')
plt.title('Regime Over Time (0 = Calm, 1 = Panic)', fontsize=12)
plt.xlabel('Time Step')
plt.yticks([0, 1], ['Calm', 'Panic'])
plt.show()


#  Stats: Final Distribution Comparison
final_basic = paths_basic[-1]
final_markov = paths_markov[-1]

plt.figure(figsize=(10, 6))
sns.kdeplot(final_basic, label='Basic Regime-Switching', fill=True)
sns.kdeplot(final_markov, label='Markov Chain Regime-Switching', fill=True)
plt.title('Distribution of Final Portfolio Values', fontsize=13)
plt.xlabel('Final Portfolio Value')
plt.legend()
plt.show()

if __name__ == "__main__":
    paths_basic = simulate_basic_regime_switching(
        S0, mu, sigma_calm, sigma_panic, panic_prob, T, N, n_simulations
    )

    paths_markov, regime_states = simulate_markov_regime_switching(
        S0, mu, sigma_calm, sigma_panic, prob_calm_to_panic, prob_panic_to_calm,
        T, N, n_simulations
    )

    # Repeat your plotting code here
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    axs[0].plot(paths_basic, linewidth=0.6, alpha=0.6)
    axs[0].set_title('Basic Regime-Switching (Memoryless)', fontsize=13)
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Stock Price')
    axs[1].plot(paths_markov, linewidth=0.6, alpha=0.6)
    axs[1].set_title('Markov Chain Regime-Switching', fontsize=13)
    axs[1].set_xlabel('Time Step')
    plt.suptitle('Regime-Switching GBM: Easy vs. Realistic', fontsize=15, fontweight='bold', color='hotpink')
    plt.show()

    # Track regime (just one path)
    plt.figure(figsize=(12, 2))
    plt.plot(regime_states, drawstyle='steps-pre', color='red')
    plt.title('Regime Over Time (0 = Calm, 1 = Panic)', fontsize=12)
    plt.xlabel('Time Step')
    plt.yticks([0, 1], ['Calm', 'Panic'])
    plt.show()

    # Distribution comparison
    final_basic = paths_basic[-1]
    final_markov = paths_markov[-1]

    plt.figure(figsize=(10, 6))
    sns.kdeplot(final_basic, label='Basic Regime-Switching', fill=True)
    sns.kdeplot(final_markov, label='Markov Chain Regime-Switching', fill=True)
    plt.title('Distribution of Final Portfolio Values', fontsize=13)
    plt.xlabel('Final Value')
    plt.legend()
    plt.show()
