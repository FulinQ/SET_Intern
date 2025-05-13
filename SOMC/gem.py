import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def ito_process_bounded_simulation(S0, T, n_steps, n_sim, mu, sigma, max_change_pct, K):
    """
    Simulates asset prices using a general Itô process with bounded price changes.

    Args:
        S0 (float): Initial asset price.
        T (float): Time horizon.
        n_steps (int): Number of time steps.
        n_sim (int): Number of simulation paths.
        mu (function): Drift rate function.
        sigma (function): Volatility function.
        max_change_pct (float): Maximum percentage change from previous price.
        K (float): Strike price for option analysis.

    Returns:
        tuple: (time_points, S_paths) where:
            time_points: Array of time points.
            S_paths: 2D array of simulated asset price paths.
    """

    dt = T / n_steps
    time_points = np.linspace(0, T, n_steps + 1)
    S_paths = np.zeros((n_steps + 1, n_sim))
    S_paths[0, :] = S0

    dW = np.random.standard_normal(size=(n_steps, n_sim)) * np.sqrt(dt)

    for t in range(1, n_steps + 1):
        prev_S = S_paths[t - 1, :]
        current_time = time_points[t - 1]

        drift = mu(prev_S, current_time) * dt
        diffusion = sigma(prev_S, current_time) * dW[t - 1, :]
        
        # Limit the price change
        max_change = max_change_pct * prev_S
        dS = np.clip(drift + diffusion, -max_change, max_change)
        
        S_paths[t, :] = prev_S + dS

    return time_points, S_paths

def calculate_option_probabilities(S_paths, K):
    """
    Calculates the probability of option prices being ITM, ATM, and OTM.

    Args:
        S_paths (ndarray): Simulated asset price paths.
        K (float): Strike price.

    Returns:
        dict: Probabilities of ITM, ATM, and OTM.
    """

    final_prices = S_paths[-1, :]  # Prices at final time step
    itm_prob = np.mean(final_prices > K)
    atm_prob = np.mean(np.abs(final_prices - K) < 1e-2)  # Using a tolerance
    otm_prob = np.mean(final_prices < K)

    return {"ITM": itm_prob, "ATM": atm_prob, "OTM": otm_prob}


if __name__ == '__main__':
    # --- Parameters ---
    S0 = 800# Initial price
    T = 1.0     # Time horizon (years)
    n_steps = int(252/2)  # Daily steps
    n_sim = 12  # Number of simulation paths
    K = 800.0    # Strike price
    max_change_pct = 1  # Maximum 30% change

    mu_func = lambda S, t: -0.0117  # Example: Constant drift
    sigma_func = lambda S, t: 0.1242  # Example: Constant volatility

    # --- Simulation ---
    time, S_paths = ito_process_bounded_simulation(S0, T, n_steps, n_sim, mu_func, sigma_func, max_change_pct, K)

    # --- Option Probability Analysis ---
    probabilities = calculate_option_probabilities(S_paths, K)
    print("Option Probabilities:")
    print(f"ITM: {probabilities['ITM']:.4f}")
    print(f"ATM: {probabilities['ATM']:.4f}")
    print(f"OTM: {probabilities['OTM']:.4f}")

    # --- Plotting (Optional) ---
    plt.plot(time, S_paths[:, :10])  # Plot first 10 paths
    plt.title('Ito Process Simulation with Bounded Price Changes')
    plt.xlabel('Time (Years)')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # ... (your simulation)

    # --- Plot histogram of final prices ---
    plt.hist(S_paths[-1, :], bins=20, density=True, alpha=0.6, color='blue')
    plt.axvline(K, color='red', linestyle='dashed', linewidth=1, label='Strike Price')
    plt.title('Distribution of Final Prices')
    plt.xlabel('Final Price')
    plt.ylabel('Density')
    plt.legend()
    plt.show()