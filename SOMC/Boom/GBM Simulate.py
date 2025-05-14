import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: Load and prepare data ---
df = pd.read_excel('SET 50 Index Spot.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# --- STEP 2: Calculate log return, mu, sigma ---
df['Log Return'] = np.log(df['SET50'] / df['SET50'].shift(1))
df['Mu (Annualized Daily Return)'] = df['Log Return'] * 252
df['Sigma (Rolling Annualized Volatility)'] = df['Log Return'].rolling(window=250).std() * np.sqrt(250)

df_model = df.dropna(subset=['Log Return', 'Mu (Annualized Daily Return)', 'Sigma (Rolling Annualized Volatility)']).copy()
df_model.reset_index(drop=True, inplace=True)

real_prices = df_model['SET50'].values
mus = df_model['Mu (Annualized Daily Return)'].values
sigmas = df_model['Sigma (Rolling Annualized Volatility)'].values

simulated_prices = [real_prices[0]]
optimal_xi = []

# --- STEP 3: Simulate GBM day-by-day using best ξ ---
for t in range(1, len(real_prices)):
    S_t = real_prices[t-1]
    S_real = real_prices[t]
    mu = mus[t-1]
    sigma = sigmas[t-1]

    best_rmse = float('inf')
    best_price = None
    best_xi = None

    for xi in np.linspace(0.01, 0.99, 99):
        S_pred = S_t * np.exp((mu - 0.5 * sigma**2) * (1/252) + sigma * np.sqrt(1/252) * xi)
        rmse = np.sqrt((S_pred - S_real)**2)
        if rmse < best_rmse:
            best_rmse = rmse
            best_price = S_pred
            best_xi = xi

    simulated_prices.append(best_price)
    optimal_xi.append(best_xi)

# --- STEP 4: Store simulation results ---
df_model = df_model.iloc[1:].copy()
df_model['Simulated Price (Historical)'] = simulated_prices[1:]
df_model['Optimal ξ'] = optimal_xi

# --- STEP 5: Forward simulate 3 months (63 trading days) ---
forward_days = 63
S_last = simulated_prices[-1]
mu_forward = mus[-1]
sigma_forward = sigmas[-1]

future_simulated_prices = [S_last]
np.random.seed(42)

for _ in range(forward_days):
    xi = np.random.normal()
    S_next = future_simulated_prices[-1] * np.exp((mu_forward - 0.5 * sigma_forward**2) * (1/252) + sigma_forward * np.sqrt(1/252) * xi)
    future_simulated_prices.append(S_next)

# --- STEP 6: Create future date range ---
last_date = df_model['Date'].iloc[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forward_days+1)

df_forward = pd.DataFrame({
    'Date': future_dates[1:],  # remove today
    'Simulated Price': future_simulated_prices[1:]
})

# --- STEP 7 (Revised): Add Strike Price Columns to only forward data ---
strike_diffs = np.arange(-200, 225, 25)  # -100 to +100 step 25 (9 columns)

for diff in strike_diffs:
    col_name = f"Strike {diff:+}"
    df_forward[col_name] = df_forward['Simulated Price'].apply(lambda x: round(x + diff, 2))


# --- STEP 8: Plot ---
plt.figure(figsize=(14, 6))
plt.plot(df_model['Date'], df_model['SET50'], label='Actual SET50', linewidth=2)
plt.plot(df_model['Date'], df_model['Simulated Price (Historical)'], label='Simulated (Historical ξ)', linestyle='--')
plt.plot(df_forward['Date'], df_forward['Simulated Price'], label='Simulated Forward 3M', linestyle='-.', color='orange')
plt.title('SET50 Simulation (Historical + Forward GBM)')
plt.xlabel('Date')
plt.ylabel('SET50 Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Set display options to show full content
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Don't break lines
pd.set_option('display.max_colwidth', None) # Show full content in each cell

print(df_forward)
# --- STEP X: Export df_forward to Excel ---
#output_filename = "SET50_Forward_Simulation.xlsx"
#df_forward.to_excel(output_filename, index=False)
#print(f"📁 Exported to: {output_filename}")
