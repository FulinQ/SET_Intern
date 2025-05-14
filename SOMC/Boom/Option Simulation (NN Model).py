import pandas as pd
import numpy as np
from scipy.stats import norm

# -----------------------------------------------------------
# STEP 1: โหลดและปรับ Strike ให้หารด้วย 25
# -----------------------------------------------------------
df = pd.read_excel('SET50_Forward_Simulation.xlsx')

for col in df.columns:
    if 'Strike' in col:
        df[col] = df[col].apply(lambda x: round(x / 25) * 25)

df['End of Contract'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp('M')

# -----------------------------------------------------------
# STEP 2: รวมราคาจริงกับราคาจำลอง
# -----------------------------------------------------------
df_real = pd.read_excel('SET 50 Index GBM.xlsx')
df_sim = pd.read_excel('SET50_Forward_Simulation.xlsx')

df_real = df_real.rename(columns={'SET50': 'Price'})
df_sim = df_sim.rename(columns={'Simulated Price': 'Price'})

df_combined = pd.concat([df_real, df_sim], ignore_index=True)
df_combined = df_combined.dropna(axis=1, how='any')
df_combined['Date'] = pd.to_datetime(df_combined['Date'])
df_combined = df_combined.sort_values('Date').reset_index(drop=True)

df_combined['Return'] = df_combined['Price'].pct_change()
df_combined['Rolling_SD_250'] = df_combined['Return'].rolling(window=250).std() * np.sqrt(252)
print(df_combined)
# -----------------------------------------------------------
# STEP 3: Merge ข้อมูลทั้งหมด
# -----------------------------------------------------------
df_merged = df.merge(df_combined[['Date', 'Return', 'Rolling_SD_250']], on='Date', how='left')
df_merged['Day Left'] = (df_merged['End of Contract'] - df_merged['Date']).dt.days
df_merged['T'] = df_merged['Day Left'] / 365
print(df_merged)
# -----------------------------------------------------------
# STEP 4: คำนวณ Delta (เลือกสูตร Call หรือ Put ตามชื่อ Strike)
# -----------------------------------------------------------
def calculate_d1(S, K, r, sigma, T):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def delta_call(S, K, r, sigma, T):
    d1 = calculate_d1(S, K, r, sigma, T)
    return norm.cdf(d1)

def delta_put(S, K, r, sigma, T):
    d1 = calculate_d1(S, K, r, sigma, T)
    return norm.cdf(d1) - 1

# ตัวแปรสำหรับคำนวณ
r = 0.014  # risk-free rate
S = df_merged['Simulated Price']
sigma = df_merged['Rolling_SD_250']
T = df_merged['T']

# วนลูปแต่ละ Strike column แล้วคำนวณ Delta
strike_columns = [col for col in df_merged.columns if 'Strike' in col]

for col in strike_columns:
    K = df_merged[col]
    if '-' in col:
        df_merged[f'Delta_{col}'] = delta_put(S, K, r, sigma, T)
    elif '+' in col:
        df_merged[f'Delta_{col}'] = delta_call(S, K, r, sigma, T)
    else:
        df_merged[f'Delta_{col}'] = np.nan

# -----------------------------------------------------------
# STEP 5: Export to Excel
# -----------------------------------------------------------
#df_merged.to_excel('SET50_Forward_Simulation_Merged.xlsx', index=False)
#print("✅ Exported to 'SET50_Forward_Simulation_Merged.xlsx'")

#########################################################################################################################

# Neural Network

#df_merged['Date'] = pd.to_datetime(df_merged['Date'])
#df_combined['Date'] = pd.to_datetime(df_combined['Date'])
#df_combined = df_combined.merge(
#    df_merged[['Date', 'Strike -200','End of Contract','Day Left']],
#    on='Date',
#    how='left'
#)
# Set display options to show full content
#pd.set_option('display.max_rows', None)     # Show all rows
#pd.set_option('display.max_columns', None)  # Show all columns
#pd.set_option('display.width', None)        # Don't break lines
#pd.set_option('display.max_colwidth', None) # Show full content in each cell

#print(df_combined)

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# กำหนดฟังก์ชันหาสถานะ Option ณ End of Contract
def get_option_status(final_price, strike):
    if final_price > strike:
        return [0, 0, 1]  # OTM
    elif final_price == strike:
        return [0, 1, 0]  # ATM
    else:
        return [1, 0, 0]  # ITM

# สมมติ End of Contract คือวันสุดท้ายใน df_combined
end_contract_date = df_combined.index[-1]
final_price = df_combined.loc[end_contract_date, 'Price']

# เตรียม DataFrame เก็บผลลัพธ์
df_results = df_merged.copy()
df_results[['%ITM', '%ATM', '%OTM']] = 0

# ทำ Prediction รายวัน
for current_date in df_combined.index:
    if current_date >= end_contract_date or current_date not in df_merged.index:
        continue

    X = df_combined.loc[:current_date, 'Price'].values.reshape(-1, 1)

    # กำหนดเป้าหมายจากสถานะของวันสิ้นสัญญา (ดูจาก Strike -200 ในวันนั้น)
    strike_today = df_merged.loc[current_date, 'Strike -50']
    y = np.array([get_option_status(final_price, strike_today)] * len(X))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # สร้าง Neural Network
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # เทรนโมเดล
    model.fit(X_scaled, y, epochs=30, verbose=0)

    # ใช้ราคา ณ วันล่าสุด predict
    X_today_scaled = scaler.transform(X_scaled[-1].reshape(1, -1))
    prob = model.predict(X_today_scaled)[0]

    # เก็บผลลัพธ์ใน DataFrame
    df_results.at[current_date, '%ITM'] = prob[0]*100
    df_results.at[current_date, '%ATM'] = prob[1]*100
    df_results.at[current_date, '%OTM'] = prob[2]*100

# เพิ่ม Column ประเภท Option
df_results['Type of Option'] = 'Short Call'

# ดูผลลัพธ์
print(df_results.head(10))
output_filename = "SET50_Option_Status_Simulation.xlsx"
df_results.to_excel(output_filename, index=False)