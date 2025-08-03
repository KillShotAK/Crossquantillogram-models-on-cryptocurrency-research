import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from matplotlib.dates import YearLocator, DateFormatter
from tqdm import tqdm
import matplotlib.dates as mdates

sns.set_style("whitegrid")

def load_data(path_marketcap, path_close):
    """
    Loads market cap and close-price data, computes daily log returns.
    """
    df_mcap = pd.read_csv(path_marketcap, parse_dates=['Date'], index_col='Date')
    df_close = pd.read_csv(path_close, parse_dates=['Date'], index_col='Date')

    start_date = '2020-01-01'
    end_date = '2024-12-31'
    df_mcap = df_mcap.loc[(df_mcap.index >= start_date) & (df_mcap.index <= end_date)]
    df_close = df_close.loc[(df_close.index >= start_date) & (df_close.index <= end_date)]

    df_mcap.sort_index(inplace=True)
    df_close.sort_index(inplace=True)
    common_dates = df_mcap.index.intersection(df_close.index)
    df_mcap = df_mcap.loc[common_dates]
    df_close = df_close.loc[common_dates]

    df_ret = np.log(df_close).diff().dropna()
    bank_size_vec = df_mcap.iloc[-1, :]

    return df_ret, bank_size_vec

def cross_quantilogram(y1, y2, tau=0.05, max_lag=10):
    """
    Computes cross-quantilogram values for lags 1 to max_lag.
    """
    results = {}
    for k in range(1, max_lag+1):
        y1_lag = y1[k:]
        y2_lag = y2[:-k]
        if len(y1_lag) == 0 or len(y2_lag) == 0:
            results[k] = 0
            continue

        q1 = np.percentile(y1_lag, tau*100)
        q2 = np.percentile(y2_lag, tau*100)
        psi1 = (y1_lag < q1).astype(float) - tau
        psi2 = (y2_lag < q2).astype(float) - tau

        num = np.sum(psi1 * psi2)
        denom = np.sqrt(np.sum(psi1**2)*np.sum(psi2**2))
        cq = num/denom if denom != 0 else 0
        results[k] = cq
    return results

def stationary_bootstrap(series, block_prob=0.1):
    """
    Implements Politis & Romano (1994) stationary bootstrap.
    """
    n = len(series)
    out = []
    idx = random.randint(0, n-1)
    for _ in range(n):
        out.append(series[idx])
        if random.random() < block_prob:
            idx = random.randint(0, n-1)
        else:
            idx = (idx + 1) % n
    return np.array(out)

def stationary_bootstrap_test(y1, y2, tau=0.05, max_lag=10, B=100, alpha=0.05):
    """
    Performs Ljung-Box type test using stationary bootstrap.
    """
    cqs_dict = cross_quantilogram(y1, y2, tau, max_lag)
    stat_actual = sum(val**2 for val in cqs_dict.values())

    dist_boot = np.zeros(B)
    for b in range(B):
        y1_star = stationary_bootstrap(y1)
        y2_star = stationary_bootstrap(y2)
        cqs_star = cross_quantilogram(y1_star, y2_star, tau, max_lag)
        dist_boot[b] = sum(x**2 for x in cqs_star.values())

    p_val = np.mean(dist_boot >= stat_actual)
    return p_val, cqs_dict

def build_adjacency_matrix(returns_window, tau=0.05, alpha=0.05, max_lag=10, B=100):
    """
    Builds adjacency matrix based on significant cross-quantilograms.
    """
    banks = returns_window.columns
    n = len(banks)
    A = pd.DataFrame(np.zeros((n, n)), index=banks, columns=banks)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            y_i = returns_window.iloc[:, i].dropna().values
            y_j = returns_window.iloc[:, j].dropna().values

            m = min(len(y_i), len(y_j))
            y_i, y_j = y_i[-m:], y_j[-m:]
            if len(y_i) < 10:
                continue

            p_val, cqs_dict = stationary_bootstrap_test(
                y_i, y_j, tau, max_lag,
                B=B, alpha=alpha
            )
            if p_val < alpha:
                A.iloc[i, j] = sum(cqs_dict.values())
    return A

def systemic_risk_score(A, size_vec):
    """
    Computes systemic risk score: S(A, c) = c^T * A * c
    """
    c = size_vec[A.index]
    c_mat = c.values.reshape(-1, 1)
    A_mat = A.values
    val = c_mat.T @ A_mat @ c_mat
    return float(val[0, 0])

def rolling_systemic_risk(df_ret, size_vec, window_size=364, step=30,
                         tau=0.05, alpha=0.05, max_lag=10, B=100):
    """
    Computes rolling window systemic risk measures.
    """
    dates = df_ret.index
    i_start = 0
    out = []

    n_windows = (len(dates) - window_size) // step + 1
    pbar = tqdm(total=n_windows, desc=f"Computing {window_size}-day window")

    while i_start + window_size <= len(dates):
        window_slice = dates[i_start: i_start + window_size]
        ret_win = df_ret.loc[window_slice]

        A_win = build_adjacency_matrix(ret_win, tau=tau, alpha=alpha,
                                     max_lag=max_lag, B=B)
        score = systemic_risk_score(A_win, size_vec)
        end_date = window_slice[-1]

        out.append((end_date, score))
        i_start += step
        pbar.update(1)

    pbar.close()
    df_out = pd.DataFrame(out, columns=['Date', 'Score'])
    df_out.set_index('Date', inplace=True)
    return df_out

def normalize_0_100(df_risk):
    """
    Normalizes systemic risk measures to 0-100 scale.
    """
    max_score = df_risk['Score'].max()
    df_risk['ScoreNorm'] = 100 * df_risk['Score'] / max_score if max_score > 0 else df_risk['Score']
    return df_risk

def run_window_analysis(df_returns, bank_size_vec, window_size):
    """Run analysis for a single window size"""
    df_risk = rolling_systemic_risk(
        df_ret=df_returns,
        size_vec=bank_size_vec,
        window_size=window_size,
        step=30,
        tau=0.05,
        alpha=0.05,
        max_lag=10,
        B=100
    )
    return normalize_0_100(df_risk)

def plot_systemic_risk_comparison(df_300, df_364, df_400):
    """
    Plot systemic risk comparison for different window sizes with updated style
    """
    plt.figure(figsize=(12, 6))

    # Plot the normalized systemic risk index for each window
    plt.plot(df_300.index, df_300['ScoreNorm'], label='300-day Window', 
             color='red', marker='o', linewidth=1)
    plt.plot(df_364.index, df_364['ScoreNorm'], label='364-day Window', 
             color='blue', marker='o', linewidth=1)
    plt.plot(df_400.index, df_400['ScoreNorm'], label='400-day Window', 
             color='green', marker='o', linewidth=1)

    # Set plot labels and title
    plt.xlabel('Date')
    plt.ylabel('Normalized Systemic Risk Index (ScoreNorm)')
    plt.title('Comparison of Normalized Systemic Risk Index Across Different Rolling Windows')

    # Format the x-axis
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    # Display legend and grid
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    return plt.gcf()

if __name__ == "__main__":
    # File paths
    path_marketcap = "/home/killshotak/Documents/crossqunatillogram/data_wide_marketcap.csv"
    path_close = "/home/killshotak/Documents/crossqunatillogram/data_wide_close.csv"

    # Load data
    print("Loading data...")
    df_returns, bank_size_vec = load_data(path_marketcap, path_close)

    # Run analysis for each window size
    print("\nComputing systemic risk for different window sizes...")
    df_risk_300 = run_window_analysis(df_returns, bank_size_vec, 300)
    df_risk_300.to_csv("systemic_risk_300.csv")

    df_risk_364 = run_window_analysis(df_returns, bank_size_vec, 364)
    df_risk_364.to_csv("systemic_risk_364.csv")

    df_risk_400 = run_window_analysis(df_returns, bank_size_vec, 400)
    df_risk_400.to_csv("systemic_risk_400.csv")

    # Create and save comparison plot
    print("\nGenerating comparison plot...")
    fig = plot_systemic_risk_comparison(df_risk_300, df_risk_364, df_risk_400)
    fig.savefig("systemic_risk_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()

    print("\nDone. Results saved in CSV files and plot saved as 'systemic_risk_comparison.png'")