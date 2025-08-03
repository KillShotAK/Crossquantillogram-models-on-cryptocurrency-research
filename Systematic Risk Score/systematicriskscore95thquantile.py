import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from matplotlib.dates import YearLocator, DateFormatter

# Set a fixed random seed for reproducible results
random.seed(42)
np.random.seed(42)

sns.set_style("whitegrid")

##############################################################################
#  1) LOAD DATA
##############################################################################
def load_data(path_marketcap, path_close):
    """
    Loads market cap and close-price data for 2020–2024, computes daily returns.
    Both CSVs must have:
      - 'Date' column (parse_dates=True, index_col='Date')
      - matching bank columns
    """
    # Read in data
    df_mcap = pd.read_csv(path_marketcap, parse_dates=['Date'], index_col='Date')
    df_close = pd.read_csv(path_close,    parse_dates=['Date'], index_col='Date')

    # Restrict to 2020–2024
    start_date = '2020-01-01'
    end_date   = '2024-12-31'
    df_mcap = df_mcap.loc[(df_mcap.index >= start_date) & (df_mcap.index <= end_date)]
    df_close = df_close.loc[(df_close.index >= start_date) & (df_close.index <= end_date)]

    # Sort by date just in case
    df_mcap.sort_index(inplace=True)
    df_close.sort_index(inplace=True)

    # Align on same dates
    common_dates = df_mcap.index.intersection(df_close.index)
    df_mcap = df_mcap.loc[common_dates]
    df_close= df_close.loc[common_dates]

    # Option A: two-day rolling returns (like paper to mitigate nonsynch).
    #           Or you could do df_ret = df_close.pct_change().
    df_ret = df_close.rolling(window=2).apply(lambda x: x.iloc[-1]/x.iloc[0] - 1)

    # We'll define bank sizes as final day market cap for demonstration:
    bank_size_vec = df_mcap.iloc[-1, :]

    return df_ret, bank_size_vec


##############################################################################
#  2) CROSS-QUANTILOGRAM & BOOTSTRAP
##############################################################################
def cross_quantilogram(y1, y2, tau=0.95, max_lag=10):  # Changed tau to 0.95
    """
    Returns a dict {lag: CQ_value} for lags = 1..max_lag.
    Using the hit-function definition (Han et al. 2016).
    """
    results = {}
    for k in range(1, max_lag+1):
        # Align y1[t] with y2[t-k]
        y1_lag = y1[k:]
        y2_lag = y2[:-k]
        if len(y1_lag) == 0 or len(y2_lag) == 0:
            results[k] = 0
            continue
        # Tau-quantiles
        q1 = np.percentile(y1_lag, tau*100)
        q2 = np.percentile(y2_lag, tau*100)
        # Hits
        psi1 = (y1_lag > q1).astype(float) - (1-tau)  # Changed to > for upper quantile
        psi2 = (y2_lag > q2).astype(float) - (1-tau)  # Changed to > for upper quantile
        num = np.sum(psi1 * psi2)
        denom = np.sqrt(np.sum(psi1**2)*np.sum(psi2**2))
        cq = num/denom if denom != 0 else 0
        results[k] = cq
    return results

def stationary_bootstrap(series, block_prob=0.1):
    """
    Draws a bootstrap sample from 'series' using
    Politis & Romano (1994) stationary bootstrap.
    """
    n = len(series)
    out = []
    idx = random.randint(0, n-1)
    for _ in range(n):
        out.append(series[idx])
        # With prob block_prob, jump to random index
        if random.random() < block_prob:
            idx = random.randint(0, n-1)
        else:
            idx = (idx + 1) % n
    return np.array(out)

def stationary_bootstrap_test(y1, y2, tau=0.95, max_lag=10,  # Changed tau to 0.95
                              B=100, alpha=0.05):  # Kept B=100 as requested
    """
    Ljung-Box-type test over lags=1..max_lag.
    H0: all CQ(1..max_lag) = 0.
    We use sum-of-squares across the lags as test statistic.
    Returns (p_value, {lag: CQ_value}).
    """
    # Actual cross-quantilogram
    cqs_dict = cross_quantilogram(y1, y2, tau, max_lag)
    stat_actual = sum(val**2 for val in cqs_dict.values())

    # Bootstrap distribution
    dist_boot = np.zeros(B)
    for b in range(B):
        y1_star = stationary_bootstrap(y1)
        y2_star = stationary_bootstrap(y2)
        cqs_star = cross_quantilogram(y1_star, y2_star, tau, max_lag)
        dist_boot[b] = sum(x**2 for x in cqs_star.values())

    # p-value
    p_val = np.mean(dist_boot >= stat_actual)
    return p_val, cqs_dict


##############################################################################
#  3) ADJACENCY MATRIX & SYSTEMIC RISK
##############################################################################
def build_adjacency_matrix(returns_window, tau=0.95, alpha=0.05,  # Changed tau to 0.95
                           max_lag=10, B=100):  # Kept B=100
    """
    For each directed pair i->j:
      - If test is significant (p<alpha), A[i,j] = sum of CQ over all lags (1..10)
      - Else 0
    Produces a directional adjacency (not forced symmetric).
    """
    banks = returns_window.columns
    n = len(banks)
    A = pd.DataFrame(np.zeros((n,n)), index=banks, columns=banks)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            y_i = returns_window.iloc[:, i].dropna().values
            y_j = returns_window.iloc[:, j].dropna().values
            # Align their lengths
            m = min(len(y_i), len(y_j))
            y_i, y_j = y_i[-m:], y_j[-m:]
            if len(y_i) < 10:
                # Not enough data for 10-lag
                continue

            p_val, cqs_dict = stationary_bootstrap_test(
                y_i, y_j, tau, max_lag,
                B=B, alpha=alpha
            )
            if p_val < alpha:
                # sum of all 10 lags
                cq_sum = sum(cqs_dict.values())
                A.iloc[i, j] = cq_sum
    return A

def systemic_risk_score(A, size_vec):
    """
    S(A, c) = c^T * A * c
    where c is the bank size vector.
    """
    # Reindex c to match A's rows
    c = size_vec[A.index]
    c_mat = c.values.reshape(-1,1)
    A_mat = A.values
    val = c_mat.T @ A_mat @ c_mat
    return float(val[0,0])

def network_density(A):
    """
    Off-diagonal fraction of nonzero entries,
    multiplied by 100 for a percent.
    """
    n = A.shape[0]
    arr = A.values
    mask = ~np.eye(n, dtype=bool)  # off-diagonal
    nnz = np.count_nonzero(arr[mask])
    total = n*(n-1)
    return 100.0 * nnz / total


##############################################################################
#  4) ROLLING-WINDOW
##############################################################################
def rolling_systemic_risk(df_ret, size_vec,
                          window_size=364,  # days in the window
                          step=30,         # move 30 days per iteration
                          tau=0.95, alpha=0.05,  # Changed tau to 0.95
                          max_lag=10, B=100):  # Kept B=100
    """
    Rolling adjacency matrix & systemic risk over 'window_size' days,
    stepping forward 'step' days each time.
    """
    dates = df_ret.index
    i_start = 0
    out = []

    while i_start + window_size <= len(dates):
        # The slice of daily data for this window
        window_slice = dates[i_start : i_start + window_size]
        ret_win = df_ret.loc[window_slice]

        # Build adjacency, compute score & density
        A_win = build_adjacency_matrix(ret_win,
                                       tau=tau, alpha=alpha,
                                       max_lag=max_lag, B=B)
        score = systemic_risk_score(A_win, size_vec)
        dens  = network_density(A_win)
        end_date = window_slice[-1]

        out.append((end_date, score, dens))
        i_start += step  # slide forward by 'step' days

    df_out = pd.DataFrame(out, columns=['Date','Score','Density'])
    df_out.set_index('Date', inplace=True)
    return df_out


##############################################################################
#  5) NORMALIZE TO 0–100 USING MAX SCORE OVER ENTIRE PERIOD
##############################################################################
def normalize_0_100(df_risk):
    """
    The paper normalizes by the COVID max. Here we have no 'COVID' in 2020–2024,
    so let's just use the entire sample's maximum as a baseline.
    """
    max_score = df_risk['Score'].max()
    max_dens  = df_risk['Density'].max()
    df_risk['ScoreNorm']   = 100 * df_risk['Score']   / max_score if max_score>0 else df_risk['Score']
    df_risk['DensityNorm'] = 100 * df_risk['Density'] / max_dens  if max_dens>0  else df_risk['Density']
    return df_risk


##############################################################################
#  6) PLOTTING
##############################################################################
def plot_systemic_risk(df_risk):
    """
    Plots ScoreNorm vs. DensityNorm, focusing on 2021–2024.
    """
    df_plot = df_risk[df_risk.index >= '2021-01-01']

    plt.figure(figsize=(12,6))
    plt.plot(df_plot.index, df_plot['ScoreNorm'],  marker='o', color='red',
             label='Systemic Risk Index (95th Quantile)',  linewidth=1, markersize=3)  # Updated label
    plt.plot(df_plot.index, df_plot['DensityNorm'], marker='o', color='blue',
             label='Network Density [%]', linewidth=1, markersize=3)

    # Title
    plt.title("Systemic Risk Score & Network Density (95th Quantile, Rolling 364-day Window)")  # Updated title

    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(YearLocator())            # major ticks each year
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))    # label as 'YYYY'
    plt.xticks(rotation=0)

    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Adjust subplot parameters to give specified padding
    plt.subplots_adjust(right=0.85)  # Reduce right margin to accommodate legend

    plt.grid(True, alpha=0.3)
    
    # Save the figure as PNG with high resolution
    plt.savefig("systemic_risk_95quantile.png", dpi=300, bbox_inches='tight')
    
    plt.show()


##############################################################################
#  7) MAIN SCRIPT
##############################################################################
if __name__ == "__main__":

    # Paths to your CSVs (change to your actual file names)
    path_marketcap = "/home/killshotak/Documents/crossqunatillogram/data_wide_marketcap.csv"
    path_close     = "/home/killshotak/Documents/crossqunatillogram/data_wide_close.csv"

    # 1) Load data
    df_returns, bank_size_vec = load_data(path_marketcap, path_close)

    # 2) Rolling analysis
    #    - 364-day window
    #    - step of 30 days
    #    - 10 lags
    #    - alpha=0.05, B=100
    df_risk = rolling_systemic_risk(
        df_ret=df_returns,
        size_vec=bank_size_vec,
        window_size=364,
        step=30,
        tau=0.95,      # Changed to 95th quantile
        alpha=0.05,
        max_lag=10,
        B=100          # Kept at 100 as requested
    )

    # 3) Normalize
    df_risk = normalize_0_100(df_risk)

    # 4) Save to CSV if desired
    df_risk.to_csv("systemic_risk_output_95quantile.csv")  # Updated filename

    # 5) Plot from 2021 onward
    plot_systemic_risk(df_risk)

    print("\nDone. Results saved to:")
    print("- systemic_risk_output_95quantile.csv (data)")
    print("- systemic_risk_95quantile.png (plot)")