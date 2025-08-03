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
def cross_quantilogram(y1, y2, tau=0.05, max_lag=10):
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
        
        # For tau < 0.5, use lower quantile logic
        # For tau >= 0.5, use upper quantile logic
        if tau < 0.5:
            psi1 = (y1_lag < q1).astype(float) - tau
            psi2 = (y2_lag < q2).astype(float) - tau
        else:
            psi1 = (y1_lag > q1).astype(float) - (1-tau)
            psi2 = (y2_lag > q2).astype(float) - (1-tau)
            
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

def stationary_bootstrap_test(y1, y2, tau=0.05, max_lag=10,
                              B=100, alpha=0.05):
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
def build_adjacency_matrix(returns_window, tau=0.05, alpha=0.05,
                           max_lag=10, B=100):
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
                          tau=0.05, alpha=0.05,
                          max_lag=10, B=100):
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
#  6) MAIN COMPUTATION FOR BOTH QUANTILES AND DELTA
##############################################################################
def compute_delta_systemic_risk(path_marketcap, path_close):
    """
    Compute systemic risk for both 5th and 95th quantiles, then calculate delta.
    """
    # Load data
    df_returns, bank_size_vec = load_data(path_marketcap, path_close)
    
    # Compute 5th quantile systemic risk
    print("Computing 5th quantile systemic risk...")
    df_risk_5th = rolling_systemic_risk(
        df_ret=df_returns,
        size_vec=bank_size_vec,
        window_size=364,
        step=30,
        tau=0.05,      # 5th quantile
        alpha=0.05,
        max_lag=10,
        B=100
    )
    df_risk_5th = normalize_0_100(df_risk_5th)
    
    # Compute 95th quantile systemic risk
    print("Computing 95th quantile systemic risk...")
    df_risk_95th = rolling_systemic_risk(
        df_ret=df_returns,
        size_vec=bank_size_vec,
        window_size=364,
        step=30,
        tau=0.95,      # 95th quantile
        alpha=0.05,
        max_lag=10,
        B=100
    )
    df_risk_95th = normalize_0_100(df_risk_95th)
    
    # Align dates (in case there are any differences)
    common_dates = df_risk_5th.index.intersection(df_risk_95th.index)
    df_risk_5th_aligned = df_risk_5th.loc[common_dates]
    df_risk_95th_aligned = df_risk_95th.loc[common_dates]
    
    # Compute delta (95th - 5th)
    df_delta = pd.DataFrame(index=common_dates)
    df_delta['Delta_TCI'] = df_risk_95th_aligned['ScoreNorm'] - df_risk_5th_aligned['ScoreNorm']
    df_delta['Reverse_TCI'] = df_risk_95th_aligned['ScoreNorm']  # 95th quantile (reverse/upper tail)
    df_delta['Direct_TCI'] = df_risk_5th_aligned['ScoreNorm']   # 5th quantile (direct/lower tail)
    
    return df_delta


##############################################################################
#  7) PLOTTING IN REFERENCE STYLE
##############################################################################
def plot_delta_systemic_risk(df_delta):
    """
    Plot delta systemic risk in the style of the reference image.
    """
    # Filter to 2021 onwards for better visualization
    df_plot = df_delta[df_delta.index >= '2021-01-01'].copy()
    
    # Set up the plot with reference style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the three lines with colors matching reference image
    ax.plot(df_plot.index, df_plot['Reverse_TCI'], 
            color='#FF6B6B', linewidth=1.5, alpha=0.8, label='Reverse TCI')  # Red/pink
    ax.plot(df_plot.index, df_plot['Direct_TCI'], 
            color='#4ECDC4', linewidth=1.5, alpha=0.8, label='Direct TCI')   # Teal/green
    ax.plot(df_plot.index, df_plot['Delta_TCI'], 
            color='#45B7D1', linewidth=2, alpha=0.9, label='Δ TCI')         # Blue
    
    # Add horizontal line at y=0 for delta reference
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    
    # Styling to match reference
    ax.set_facecolor('#F8F9FA')
    ax.grid(True, alpha=0.3, color='white', linewidth=1)
    
    # Set title and labels
    ax.set_title('Fig. 8. Dynamic directly related and reversely related quantiles risk transmission.', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    # Set y-axis range similar to reference (roughly 0-90)
    y_min = min(df_plot[['Reverse_TCI', 'Direct_TCI', 'Delta_TCI']].min()) - 5
    y_max = max(df_plot[['Reverse_TCI', 'Direct_TCI', 'Delta_TCI']].max()) + 5
    ax.set_ylim(y_min, y_max)
    
    # Add legend in bottom right
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, 
              framealpha=0.9, fontsize=11)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    
    plt.tight_layout()
    
    # Save high-quality figure
    plt.savefig("delta_systemic_risk_transmission.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.show()


##############################################################################
#  8) MAIN EXECUTION
##############################################################################
if __name__ == "__main__":
    # Update these paths to your actual CSV files
    path_marketcap = "data_wide_marketcap.csv"  # Update this path
    path_close     = "data_wide_close.csv"      # Update this path
    
    # Compute delta systemic risk
    df_delta = compute_delta_systemic_risk(path_marketcap, path_close)
    
    # Save results
    df_delta.to_csv("delta_systemic_risk_output.csv")
    
    # Create the plot
    plot_delta_systemic_risk(df_delta)
    
    print("\nDone! Results saved to:")
    print("- delta_systemic_risk_output.csv (data)")
    print("- delta_systemic_risk_transmission.png (plot)")
    print(f"\nDelta TCI statistics:")
    print(f"Mean: {df_delta['Delta_TCI'].mean():.2f}")
    print(f"Std: {df_delta['Delta_TCI'].std():.2f}")
    print(f"Min: {df_delta['Delta_TCI'].min():.2f}")
    print(f"Max: {df_delta['Delta_TCI'].max():.2f}")