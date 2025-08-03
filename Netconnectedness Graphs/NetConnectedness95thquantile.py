import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from matplotlib.colors import LinearSegmentedColormap

#######################################################
#  1) Visualization Setup
#######################################################

# Custom yellow->orange->red colormap
colors = [
    (255/255, 255/255,  0/255),    # bright yellow (#ffff00)
    (255/255, 165/255,  0/255),    # orange (#ffa500)
    (255/255, 69/255,   0/255),    # orange-red (#ff4500)
    (139/255,  0/255,   0/255)     # dark red (#8b0000)
]
yellow_red = LinearSegmentedColormap.from_list(
    "yellow_red", colors, N=256
)

# Set global plotting parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.weight': 'medium',
    'axes.labelweight': 'medium',
    'figure.figsize': (15, 12)  # Ensure figures are large enough by default
})

#######################################################
#  2) Dual Scaling Function
#######################################################
def dual_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced dual scaling with improved handling of edge cases.
    Scales negative values to at most -100 and positive values to at most 100.
    Leaves zeros at 0.
    """
    df_scaled = df.copy()

    # Handle zeros separately
    zero_mask = (df == 0)

    # Scale positive values
    pos_mask = (df > 0)
    if pos_mask.any().any():
        # Get the maximum positive value - this should be a scalar
        # Use .max().max() to get a scalar from a DataFrame
        pos_max = df[pos_mask].max().max()
        if pos_max > 0:  # Avoid division by zero
            df_scaled[pos_mask] = df_scaled[pos_mask] * (100 / pos_max)

    # Scale negative values
    neg_mask = (df < 0)
    if neg_mask.any().any():
        # Get the minimum negative value - this should be a scalar
        # Use .min().min() to get a scalar from a DataFrame
        neg_min = df[neg_mask].min().min()
        if neg_min < 0:  # Avoid division by zero
            df_scaled[neg_mask] = df_scaled[neg_mask] * (100 / abs(neg_min))

    # Ensure bounds
    df_scaled = df_scaled.clip(-100, 100)

    # Keep zeros as zeros
    df_scaled[zero_mask] = 0

    return df_scaled

#######################################################
#  3) Plotting Functions
#######################################################
def plot_adjacency_matrix(adjacency_df, title="Adjacency Matrix", filename=None):
    """
    Plot the full adjacency matrix as a heatmap.
    
    Parameters:
    adjacency_df: DataFrame with the adjacency matrix
    title: Plot title
    filename: If provided, save plot to this filename as PNG
    """
    n_rows, n_cols = adjacency_df.shape
    figsize = (max(15, n_cols * 0.8), max(12, n_rows * 0.8))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use yellow_red colormap for consistency with the desired appearance
    sns.heatmap(
        adjacency_df,
        cmap=yellow_red,
        vmin=-100,
        vmax=100,
        annot=True,
        fmt=".1f",
        annot_kws={
            "size": 11,
            "weight": "bold",
            "ha": "center",
            "va": "center"
        },
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        square=True,
        cbar_kws={
            "shrink": 0.8,
            "aspect": 20,
            "orientation": "vertical"
        }
    )
    
    # Title formatting
    ax.set_title(
        title,
        fontsize=18,
        pad=20,
        weight='bold'
    )
    
    # Labels
    ax.set_xlabel("Target Token", fontsize=14, labelpad=10)
    ax.set_ylabel("Source Token", fontsize=14, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Colorbar customization
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.set_ticklabels(["-100", "-50", "0", "50", "100"])
    cbar.set_label(
        "Effect Strength",
        rotation=270,
        labelpad=20,
        fontsize=14,
        weight='bold'
    )
    
    plt.tight_layout()
    
    # Save figure to file if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
        
    return fig, ax

def plot_net_effect(df_net_effect, title="Net Effect", filename=None):
    """
    Plot a single-column net effect visualization.
    
    Parameters:
    df_net_effect: DataFrame with net effect values
    title: Plot title
    filename: If provided, save plot to this filename as PNG
    """
    n_rows = len(df_net_effect)
    
    # Adjust figsize based on number of rows
    figsize = (10, max(12, n_rows * 0.6))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # If df_net_effect is a Series, convert to DataFrame
    if isinstance(df_net_effect, pd.Series):
        df_net_effect = df_net_effect.to_frame()
    
    sns.heatmap(
        df_net_effect,
        cmap=yellow_red,
        vmin=-100,
        vmax=100,
        annot=True,
        fmt=".1f",
        annot_kws={
            "size": 12,
            "weight": "bold",
            "ha": "center",
            "va": "center"
        },
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        cbar_kws={
            "shrink": 0.8,
            "aspect": 20,
            "orientation": "vertical"
        }
    )
    
    # Title formatting
    ax.set_title(
        title,
        fontsize=16,
        pad=20,
        weight='bold'
    )
    
    # Labels
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Token", fontsize=14, labelpad=10)
    
    plt.yticks(fontsize=12, rotation=0)
    
    # Colorbar customization
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.set_ticklabels(["-100", "-50", "0", "50", "100"])
    cbar.set_label(
        "Net Effect",
        rotation=270,
        labelpad=20,
        fontsize=14,
        weight='bold'
    )
    
    plt.tight_layout()
    
    # Save figure to file if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
        
    return fig, ax

def plot_rolling_net_effect(df_net_effect, title="Rolling Net Effect", filename=None):
    """
    Enhanced plot for rolling net effect matrix with improved formatting.
    
    Parameters:
    df_net_effect: DataFrame with rolling net effect values
    title: Plot title
    filename: If provided, save plot to this filename as PNG
    """
    n_rows, n_cols = df_net_effect.shape
    
    # Larger figure size for rolling visualization
    figsize = (max(20, n_cols * 0.6), max(15, n_rows * 0.8))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        df_net_effect,
        cmap=yellow_red,
        vmin=-100,
        vmax=100,
        annot=True,
        fmt=".1f",
        annot_kws={
            "size": 10,
            "weight": "bold",
            "ha": "center",
            "va": "center"
        },
        linewidths=1,
        linecolor="white",
        ax=ax,
        cbar_kws={
            "shrink": 0.8,
            "aspect": 20,
            "orientation": "vertical"
        }
    )
    
    # Title formatting
    ax.set_title(
        title,
        fontsize=18,
        pad=20,
        weight='bold'
    )
    
    # Labels
    ax.set_xlabel("Window End Date", fontsize=14, labelpad=15)
    ax.set_ylabel("Token", fontsize=14, labelpad=15)
    
    # Format x-axis to show only years
    if isinstance(df_net_effect.columns[0], pd.Timestamp):
        years = sorted(set(date.year for date in df_net_effect.columns))
        year_positions = {}
        
        # Find positions for year labels
        for year in years:
            year_dates = [i for i, date in enumerate(df_net_effect.columns) 
                         if date.year == year]
            if year_dates:
                # Place label in middle of year's range
                year_positions[year] = year_dates[len(year_dates)//2]
        
        tick_positions = [pos + 0.5 for pos in year_positions.values()]
        tick_labels = [str(year) for year in year_positions.keys()]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels,
            rotation=0,
            ha='center',
            fontsize=11
        )
    else:
        plt.xticks(rotation=45, ha='right', fontsize=10)
    
    plt.yticks(fontsize=12, rotation=0)
    
    # Colorbar customization
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.set_ticklabels(["-100", "-50", "0", "50", "100"])
    cbar.set_label(
        "Net Effect",
        rotation=270,
        labelpad=20,
        fontsize=14,
        weight='bold'
    )
    
    plt.tight_layout()
    
    # Save figure to file if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
        
    return fig, ax

#######################################################
#  4) Analysis Functions
#######################################################
def cross_quantilogram(y1, y2, tau=0.95, max_lag=10):
    """
    Compute cross-quantilograms up to max_lag for two arrays y1,y2 at quantile tau.
    Now defaults to tau=0.95 for 95th quantile analysis.
    """
    y1 = np.array(y1)
    y2 = np.array(y2)
    T  = len(y1)
    if T != len(y2):
        raise ValueError("Series must have same length.")
    q1 = np.quantile(y1, tau)
    q2 = np.quantile(y2, tau)

    psi1 = ((y1 - q1) < 0).astype(float) - tau
    psi2 = ((y2 - q2) < 0).astype(float) - tau

    denom_left  = np.sum(psi1**2)
    denom_right = np.sum(psi2**2)
    # Avoid divide-by-zero
    denom = np.sqrt(denom_left * denom_right) if (denom_left > 0 and denom_right > 0) else 1e-12

    results = {}
    for k in range(1, max_lag+1):
        numerator = 0.0
        for t in range(k, T):
            numerator += psi1[t] * psi2[t-k]
        cq_val = numerator / denom
        # p-values are handled via bootstrap test, so store np.nan for now
        results[k] = (cq_val, np.nan)
    return results

def stationary_bootstrap(series, block_prob=0.1):
    """
    Sample from 'series' using stationary (Politis-Romano) bootstrap.
    """
    n = len(series)
    out = []
    current_idx = random.randint(0, n-1)
    for _ in range(n):
        out.append(series[current_idx])
        if random.random() < block_prob:
            current_idx = random.randint(0, n-1)
        else:
            current_idx = (current_idx + 1) % n
    return np.array(out)

def stationary_bootstrap_test(y1, y2, tau=0.95, max_lag=10, B=200, block_prob=0.1):
    """
    Ljung-Box style significance test for cross-quantilogram via sum-of-squares.
    Now defaults to tau=0.95 for 95th quantile analysis.
    """
    cqs_actual = cross_quantilogram(y1, y2, tau, max_lag)
    stat_actual = sum(val[0]**2 for val in cqs_actual.values())

    cqs_dist = np.zeros(B)
    for b_ in range(B):
        y1_star = stationary_bootstrap(y1, block_prob)
        y2_star = stationary_bootstrap(y2, block_prob)
        cqs_star = cross_quantilogram(y1_star, y2_star, tau, max_lag)
        stat_b = sum(vv[0]**2 for vv in cqs_star.values())
        cqs_dist[b_] = stat_b

    p_val = np.mean(cqs_dist >= stat_actual)
    return p_val, cqs_actual

def build_adjacency_matrix(returns_data, tau=0.95, max_lag=10,
                           alpha=0.05, B=200, block_prob=0.1):
    """
    Build NxN adjacency matrix from returns_data using cross-quantilogram & stationary bootstrap.
    Now defaults to tau=0.95 for 95th quantile analysis.
    """
    tokens = returns_data.columns
    n = len(tokens)
    A = pd.DataFrame(np.zeros((n, n)), index=tokens, columns=tokens)

    for i in range(n):
        for j in range(n):
            if i == j:
                A.iloc[i,j] = 0.0
                continue
            y1 = returns_data[tokens[i]].dropna().values
            y2 = returns_data[tokens[j]].dropna().values
            length = min(len(y1), len(y2))
            y1 = y1[-length:]
            y2 = y2[-length:]

            p_val, cqs_actual = stationary_bootstrap_test(
                y1, y2, tau, max_lag, B=B, block_prob=block_prob
            )
            # If significant, store lag=1 value; else zero
            if p_val < alpha:
                lag1_val = cqs_actual[1][0]
                A.iloc[i,j] = lag1_val
            else:
                A.iloc[i,j] = 0.0
    return A

def compute_net_effect(A):
    """
    Compute net_effect(i) = sum(row i) - sum(col i).
    """
    row_sum = A.sum(axis=1)
    col_sum = A.sum(axis=0)
    return row_sum - col_sum

def rolling_net_effect(returns_data, window_size=250, step=50,
                       tau=0.95, max_lag=10, alpha=0.05,
                       B=50, block_prob=0.1):
    """
    Calculate rolling net effect with specified window size and step.
    Now defaults to tau=0.95 for 95th quantile analysis.
    """
    dates = returns_data.index
    net_effect_dict = {}
    start_idx = 0

    while start_idx + window_size <= len(dates):
        window_slice = dates[start_idx : start_idx+window_size]
        ret_window = returns_data.loc[window_slice]

        A_win = build_adjacency_matrix(
            ret_window,
            tau=tau,
            max_lag=max_lag,
            alpha=alpha,
            B=B,
            block_prob=block_prob
        )
        net_eff = compute_net_effect(A_win)
        end_date = window_slice[-1]
        net_effect_dict[end_date] = net_eff

        start_idx += step

    return pd.DataFrame(net_effect_dict)

#######################################################
#  5) Main Program
#######################################################
def main():
    """
    Example main function that reads CSVs,
    computes adjacency, net effect for 95th quantile, applies dual_scale, and plots.
    """
    # ----------------------------------------------------------------------------
    # 1) Specify your CSV file paths
    # ----------------------------------------------------------------------------
    path_marketcap = 'data_wide_marketcap.csv'  # Update with your actual path
    path_returns   = 'data_wide_close.csv'      # Update with your actual path

    # ----------------------------------------------------------------------------
    # 2) Define a rename map IF your raw CSV columns differ from canonical names
    # ----------------------------------------------------------------------------
    rename_map = {
        "ADA ": "ADA",
        "ATOM ": "ATOM",
        "BAT ": "BAT",
        "BCH": "BCH",
        "BNB": "BNB",
        "BSV": "BSV",
        "BTC": "BTC",
        "CRO": "CRO",
        "DASH": "DASH",
        "DCR": "DCR",
        "DOGE": "DOGE",
        "EOS": "EOS",
        "ETC": "ETC",
        "ETH": "ETH",
        "FTT": "FTT",
        "IOTA": "IOTA",
        "LEO": "LEO",
        "LINK": "LINK",
        "LTC": "LTC",
        "MKR": "MKR",
        "NEO": "NEO",
        "OKB": "OKB",
        "ONT": "ONT",
        "QTUM": "QTUM",
        "TRX": "TRX",
        "TUSD": "TUSD",
        "USDC": "USDC",
        "USDP": "USDP",
        "USDT": "USDT",
        "VET": "VET",
        "XEM": "XEM",
        "XLM": "XLM",
        "XMR": "XMR",
        "XRP": "XRP",
        "XTZ": "XTZ",
        "ZEC": "ZEC"
    }

    # ----------------------------------------------------------------------------
    # 3) Read CSVs and parse dates in mm/dd/yyyy format (dayfirst=False)
    # ----------------------------------------------------------------------------
    print("Reading CSV files...")
    try:
        df_marketcap = pd.read_csv(
            path_marketcap,
            parse_dates=['Date'],
            index_col='Date',
            dayfirst=False
        )
        df_returns = pd.read_csv(
            path_returns,
            parse_dates=['Date'],
            index_col='Date',
            dayfirst=False
        )
        print(f"Successfully read data: {len(df_returns)} rows")
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        print("Please check file paths and try again.")
        return

    # ----------------------------------------------------------------------------
    # 4) Rename columns if necessary
    # ----------------------------------------------------------------------------
    df_marketcap.rename(columns=rename_map, inplace=True)
    df_returns.rename(columns=rename_map, inplace=True)

    # ----------------------------------------------------------------------------
    # 5) Align dates
    # ----------------------------------------------------------------------------
    common_dates = df_marketcap.index.intersection(df_returns.index)
    df_marketcap = df_marketcap.loc[common_dates]
    df_returns   = df_returns.loc[common_dates]

    if df_marketcap.empty or df_returns.empty:
        print("No overlapping data. Check your CSVs or date parsing.")
        return

    # ----------------------------------------------------------------------------
    # 6) Full Sample Analysis (NOW FOR 95TH QUANTILE)
    # ----------------------------------------------------------------------------
    print("=== Full Sample Adjacency Calculation (95th Quantile) ===")
    print(f"Analyzing {len(df_returns.columns)} tokens at 95th quantile...")
    
    A_full = build_adjacency_matrix(
        df_returns,
        tau=0.95,        # Changed from 0.05 to 0.95
        max_lag=10,
        alpha=0.05,
        B=50,
        block_prob=0.1
    )

    # Calculate net effect
    net_effect_series = compute_net_effect(A_full)
    df_net_effect_full = net_effect_series.to_frame(name="FullSample_95th")

    # Apply dual scaling to map negative side to -100 and positive side to +100
    print("Applying dual scaling...")
    df_net_effect_full_scaled = dual_scale(df_net_effect_full)
    A_full_scaled = dual_scale(A_full)

    # ----------------------------------------------------------------------------
    # 7) Plotting (Updated titles to reflect 95th quantile)
    # ----------------------------------------------------------------------------
    print("Generating plots...")
    
    # Plot the full sample net effect as a single column
    fig1, ax1 = plot_net_effect(
        df_net_effect_full_scaled,
        "Net Effect (Transmission - Reception)\n(Full Sample, 95th Quantile, Dual-Scaled)",
        filename="full_sample_net_effect_95th.png"
    )
    
    # Plot the full adjacency matrix
    fig_adj, ax_adj = plot_adjacency_matrix(
        A_full_scaled,
        "Adjacency Matrix (Full Sample, 95th Quantile, Dual-Scaled)",
        filename="full_sample_adjacency_matrix_95th.png"
    )

    # ----------------------------------------------------------------------------
    # 8) Rolling Analysis (Optional, can be slow with large datasets)
    # ----------------------------------------------------------------------------
    print("=== Rolling Net Effect Calculation (95th Quantile) ===")
    print("This may take some time for large datasets...")
    
    # Comment out this section if it's taking too long
    df_net_effect_roll = rolling_net_effect(
        df_returns,
        window_size=250,  # ~1 year of trading days
        step=50,          # Move forward ~2 months at a time
        tau=0.95,         # Changed from 0.05 to 0.95 - 95% quantile (upper tail risk)
        max_lag=10,       # Look back up to 10 days
        alpha=0.05,       # 5% significance
        B=50,             # 50 bootstrap iterations (increase for more accuracy)
        block_prob=0.1    # Block length ~10 days on average
    )

    # Apply dual scaling to rolling net effect
    if not df_net_effect_roll.empty:
        df_net_effect_roll_scaled = dual_scale(df_net_effect_roll)
        
        # Plot the rolling net effect for first window
        fig2, ax2 = plot_net_effect(
            df_net_effect_roll_scaled.iloc[:, 0].to_frame(),
            f"Rolling Net Effect - 95th Quantile ({df_net_effect_roll.columns[0].strftime('%Y-%m-%d')})",
            filename="first_window_net_effect_95th.png"
        )
        
        # Plot the full rolling visualization (with all windows)
        fig_roll, ax_roll = plot_rolling_net_effect(
            df_net_effect_roll_scaled,
            "Rolling Net Effect (All Windows, 95th Quantile, Dual-Scaled)",
            filename="rolling_net_effect_matrix_95th.png"
        )
        
        print(f"Created {len(df_net_effect_roll.columns)} rolling windows for 95th quantile analysis")
    else:
        print("No rolling windows were calculated. Check your data length.")

    print("95th quantile analysis complete!")
    return A_full_scaled, df_net_effect_full_scaled, df_net_effect_roll_scaled if 'df_net_effect_roll_scaled' in locals() else None

if __name__ == "__main__":
    main()