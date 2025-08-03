import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import os
import logging
from joblib import Parallel, delayed

# ------------------------------------------------------------------------
# 1) LOGGING SETUP
# ------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crypto_analysis.log'
)

# ------------------------------------------------------------------------
# 2) CRYPTO CATEGORIES + COLOR MAP
# ------------------------------------------------------------------------
def assign_token_category(token_name: str) -> str:
    """
    Assign each crypto token to one of your predefined categories.
    """
    payment_coins    = ['BTC','BCH','BSV','DASH','DOGE','LTC','XMR','ZEC']
    smart_contract   = ['ADA','ETH','EOS','TRX','NEO','ONT','QTUM','XTZ','ATOM']
    stablecoins      = ['TUSD','USDC','USDP','USDT']
    exchange_tokens  = ['BNB','FTT','OKB','LEO','CRO']
    defi_utility     = ['MKR','LINK','BAT']
    iot              = ['IOTA']
    enterprise       = ['VET']
    privacy          = ['XMR','ZEC']
    other            = ['XEM','XLM','XRP','ETC']

    token = token_name.upper()

    if token in payment_coins:
        return 'Payment'
    elif token in smart_contract:
        return 'SmartContract'
    elif token in stablecoins:
        return 'Stablecoin'
    elif token in exchange_tokens:
        return 'Exchange'
    elif token in defi_utility:
        return 'DeFiUtility'
    elif token in iot:
        return 'IoT'
    elif token in enterprise:
        return 'Enterprise'
    elif token in privacy:
        return 'Privacy'
    elif token in other:
        return 'Other'
    else:
        return 'Misc'

def create_token_groups() -> Dict[str, str]:
    """
    Map each category to a distinct color.
    """
    return {
        'Payment':       '#1f77b4',  # Blue
        'SmartContract': '#ff7f0e',  # Orange
        'Stablecoin':    '#2ca02c',  # Green
        'Exchange':      '#d62728',  # Red
        'DeFiUtility':   '#9467bd',  # Purple
        'IoT':           '#8c564b',  # Brown
        'Enterprise':    '#7f7f7f',  # Gray
        'Privacy':       '#000000',  # Black
        'Other':         '#e377c2',  # Pink
        'Misc':          '#bcbd22'   # Olive
    }

# ------------------------------------------------------------------------
# 3) CROSS‑QUANTILOGRAM UTILS
# ------------------------------------------------------------------------
def CrossQuantilogram(x1, alpha1, x2, alpha2, k):
    """
    Calculate cross-quantilogram ρ between two series at lag k.
    """
    if k == 0:
        array_x1 = np.array(x1)
        array_x2 = np.array(x2)
    elif k > 0:
        array_x1 = np.array(x1[k:])
        array_x2 = np.array(x2[:-k])
    else:  # k < 0
        array_x1 = np.array(x1[:k])
        array_x2 = np.array(x2[-k:])

    if len(array_x2.shape) > 1:
        raise ValueError("x2 must be 1D array.")

    if len(array_x1.shape) == 1:
        array_x1 = array_x1.reshape(-1, 1)

    q1 = np.percentile(array_x1, alpha1*100, axis=0, method='higher')
    q2 = np.percentile(array_x2, alpha2*100, axis=0, method='higher')

    psi1 = (array_x1 < q1) - alpha1
    psi2 = (array_x2 < q2) - alpha2

    numerator = np.sum(psi1 * psi2.reshape(-1, 1), axis=0)
    denominator = np.sqrt(np.sum(psi1**2, axis=0)) * np.sqrt(np.sum(psi2**2))

    if numerator.shape[0] == 1:
        return numerator[0] / denominator[0]
    else:
        return numerator / denominator

def LjungBoxQ(cqlist, maxp, T):
    """
    Ljung-Box type statistic for cross-quantilogram residuals.
    """
    cq = np.array(cqlist[:maxp])
    return T*(T+2)*np.cumsum(np.power(cq,2) / np.arange(T-1, T-maxp-1, -1))

def Bootstrap(x1: np.ndarray, x2: np.ndarray, lag: int, bslength: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple bootstrap for cross-quantilogram inference.
    """
    dtlen = x1.shape[0] - lag
    indices = np.random.randint(0, dtlen, size=bslength)
    return x1[indices], x2[indices]

def CQBS(
    data1: np.ndarray,
    a1: float,
    data2: np.ndarray,
    a2: float,
    k: int,
    cqcl: float = 0.95,
    testf=LjungBoxQ,
    testcl: float = 0.95,
    n: int = 100
):
    """
    Cross-quantilogram with bootstrap for confidence intervals.
    """
    import pandas as pd

    length = len(data1)
    cqdata = np.zeros((n, k))
    qdata  = np.zeros((n, k))

    for i in range(n):
        for lag in range(1, k+1):
            bs1, bs2 = Bootstrap(data1, data2, lag, length)
            cqdata[i, lag-1] = CrossQuantilogram(bs1, a1, bs2, a2, lag)
        qdata[i] = testf(cqdata[i], k, length)

    cqsample = np.array([
        CrossQuantilogram(data1, a1, data2, a2, lag)
        for lag in range(1, k+1)
    ])
    cquc, cqlc = (1 + cqcl)/2, (1 - cqcl)/2

    cq_upper = np.quantile(cqdata, cquc, axis=0)
    cq_lower = np.quantile(cqdata, cqlc, axis=0)
    qc = np.quantile(qdata, testcl, axis=0)

    return pd.DataFrame({
        "cq": cqsample,
        "cq_upper": cq_upper,
        "cq_lower": cq_lower,
        "q": testf(cqsample, k, length),
        "qc": qc
    }, index=list(range(1, k+1)))

def compute_pair_cq(
    data1: np.ndarray,
    data2: np.ndarray,
    quantile: float,
    max_lag: int,
    n_bootstrap: int
) -> Optional[float]:
    """
    Compute the cross-quantilogram for a single pair,
    returning the maximum absolute value over lags [1..max_lag].
    """
    try:
        if data1.shape != data2.shape:
            raise ValueError(f"Shape mismatch: {data1.shape} vs {data2.shape}")

        res = CQBS(data1, quantile, data2, quantile, max_lag, n=n_bootstrap)
        return res['cq'].abs().max()
    except Exception as e:
        logging.error(f"Error in compute_pair_cq: {str(e)}", exc_info=True)
        return None

# ------------------------------------------------------------------------
# 4) SINGLE-EVENT ADJACENCY (NO ROLLING)
# ------------------------------------------------------------------------
def compute_event_adjacency(
    returns_data: pd.DataFrame,
    market_caps: pd.DataFrame,
    quantile: float = 0.95,  # 95th quantile
    max_lag: int = 5,
    n_bootstrap: int = 100,
    threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the entire event dataset (returns_data, market_caps),
    compute a single adjacency matrix (cross-quantilogram > threshold)
    and a weighted adjacency matrix (with market cap weighting).
    """
    tokens = returns_data.columns
    n_tokens = len(tokens)

    data_matrix = returns_data.values
    avg_caps = market_caps.mean()
    total_cap = avg_caps.sum()
    if total_cap == 0:
        cap_weights = np.ones_like(avg_caps)
    else:
        cap_weights = avg_caps / total_cap

    adj_matrix = np.zeros((n_tokens, n_tokens))
    w_matrix = np.zeros((n_tokens, n_tokens))

    token_pairs = [(j, k) for j in range(n_tokens) for k in range(j+1, n_tokens)]

    # Parallel cross-quantilogram
    results = Parallel(n_jobs=-1)(
        delayed(compute_pair_cq)(
            data_matrix[:, j],
            data_matrix[:, k],
            quantile,
            max_lag,
            n_bootstrap
        )
        for (j, k) in token_pairs
    )

    for (j, k), cq_val in zip(token_pairs, results):
        if cq_val is not None and cq_val > threshold:
            adj_matrix[j, k] = cq_val
            adj_matrix[k, j] = cq_val
            w_matrix[j, k] = cq_val * cap_weights.iloc[k]
            w_matrix[k, j] = cq_val * cap_weights.iloc[j]

    return adj_matrix, w_matrix

# ------------------------------------------------------------------------
# 5) DETERMINE TOP-100 THRESHOLD
# ------------------------------------------------------------------------
def threshold_top100_adjacency(
    adjacency: np.ndarray,
    top_n: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Flatten adjacency, remove zeros, sort descending, take top_n, average => threshold.
    Build new NxN adjacency with edges > threshold.
    """
    adj_flat = adjacency.flatten()
    adj_nonzero = adj_flat[adj_flat > 0]

    if len(adj_nonzero) == 0:
        return np.zeros_like(adjacency), 0.0

    if len(adj_nonzero) < top_n:
        logging.warning(f"Fewer than {top_n} edges found. Using all edges to compute threshold.")
        top_values = np.sort(adj_nonzero)[::-1]
        threshold_val = np.mean(top_values)
    else:
        sorted_vals = np.sort(adj_nonzero)[::-1]
        top_values = sorted_vals[:top_n]
        threshold_val = np.mean(top_values)

    new_adj = np.where(adjacency > threshold_val, adjacency, 0.0)
    return new_adj, threshold_val

# ------------------------------------------------------------------------
# 6) CREATE THRESHOLDED NETWORK
# ------------------------------------------------------------------------
def create_thresholded_spillover_network(
    thresholded_adj: np.ndarray,
    weighted_matrix: np.ndarray,
    returns_data: pd.DataFrame,
    market_caps: pd.DataFrame
) -> nx.DiGraph:
    """
    Build a directed graph from thresholded adjacency.
    """
    G = nx.DiGraph()
    tokens = returns_data.columns.tolist()
    n_tokens = len(tokens)

    avg_caps = market_caps.mean()
    total_cap = avg_caps.sum()
    if total_cap == 0:
        rel_caps = np.ones_like(avg_caps)
    else:
        rel_caps = avg_caps / total_cap

    for i, token in enumerate(tokens):
        category = assign_token_category(token)
        G.add_node(
            token,
            category=category,
            market_cap=avg_caps.iloc[i],
            relative_size=rel_caps.iloc[i]
        )

    for i in range(n_tokens):
        for j in range(n_tokens):
            if thresholded_adj[i, j] > 0:
                w_spill = abs(weighted_matrix[i, j])
                raw_val = thresholded_adj[i, j]
                G.add_edge(
                    tokens[i],
                    tokens[j],
                    raw_weight=raw_val,
                    weighted_spillover=w_spill
                )

    return G

# ------------------------------------------------------------------------
# 7) PLOTTING UTILS: CLUSTERED NETWORK ON A GIVEN AXIS
# ------------------------------------------------------------------------
def plot_network_on_axis(
    G: nx.DiGraph,
    ax: plt.Axes,
    title: str = ""
):
    """
    Plots the network on the given Axes 'ax' in a circular cluster layout.
    No legend is drawn here, so we can combine a single legend for all subplots later.
    """
    color_map = create_token_groups()

    # 1. Group nodes by category
    categories = {}
    for node, attrs in G.nodes(data=True):
        cat = attrs['category']
        categories.setdefault(cat, []).append(node)

    # 2. Circular layout by category
    pos = {}
    sorted_cats = sorted(categories.keys())
    num_cats = len(sorted_cats)

    # Increase if needed
    outer_radius = 7.0
    category_angles = np.linspace(0, 2*np.pi, num_cats, endpoint=False)

    for idx, cat in enumerate(sorted_cats):
        cat_nodes = categories[cat]
        num_nodes = len(cat_nodes)

        cat_angle = category_angles[idx]
        cat_center_x = outer_radius * np.cos(cat_angle)
        cat_center_y = outer_radius * np.sin(cat_angle)

        # Slightly bigger inner circle if needed
        inner_radius = 1.5
        if num_nodes == 1:
            pos[cat_nodes[0]] = (cat_center_x, cat_center_y)
        else:
            node_angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
            for i, node in enumerate(cat_nodes):
                node_angle = node_angles[i]
                nx_coord = cat_center_x + inner_radius * np.cos(node_angle)
                ny_coord = cat_center_y + inner_radius * np.sin(node_angle)
                pos[node] = (nx_coord, ny_coord)

    # 3. Draw nodes
    BASE_NODE_SIZE = 650
    for cat in sorted_cats:
        cat_nodes = categories[cat]
        color = color_map.get(cat, '#cccccc')
        node_sizes = [
            BASE_NODE_SIZE + (G.nodes[n]['relative_size'] * 2700)
            for n in cat_nodes
        ]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=cat_nodes,
            node_color=color,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )

    # 4. Draw edges
    MIN_EDGE_WIDTH = 0.5
    MAX_EDGE_WIDTH = 3.0
    edges_data = G.edges(data=True)
    weights = [d['weighted_spillover'] for (_, _, d) in edges_data]
    max_w = max(weights) if weights else 1.0

    for (u, v, d) in edges_data:
        w_norm = d['weighted_spillover'] / max_w if max_w else 0
        edge_width = MIN_EDGE_WIDTH + (MAX_EDGE_WIDTH * w_norm)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=edge_width,
            edge_color='green',
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.2',
            ax=ax
        )

    # 5. Draw labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_color='black', ax=ax)

    # 6. Title
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()

# ------------------------------------------------------------------------
# 8) MAIN FUNCTION
# ------------------------------------------------------------------------
def main():
    """
    Main flow:
      1) Load data
      2) For each of the 5 events, slice data
      3) Compute single adjacency with 95th quantile
      4) Threshold top-100
      5) Create network
      6) Plot all subplots in a single figure (2 columns, 3 rows => 5 used + 1 blank)
      7) Single legend on the right, outside the plot area
      8) Save a high-resolution PNG
    """
    try:
        # 1) Load data
        close_prices = pd.read_csv(
            '/home/killshotak/Documents/crossqunatillogram/data_wide_close.csv',
            parse_dates=['Date'],
            index_col='Date'
        )
        market_caps = pd.read_csv(
            '/home/killshotak/Documents/crossqunatillogram/data_wide_marketcap.csv',
            parse_dates=['Date'],
            index_col='Date'
        )

        # 2) Compute log returns
        log_returns = np.log(close_prices).diff().dropna()
        log_returns = log_returns.replace('NA', np.nan).ffill().dropna()
        market_caps = market_caps.replace('NA', np.nan).ffill()

        common_dates = log_returns.index.intersection(market_caps.index)
        log_returns = log_returns.loc[common_dates]
        market_caps = market_caps.loc[common_dates]

        # Define the 5 events
        events = {
            'COVID19':         ('2020-03-11', '2020-12-31'),
            'RussiaUkraine':   ('2022-02-24', '2022-12-31'),
            'TeslaWithdrawal': ('2021-03-24', '2021-07-21'),
            'SVBCrisis':       ('2023-03-01', '2023-03-31'),
            'IsraelPalestine': ('2023-10-07', None)
        }

        # 3) For each event, compute adjacency and build network
        event_graphs = {}
        for event_name, (start_date, end_date) in events.items():
            logging.info(f"Analyzing event: {event_name}")

            event_start_dt = pd.to_datetime(start_date)
            if end_date:
                event_end_dt = pd.to_datetime(end_date)
            else:
                event_end_dt = log_returns.index[-1]

            mask = (log_returns.index >= event_start_dt) & (log_returns.index <= event_end_dt)
            event_returns = log_returns[mask]
            event_caps = market_caps[mask]

            if len(event_returns) < 2:
                logging.warning(f"Skipping {event_name} — not enough data.")
                continue

            adj_matrix, w_matrix = compute_event_adjacency(
                returns_data=event_returns,
                market_caps=event_caps,
                quantile=0.95,
                max_lag=5,
                n_bootstrap=100,
                threshold=0.05
            )
            thres_adj, _ = threshold_top100_adjacency(adj_matrix, top_n=100)
            G = create_thresholded_spillover_network(thres_adj, w_matrix, event_returns, event_caps)
            event_graphs[event_name] = G

        # 4) Plot in 2x3 grid (use (16,18) for bigger subplots)
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
        all_event_names = list(event_graphs.keys())
        color_map = create_token_groups()

        for i, event_name in enumerate(all_event_names):
            row = i // 2
            col = i % 2
            G = event_graphs[event_name]
            ax = axes[row, col]
            plot_network_on_axis(G, ax, title=event_name)

        # Hide any unused subplot (if fewer than 6 events)
        if len(all_event_names) < 6:
            axes[2,1].set_axis_off()

        # 5) Adjust margins so there's space on right for the legend
        #    place the legend outside the axes at (0.82, 0.5).
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.80, top=0.95, 
                            wspace=0.30, hspace=0.30)

        # Create a single legend
        legend_elems = []
        for cat, c in color_map.items():
            legend_elems.append(
                plt.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    label=cat,
                    markerfacecolor=c,
                    markersize=10
                )
            )

        fig.legend(
            handles=legend_elems,
            loc='center left',
            bbox_to_anchor=(0.82, 0.5),
            fontsize=10,
            title='Categories'
        )

        # 6) Save as high-resolution PNG
        os.makedirs('crypto_spillover_results', exist_ok=True)
        outpath = 'crypto_spillover_results/multi_event_networks_noOverlap.png'
        fig.savefig(outpath, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close(fig)

        logging.info("All event analyses completed successfully.")
        return event_graphs

    except Exception as e:
        logging.error(f"Main function error: {str(e)}", exc_info=True)
        raise

# ------------------------------------------------------------------------
# RUN SCRIPT
# ------------------------------------------------------------------------
if __name__ == "__main__":
    final_graphs = main()
    print("All analyses completed and multi-event figure saved at high resolution.")
