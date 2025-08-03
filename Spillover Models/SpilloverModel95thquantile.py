import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
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
# 2) CRYPTO CATEGORIES
# ------------------------------------------------------------------------
def assign_token_category(token_name: str) -> str:
    """Assign each crypto token to predefined categories."""
    payment_coins = ['BTC','BCH','BSV','DASH','DOGE','LTC','XMR','ZEC']
    smart_contract = ['ADA','ETH','EOS','TRX','NEO','ONT','QTUM','XTZ','ATOM']
    stablecoins = ['TUSD','USDC','USDP','USDT']
    exchange_tokens = ['BNB','FTT','OKB','LEO','CRO']
    defi_utility = ['MKR','LINK','BAT']
    iot = ['IOTA']
    enterprise = ['VET']
    privacy = ['XMR','ZEC']
    other = ['XEM','XLM','XRP','ETC']

    token = token_name.upper()

    if token in payment_coins: return 'Payment'
    elif token in smart_contract: return 'SmartContract'
    elif token in stablecoins: return 'Stablecoin'
    elif token in exchange_tokens: return 'Exchange'
    elif token in defi_utility: return 'DeFiUtility'
    elif token in iot: return 'IoT'
    elif token in enterprise: return 'Enterprise'
    elif token in privacy: return 'Privacy'
    elif token in other: return 'Other'
    else: return 'Misc'

def create_token_groups() -> Dict[str, str]:
    """Map categories to colors."""
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
    """Calculate cross-quantilogram ρ between two series at lag k."""
    if k == 0:
        array_x1, array_x2 = np.array(x1), np.array(x2)
    elif k > 0:
        array_x1, array_x2 = np.array(x1[k:]), np.array(x2[:-k])
    else:
        array_x1, array_x2 = np.array(x1[:k]), np.array(x2[-k:])

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

    return numerator[0] / denominator[0] if numerator.shape[0] == 1 else numerator / denominator

def LjungBoxQ(cqlist, maxp, T):
    """Ljung-Box type statistic for cross-quantilogram residuals."""
    cq = np.array(cqlist[:maxp])
    return T*(T+2)*np.cumsum(np.power(cq,2) / np.arange(T-1, T-maxp-1, -1))

def Bootstrap(x1: np.ndarray, x2: np.ndarray, lag: int, bslength: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simple bootstrap for cross-quantilogram inference."""
    dtlen = x1.shape[0] - lag
    indices = np.random.randint(0, dtlen, size=bslength)
    return x1[indices], x2[indices]

def CQBS(data1, a1, data2, a2, k, cqcl=0.95, testf=LjungBoxQ, testcl=0.95, n=100):
    """Cross-quantilogram with bootstrap for confidence intervals."""
    length = len(data1)
    cqdata = np.zeros((n, k))
    qdata  = np.zeros((n, k))

    for i in range(n):
        for lag in range(1, k+1):
            bs1, bs2 = Bootstrap(data1, data2, lag, length)
            cqdata[i, lag-1] = CrossQuantilogram(bs1, a1, bs2, a2, lag)
        qdata[i] = testf(cqdata[i], k, length)

    cqsample = np.array([CrossQuantilogram(data1, a1, data2, a2, lag) for lag in range(1, k+1)])
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

def compute_pair_cq(data1: np.ndarray,
                    data2: np.ndarray,
                    quantile: float,
                    max_lag: int,
                    n_bootstrap: int) -> Optional[float]:
    """Compute cross-quantilogram for a single pair."""
    try:
        if data1.shape != data2.shape:
            raise ValueError(f"Shape mismatch: {data1.shape} vs {data2.shape}")
        res = CQBS(data1, quantile, data2, quantile, max_lag, n=n_bootstrap)
        return res['cq'].abs().max()
    except Exception as e:
        logging.error(f"Error in compute_pair_cq: {str(e)}", exc_info=True)
        return None

# ------------------------------------------------------------------------
# 4) ANALYZE NETWORK
# ------------------------------------------------------------------------
def analyze_network(
    returns_data: pd.DataFrame,
    market_caps: pd.DataFrame,
    window_size: int = 364,
    quantile: float = 0.95,  # Updated to 95th percentile
    max_lag: int = 5,
    n_bootstrap: int = 100,
    step_size: int = 30,
    threshold: float = 0.05
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute rolling cross-quantilograms and generate:
      - rolling_cq_matrices: unweighted adjacency
      - rolling_weighted_matrices: adjacency * marketCap weighting
    """
    n_tokens = len(returns_data.columns)
    total_windows = len(returns_data) - window_size + 1
    if total_windows <= 0:
        raise ValueError("Not enough data for window size.")

    n_windows = (total_windows + step_size - 1) // step_size
    logging.info(f"Total windows: {n_windows}")

    rolling_cq_matrices = []
    rolling_weighted_matrices = []

    def process_window(w_idx):
        start_idx = w_idx * step_size
        end_idx = start_idx + window_size
        if end_idx > len(returns_data):
            return None, None

        window_data = returns_data.iloc[start_idx:end_idx].values
        window_caps = market_caps.iloc[start_idx:end_idx].mean()

        cq_matrix = np.zeros((n_tokens, n_tokens))
        w_matrix = np.zeros((n_tokens, n_tokens))

        token_pairs = [(j, k) for j in range(n_tokens) for k in range(j+1, n_tokens)]

        results = Parallel(n_jobs=-1)(
            delayed(compute_pair_cq)(
                window_data[:, j],
                window_data[:, k],
                quantile,
                max_lag,
                n_bootstrap
            )
            for (j, k) in token_pairs
        )

        total_cap = window_caps.sum()
        cap_weights = np.ones_like(window_caps) if total_cap == 0 else window_caps / total_cap

        for (j, k), cq_val in zip(token_pairs, results):
            if cq_val is not None and cq_val > threshold:
                cq_matrix[j, k] = cq_val
                cq_matrix[k, j] = cq_val
                w_matrix[j, k] = cq_val * cap_weights.iloc[k]
                w_matrix[k, j] = cq_val * cap_weights.iloc[j]

        return cq_matrix, w_matrix

    for window_idx in range(n_windows):
        logging.info(f"Processing window {window_idx+1} of {n_windows}")
        cqm, wm = process_window(window_idx)
        if cqm is not None and wm is not None:
            rolling_cq_matrices.append(cqm)
            rolling_weighted_matrices.append(wm)

    if not rolling_cq_matrices:
        raise ValueError("No valid matrices generated.")

    logging.info(f"Generated {len(rolling_cq_matrices)} adjacency matrices.")
    return rolling_cq_matrices, rolling_weighted_matrices

# ------------------------------------------------------------------------
# 5) TOP-100 THRESHOLD
# ------------------------------------------------------------------------
def threshold_top100_adjacency(adjacency: np.ndarray, top_n: int = 100) -> Tuple[np.ndarray, float]:
    """Calculate threshold from top N strongest connections."""
    adj_flat = adjacency.flatten()
    adj_nonzero = adj_flat[adj_flat > 0]

    if len(adj_nonzero) == 0:
        return np.zeros_like(adjacency), 0.0

    if len(adj_nonzero) < top_n:
        logging.warning(f"Fewer than {top_n} edges found.")
        top_values = np.sort(adj_nonzero)[::-1]
        threshold_val = np.mean(top_values)
    else:
        sorted_vals = np.sort(adj_nonzero)[::-1]
        top_values = sorted_vals[:top_n]
        threshold_val = np.mean(top_values)

    new_adj = np.where(adjacency > threshold_val, adjacency, 0.0)
    return new_adj, threshold_val

# ------------------------------------------------------------------------
# 6) CREATE NETWORK
# ------------------------------------------------------------------------
def create_thresholded_spillover_network(
    thresholded_adj: np.ndarray,
    weighted_matrix: np.ndarray,
    returns_data: pd.DataFrame,
    market_caps: pd.DataFrame,
    window_idx: int = -1
) -> nx.DiGraph:
    """Build directed graph with thresholded adjacency."""
    G = nx.DiGraph()
    tokens = returns_data.columns.tolist()
    n_tokens = len(tokens)

    day_idx = (len(market_caps) + window_idx) if (window_idx < 0) else (window_idx)
    day_idx = max(0, min(day_idx, len(market_caps)-1))

    day_caps = market_caps.iloc[day_idx]
    total_cap = day_caps.sum()
    rel_caps = np.ones_like(day_caps) if total_cap == 0 else day_caps / total_cap

    for i, token in enumerate(tokens):
        category = assign_token_category(token)
        G.add_node(token,
                   category=category,
                   market_cap=day_caps.iloc[i],
                   relative_size=rel_caps.iloc[i])

    for i in range(n_tokens):
        for j in range(n_tokens):
            if thresholded_adj[i, j] > 0:
                w_spill = abs(weighted_matrix[i, j])
                raw_val = thresholded_adj[i, j]
                G.add_edge(tokens[i],
                           tokens[j],
                           raw_weight=raw_val,
                           weighted_spillover=w_spill)

    return G

# ------------------------------------------------------------------------
# 7) PLOT NETWORK
# ------------------------------------------------------------------------
def plot_clustered_network(
    G: nx.DiGraph,
    title="Crypto Network (364‑day, Weighted, Top-100 Threshold, 95th Quantile)"  # Updated title
):
    """Circular cluster layout with category-based grouping."""
    plt.figure(figsize=(20, 20))
    color_map = create_token_groups()

    categories = {}
    for node, attrs in G.nodes(data=True):
        cat = attrs['category']
        categories.setdefault(cat, []).append(node)

    pos = {}
    sorted_cats = sorted(categories.keys())
    num_cats = len(sorted_cats)
    angle_gap = 2.0 * np.pi / num_cats
    radius = 2.0

    for idx, cat in enumerate(sorted_cats):
        cat_nodes = categories[cat]
        center_angle = angle_gap * idx
        if len(cat_nodes) > 1:
            sub_angle = min(angle_gap * 0.8 / (len(cat_nodes)), angle_gap * 0.3)
            start_angle = center_angle - (sub_angle * (len(cat_nodes) - 1) / 2)
            for i, node in enumerate(cat_nodes):
                angle = start_angle + sub_angle * i
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pos[node] = (x, y)
        else:
            x = radius * np.cos(center_angle)
            y = radius * np.sin(center_angle)
            pos[cat_nodes[0]] = (x, y)

    # Draw nodes
    BASE_NODE_SIZE = 400
    for cat in sorted_cats:
        cat_nodes = categories[cat]
        color = color_map.get(cat, '#cccccc')
        node_sizes = [
            BASE_NODE_SIZE + (G.nodes[n]['relative_size'] * 2000)
            for n in cat_nodes
        ]
        nx.draw_networkx_nodes(G, pos,
                             nodelist=cat_nodes,
                             node_color=color,
                             node_size=node_sizes,
                             alpha=0.9)

    # Draw edges
    MIN_EDGE_WIDTH = 1.0
    MAX_EDGE_WIDTH = 2.5
    edges_data = G.edges(data=True)
    weights = [d.get('weighted_spillover', 0) for (_, _, d) in edges_data]
    max_w = max(weights) if weights else 1.0

    for (u, v, d) in edges_data:
        w_norm = d.get('weighted_spillover', 0) / max_w
        edge_width = MIN_EDGE_WIDTH + (MAX_EDGE_WIDTH * w_norm)

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=edge_width,
            edge_color='red',  # Changed from blue to red for 95th quantile (upside risk)
            alpha=0.5,
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.2'
        )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Legend
    legend_elems = []
    for cat in sorted_cats:
        c = color_map.get(cat, '#cccccc')
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

    plt.legend(handles=legend_elems,
              loc='upper right',
              bbox_to_anchor=(1.15, 1),
              title='Categories')
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    return plt

# ------------------------------------------------------------------------
# 8) MAIN FUNCTION
# ------------------------------------------------------------------------
def main():
    """Main analysis pipeline."""
    try:
        # 1) Load data
        close_prices = pd.read_csv('/content/data_wide_close.csv',
                                   parse_dates=['Date'], index_col='Date')
        market_caps  = pd.read_csv('/content/data_wide_marketcap.csv',
                                   parse_dates=['Date'], index_col='Date')

        # 2) Compute returns
        log_returns = np.log(close_prices).diff().dropna()

        # 3) Clean & align
        log_returns = log_returns.replace('NA', np.nan).fillna(method='ffill').dropna()
        market_caps = market_caps.replace('NA', np.nan).fillna(method='ffill')
        common_dates = log_returns.index.intersection(market_caps.index)
        log_returns  = log_returns.loc[common_dates]
        market_caps  = market_caps.loc[common_dates]

        # 4) Rolling cross-quantilogram
        rolling_cq, rolling_weighted = analyze_network(
            returns_data=log_returns,
            market_caps=market_caps,
            window_size=364,
            quantile=0.95,  # Updated to 95th quantile
            max_lag=5,
            n_bootstrap=100,
            step_size=30,
            threshold=0.05
        )

        # 5) Get latest adjacency
        last_adj = rolling_cq[-1]
        last_weighted = rolling_weighted[-1]

        # 6) Apply TOP-100 threshold
        thres_adj, top100_val = threshold_top100_adjacency(last_adj, top_n=100)
        print(f"Top-100 threshold = {top100_val:.4f}")

        # 7) Build network
        G = create_thresholded_spillover_network(
            thresholded_adj=thres_adj,
            weighted_matrix=last_weighted,
            returns_data=log_returns,
            market_caps=market_caps,
            window_idx=-1
        )

        # 8) Plot network
        os.makedirs('crypto_systemic_risk_results', exist_ok=True)
        fig = plot_clustered_network(G)
        fig.savefig('crypto_systemic_risk_results/network_visualization_95quantile.png',  # Updated filename
                   dpi=300,
                   bbox_inches='tight')
        fig.show()

        logging.info("Analysis completed successfully.")
        return G, rolling_cq, rolling_weighted

    except Exception as e:
        logging.error(f"Main function error: {str(e)}", exc_info=True)
        raise

# ------------------------------------------------------------------------
# 9) RUN
# ------------------------------------------------------------------------
if __name__ == "__main__":
    G, cq_matrices, w_matrices = main()
    print("Analysis completed successfully with 95th quantile!")