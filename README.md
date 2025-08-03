Cross‑Quantilogram Crypto Spillover Analysis
This repository contains Python scripts for analyzing crypto market spillovers, connectedness, and systemic risk using the Cross‑Quantilogram (CQ) model.
It focuses on tail dependence at both the 5th quantile (downside risk) and 95th quantile (upside risk).

Core Idea
The Cross‑Quantilogram (CQ) model measures how extreme movements in one crypto asset affect extreme movements in another, focusing on specific quantiles rather than average correlations.

5th quantile models capture negative tail spillovers (market crashes).

95th quantile models capture positive tail spillovers (market rallies).

Outputs include spillover networks, connectedness measures, and systemic risk scores.

File Overview
Script	Purpose
Multieventspillovermodel5thquantile.py	Multi‑event spillover network for 5th quantile (downside).
Multieventspillovermodel95thquantile.py	Multi‑event spillover network for 95th quantile (upside).
SpilloverModel5thquantile.py	Single‑event spillover network (5th quantile).
SpilloverModel95thquantile.py	Single‑event spillover network (95th quantile).
NetConnectedness5thquantile.py	Computes network connectedness for downside tail risk.
NetConnectedness95thquantile.py	Computes network connectedness for upside tail risk.
systematicriskscore5thquantile.py	Calculates systemic risk scores for 5th quantile events.
systematicriskscore95thquantile.py	Calculates systemic risk scores for 95th quantile events.
Robustness5thquantile.py	Robustness checks for downside spillover networks.
Robustness95thquantile.py	Robustness checks for upside spillover networks.
delta.py	Utility functions for incremental CQ calculations.

Workflow
Data Preparation

CSV files containing historical prices and market capitalizations of crypto tokens.

Model Execution

Choose 5th or 95th quantile scripts depending on tail risk focus.

Run multi‑event or single‑event spillover scripts first.

Compute connectedness and systemic risk scores.

Perform robustness checks if needed.

Outputs

Network plots of spillovers.

Connectedness measures for each event.

Systemic risk score summaries.

Requirements
Install dependencies:

bash
Copy
Edit
pip install pandas numpy matplotlib networkx joblib
