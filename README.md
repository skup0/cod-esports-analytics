# CDL Match Prediction

**Data science project: predictive modeling for Call of Duty League (CDL) esports.**

End-to-end pipeline that combines **team + player statistics** with **multiple ML classifiers** and **Monte Carlo simulation** to predict match outcomes and best-of-5 score distributions.

---

## Highlights

- **Predictive modeling**: Trained and compared 7 classifiers (Logistic Regression, Random Forest, Gradient Boosting, SVM RBF/Linear, K-NN, Naive Bayes) with **repeated stratified K-fold cross-validation**; selected best model at **~95% accuracy** on 132 matchup samples, 55 engineered features.
- **Feature engineering**: Built matchup-level features from **team stats** (HP/SND/OVL win%, KD, plus/minus) and **aggregated player stats** per team; standardized and validated on all 12 CDL teams.
- **Probabilistic simulation**: **Monte Carlo Bo5 simulator** (10k runs) with mode-specific win probabilities (HP → SND → OVL map order); combined with ML classifier in a 50/50 ensemble for final win probability and most-likely score (e.g. 3–0, 3–1, 3–2).
- **Reproducible workflow**: Jupyter notebooks, pandas/numpy for EDA and feature construction, scikit-learn for modeling and evaluation, matplotlib for visualizations (e.g. predicted wins per team).

---

## Project structure

```
cod-analysis/
├── README.md
├── cod-analysis.ipynb    # Match prediction: models, simulation, predict(), week predictions, viz
├── team_stats.csv       # Team-level stats (HP/SND/OVL win%, KD, etc.)
├── player_stats.csv     # Player-level stats (K/D, HP K/10m, SND KpR, OVL, etc.)
└── team_roster.csv      # Player → team mapping
```

---

## Data

- **team_stats.csv**: One row per team; columns include `HP_Win_Percent`, `HP_KD`, `SND_Win_Percent`, `SND_KD`, `OVL_Win_Percent`, `OVL_KD`, plus/minus and round wins by mode.
- **player_stats.csv**: Per-player stats (K/D, HP K/10m, SND KpR, plants/defuses, OVL K/10m, overloads, etc.).
- **team_roster.csv**: `Player` → `team_code` for joining player stats to teams.

Teams covered: OpTic Texas, FaZe Vegas, G2 Minnesota, Paris Gentle Mates, Los Angeles Thieves, Carolina Royal Ravens, Miami Heretics, Boston Breach, Riyadh Falcons, Cloud9 New York, Toronto KOI, Vancouver Surge.

---

## Match prediction (cod-analysis.ipynb)

1. **Load & merge** `team_stats`, `player_stats`, `team_roster`; coerce numerics, aggregate player stats by team.
2. **Build matchup matrix**: For every ordered pair (A, B), compute feature vector = (team A − team B) stats and (player-agg A − player-agg B); label = 1 if A’s combined mode win% > B’s.
3. **Train & evaluate**: `StandardScaler` + 7 models; `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)`; report mean accuracy ± std; retain best model (e.g. SVM Linear).
4. **Simulation**: For a matchup, compute per-map win probability for HP, SND, OVL using win%, KD, plus/minus, and player skill; run 10k Bo5 trials; get distribution over 3–0, 3–1, 3–2.
5. **Predict**: `predict(team_a, team_b)` returns winner, predicted score, win probability, and per-mode (HP/SND/OVL) win probs; blend 50% ML probability and 50% simulation series win rate.
6. **Outputs**: Table of week matchups with predicted winner, score, win %, and mode probs; bar chart of predicted match wins per team.

---

## Requirements

- Python 3.x  
- pandas, numpy  
- scikit-learn  
- matplotlib  

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## How to run

1. Clone or download the repo; ensure the three CSVs are in the same directory as the notebook.
2. Open **cod-analysis.ipynb**: run cells top to bottom to train models, then use `predict("OpTic Texas", "FaZe Vegas")` or `predict_str(...)` for a one-line summary.

---

## Possible extensions

- Backtest predictions against actual CDL results when available.
- Add more features (recent form, head-to-head, map veto assumptions).
- Tune hyperparameters (e.g. grid search for SVM C, RF depth).
- Deploy as a small API or Streamlit app for interactive matchup queries.

---

*Data reflects CDL structure and typical stat sources.*
