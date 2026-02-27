# CDL Match Prediction

**Data science project: predictive modeling and statistical inference for Call of Duty League (CDL) esports.**

An end-to-end machine learning pipeline that uses **team statistics** and **historical match data** to predict match outcomes. This project prioritizes mathematical interpretability over "black box" algorithms, proving exactly which in-game metrics dictate series outcomes.

---

## Highlights

- **Predictive Modeling**: Utilizes an L2-Regularized Logistic Regression (Ridge) model. Evaluated using a rigorous **Walk-Forward Backtest** to simulate real-world betting conditions without look-ahead bias, achieving an optimized ~64% accuracy against the spread.
- **Statistical Inference**: Integrates `statsmodels` to generate formal statistical summaries (p-values, z-scores, coefficients). Mathematically proves that **Hardpoint superiority** is the single greatest driver of series wins, while Search and Destroy (SND) is highly volatile and statistically unreliable for predictions.
- **Dynamic Feature Engineering**: Dynamically calculates rolling win rates, head-to-head records, and game-mode differentials (HP, SND, OVL) on a match-by-match basis to prevent data leakage.
- **Advanced Metrics**: Tracks deep game-mode analytics, including raw Win Percentages, Team K/D Ratios by mode, and Overall Round Plus/Minus.
- **Multicollinearity Handling**: Utilizes L2 Regularization to smoothly handle the heavy mathematical correlation between a team's Win Percentage and their K/D ratio without aggressively deleting features.
- **Automated Visual Diagnostics**: Generates Seaborn/Matplotlib charts natively in the modeling block to instantly visualize the "Raw Win Rate Illusion" versus the "Mathematical Reality" of the model's feature weights.

---

## Project Structure

```text
cod-analysis/
├── README.md
├── cod-analysis.ipynb    # Main pipeline: data prep, rolling features, ML modeling, statsmodels inference, and viz
├── team_stats.csv        # Team-level stats (Win%, K/D by mode, Plus/Minus)
├── match_results.csv     # Historical match outcomes (Date, Team A, Team B, Winner)
