# CDL Match Prediction

**Data science project: predictive modeling and statistical inference for Call of Duty League (CDL) esports.**

An end-to-end machine learning pipeline that uses **team statistics** and **historical match data** to predict match outcomes. This project prioritizes mathematical interpretability over "black box" algorithms, proving that specific game modes dictate series outcomes.

---

## Highlights

- **Predictive Modeling**: Utilizes an L2-Regularized Logistic Regression (Ridge) model. Evaluated using a rigorous **Walk-Forward Backtest** to simulate real-world betting conditions without look-ahead bias, achieving ~63% accuracy against the spread.
- **Statistical Inference**: Integrates `statsmodels` to generate formal statistical summaries (p-values, z-scores, coefficients). Mathematically proved that **Hardpoint superiority** is the single greatest driver of series wins (p = 0.034), while Search and Destroy (SND) is highly volatile and statistically unreliable for predictions.
- **Dynamic Feature Engineering**: Dynamically calculates rolling win rates, head-to-head records, and game-mode differentials (HP, SND, OVL) on a match-by-match basis to prevent data leakage.
- **Multicollinearity Handling**: Utilizes L2 Regularization to smoothly handle the heavy mathematical correlation between a team's Hardpoint and SND statistics without aggressively deleting features.

---

## Project Structure

```text
cod-analysis/
├── README.md
├── cod-analysis.ipynb    # Main pipeline: data prep, rolling features, ML modeling, and statsmodels inference
├── team_stats.csv        # Team-level stats (HP/SND/OVL win%)
├── match_results.csv     # Historical match outcomes (Date, Team A, Team B, Winner)
