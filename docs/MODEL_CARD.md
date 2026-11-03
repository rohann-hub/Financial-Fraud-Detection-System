# Model card: fraud-blend

**Purpose.** Rank payment transactions by fraud risk so analysts review the riskiest first. Educational capstone; not
approved for real payment decisions.

**Components.** XGBoost on transaction and behavioural features, a heterogeneous GraphSAGE on the
account, card and device graph, and a logistic blend of the two with an "unseen by the graph" term.

**Training data.** Designed for IEEE-CIS Fraud Detection (Vesta / IEEE Computational Intelligence Society), split by
time. **The results shipped in `reports/results/` come from the synthetic generator, not from IEEE-CIS.** Replace them
with your own run before quoting any performance figure. Your numbers land in `data/artifacts/evaluation.json`.

**Metrics.** PR-AUC, ROC-AUC, recall and precision at a 1% review budget, recall at 90% precision, amount recall.

**Known limitations.**
- Accounts are reconstructed, not real identifiers, so some are merged or split.
- Device data exists for about a quarter of transactions, so device features and graph edges are missing for the rest.
- The graph score is batch-refreshed, and accounts and devices it has not seen carry a separate "unseen" term.
- IEEE-CIS covers one payment provider over six months in anonymised form. Results do not transfer to other populations.
- The data's labels partly reflect how the provider marked accounts after a first confirmed fraud, which can
  inflate account-history signals. Time-based evaluation reduces this but does not remove it.

**Fairness.** The data has no protected attributes, so group fairness cannot be measured directly. Device type
and email domain can act as proxies. Compare alert rates across them before any real use.

**Human oversight.** Analysts make the final decision. Their labels feed retraining, which is gated on
held-out performance.
