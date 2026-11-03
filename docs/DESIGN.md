# Design notes

Why the project is built this way. This is the file to read before a viva or code review.

## Choice of dataset

IEEE-CIS is the primary dataset because it is the only option with the columns the brief asks for:

| Brief needs | IEEE-CIS | PaySim | ULB credit card | BAF | Elliptic |
|---|---|---|---|---|---|
| Account history | Rebuilt from card and D1 | Yes | No | No | Not applicable |
| Device | Yes, about 25% of rows | No | No | Partly | No |
| Location | addr1, addr2 | No | No | No | No |
| Card nodes for the graph | Yes | No | No | No | No |

**The brief says fraud is under 1%. On IEEE-CIS it is about 3.5%.** PaySim (about 0.13%) and ULB (about 0.17%)
are the under-1% datasets. The imbalance is still heavy and the method is unchanged, but report the
number from `reports/eda.md`, not the one in the brief.

## Rebuilding accounts

IEEE-CIS has no account column. `D1` means days since the card was first seen, so `transaction day - D1`
is constant for one customer. `user_id` is the card fields plus that value.

Competition solutions also added `addr1`. We leave it out, because `addr1` is the location signal and
including it would make a location change impossible by construction.

## Training and serving use the same features

The usual failure: features are written once in pandas for training and again for production, and
the two quietly disagree. Here there is one implementation, `FeatureEngine.process()`.

- Batch: every transaction goes through the engine in time order; rows go to Feast's offline Parquet tables.
- Live: the same method runs per event; rows go to Feast's online store and are read straight back.

`scripts/verify_parity.py` replays real events through the live path and compares every feature with
the training table. Keep that output: it is the proof.

Each row includes the current transaction, because it is known at decision time. No feature reads a
label, so this is not leakage. `user_amount_vs_avg` compares against history before the current
transaction, otherwise a large payment would raise its own baseline.

## Time is the main leakage risk

| Risk | What this project does |
|---|---|
| Random split mixes future into past | Split strictly by time: 60% train, 20% validation, 20% test |
| SMOTE copies leak across the split | SMOTE runs on training rows only, after the split |
| Early stopping tunes on evaluation data | Uses the last 15% of train, not validation |
| Blend and thresholds tuned on test | Fitted on validation; test is only reported |
| Graph contains future edges | `build_snapshot` refuses rows after its cutoff (there is a test) |
| GNN sees labels that had not arrived | Node features only use labels older than `label_delay_days` |
| Retraining reads rows not yet replayed | `roll_forward` keeps only pre-replay history and confirmed live labels (there is a test) |

## Evaluation

Accuracy is excluded. Always answering "legitimate" scores about 96.5% on IEEE-CIS and catches nothing.

- **PR-AUC** is the headline, always shown next to its baseline (the fraud rate, which is what a random
  model scores).
- **Recall and precision at a 1% review budget** answer the operational question: if analysts can
  look at 1% of traffic, how much fraud do they see?
- **Amount recall** weights by money, since missing a large fraud costs more.

## Handling imbalance

Three MLflow runs compare no reweighting, class weights and SMOTE. The winner is picked on validation
PR-AUC. Note that `scale_pos_weight` and SMOTE distort probabilities, so ranking may improve while
raw scores are uncalibrated. The blend's logistic regression recalibrates them, so the final score
can be read as a probability and thresholds mean something.

SMOTE interpolates between neighbours, which treats category codes as numbers and needs missing values
filled. On tabular fraud data it often loses to class weights. The experiment shows which it is here
rather than assuming.

## Graph model

Nodes: account, card, device. Edges: account uses card, account on device, device sees card.
Model: two-layer heterogeneous GraphSAGE with skip connections.

Each node is scored for a simple question: **will this node show fraud in the next 14 days?**
- `graph(T)` uses transactions before T only, and fraud labels at least 7 days old at T.
- The model trains on four past dates, so it sees rings at different ages, and validates on a later date.
  This works because GraphSAGE is inductive: the same weights run on any graph.
- The value of a graph over a tabular model is **label propagation**. A tabular feature cannot say "a
  neighbour of this account was confirmed as fraud"; no feature reads a label. The graph can.

A transaction's graph score is the highest risk among its account, card and device. Scores are computed
in batch and stored in Feast, so serving needs no PyTorch and the lookup costs a few milliseconds.

**Refresh.** A scheduled job rescoring the graph is simulated in evaluation: `train_gnn` writes one row per
node per weekly date, and Feast's point-in-time join gives each transaction the newest graph built before
it. In production the `graph-refresh` CronJob runs `python -m fraudlib.pipeline refresh-graph`.

**Known limit:** scores are as fresh as the last refresh. A brand-new account has no score until then. The
behavioural features still cover it, including `device_distinct_users_7d`, which reacts immediately to a
device collecting accounts. A streaming GNN would close this gap but is a project of its own.

## Blending

A logistic regression over three inputs, fitted on validation: the XGBoost logit, the graph logit, and an
"unseen by the graph" flag. The flag matters: a graph score of 0 means "no opinion", and feeding it through
a logit as if it meant "very safe" would swing every new account's score. Four parameters cannot overfit
much, are easy to explain, and the weights show how much the graph contributes.

## Decision thresholds

Both are set on validation data:
- **Block:** from this score up, precision stays at or above 95%. If fewer than 30 validation rows qualify,
  nothing is auto-blocked.
- **Review:** the 2% of traffic just below block. Blocked transactions do not use analyst capacity, so
  capacity goes where a human adds something.

## Streaming design

```
replay ──► fraud.transactions ──► scoring service ──► fraud.scores
                                      │    │
                                 Feast │    └──► fraud.alerts ──► analyst API ──► dashboard
                                      │
                                 Parquet live log ──► Evidently ──► retrain ──► MLflow champion ──► hot reload
```

- **At-least-once delivery.** Offsets commit after results are published. A crash redelivers events, so
  the scorer remembers recent transaction IDs (feature state is not counted twice), and the case table
  has a unique key (no duplicate cases).
- **Bad messages** go to `fraud.dead-letter` rather than stopping the consumer.
- **Micro-batching.** Feast keeps only the latest row per entity, so two events for one account cannot
  share a read. `split_without_repeats` starts a new batch when an entity repeats.
- **One replica.** Sliding-window state lives in the scoring process and is snapshotted every 30 seconds.
  One partition keeps global order. Scaling out means moving that state to Redis or a stream processor
  such as Flink, and partitioning by account. Card and device state would then need their own keyed
  stages. This is the first thing to change for real production use.

## Drift and retraining

`fraudlib.monitor.drift` compares the newest 5,000 live rows with a reference sample stored inside the
champion bundle, so drift is measured against what that model saw. Evidently chooses a test per column: on
samples over 1,000 rows that is a distance, where high means drift, and on small samples a p-value, where
low means drift. `has_drifted` reads which one was used. If the share of drifted monitored columns passes
`drift_share_threshold`, and the cooldown has passed, it retrains.

**What retraining learns from.** `roll_forward` joins pre-replay history with live rows that have a confirmed
label. Unreviewed rows are left out, because assuming "unreviewed means legitimate" would teach the model that
the fraud it missed was fine. Retraining waits for at least 3,000 confirmed rows and 30 confirmed frauds.

The live window is cut in three by time, because a model only improves if it trains on the new traffic:

| Third of live window | Used for |
|---|---|
| Oldest | Joins training |
| Middle | Validation: blend weights and thresholds |
| Newest | Promotion gate, unseen by both models |

**Drift does not promote a model.** Challenger and champion are scored on the gate window, each with its own
category vocabulary, and the alias only moves if the challenger wins. Both use the same new graph scores, so
the gate compares the tabular model and blend weights, not two graphs. The scoring service checks the alias
every 30 seconds and swaps the model without restarting.

After a promotion the reference sample comes from the validation third. If drift began partway through that
third, the reference is a mix of old and new traffic, so the monitor can keep reporting drift until the next
retrain. The cooldown stops this from causing a retraining loop.

## Defects found only by running the whole system

Worth raising in a viva: 26 unit tests passed while all of these were present.

| Defect | Effect | Found by |
|---|---|---|
| Serving used every tree, training reported early-stopped trees | Live scores came from a different model than the one evaluated | Comparing the two predictors in `verify_parity.py` |
| `event_ts` with and without milliseconds in one batch | Scoring crashed | Parity run on real replay rows |
| Raw and `feat_` columns collapsed to the same name | Drift check crashed on its first run | First drift check |
| Wasserstein distance read as a p-value | Dashboard listed stable columns as drifted | Reading Evidently's output |
| Retraining kept the old time split | New traffic only ever reached the test set, so no model could improve | Reading `retrain` against the split |
| Retraining read the raw file | It could use labels for rows not yet replayed | Same review |
| Champion scored with the challenger's category codes | Promotion gate was unfair | Same review |
| Review and block thresholds collapsed into one | Analysts received no review cases | First evaluation |
| `unknown` treated as one shared device | Micro-batches fell to one or two events | Code review before load testing |
| MLflow's default job workers | About 1.5 GB of extra memory, past its Kubernetes limit | Investigating an out-of-memory kill |

## What is intentionally simple

These are fine for a capstone and would change in production:

- Passwords in environment variables; a bank would use SSO and role-based access.
- Parquet on a shared volume for logs; production would use object storage.
- Retraining sees only labelled live rows, so behavioural features on those rows are slightly undercounted.
  Rebuilding features from all live rows and masking unlabelled ones would fix it.
- A single Kafka broker with no replication or TLS.
- Automatic blocking is a recommendation in this system. Nothing here stops a real payment.
