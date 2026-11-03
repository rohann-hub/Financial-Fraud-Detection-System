# Exploratory data analysis

Data source: SYNTHETIC test data, not a real dataset

- 199,947 transactions over 180 days; 6,947 are fraud.
- **Fraud rate 3.47%, or 1 fraud per 28 legitimate transactions.**
- A model that always answers "legitimate" scores 96.53% accuracy and catches nothing.
  That is why this project evaluates with PR-AUC.
- 26,724 rebuilt accounts, 5,779 cards, 12,002 devices.
- A device is known for only 41.9% of transactions. Device features and graph edges cover that share.
- Median amount: fraud 63.73, legitimate 47.98.

![Fraud rate by week](eda_fraud_rate_by_week.png)

![Amount distribution](eda_amount.png)

![Fraud by device sharing](eda_shared_devices.png)
