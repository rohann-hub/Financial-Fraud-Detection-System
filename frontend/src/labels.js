// Plain-language names for model features, shown to analysts as reasons.
export const FEATURE_NAMES = {
  amount: "Transaction amount",
  log_amount: "Transaction amount",
  hour: "Time of day",
  user_amount_vs_avg: "Amount compared with this account's average",
  user_amount_zscore: "Amount is unusual for this account",
  user_amount_mean: "This account's average amount",
  user_amount_std: "How much this account's amounts vary",
  user_txn_count_1h: "Account's payments in the last hour",
  user_txn_count_24h: "Account's payments in the last 24 hours",
  user_txn_count_7d: "Account's payments in the last 7 days",
  user_amount_sum_1h: "Account's spend in the last hour",
  user_amount_sum_24h: "Account's spend in the last 24 hours",
  user_txn_count_total: "Account's total payments so far",
  user_secs_since_last: "Seconds since the account's last payment",
  user_device_changed: "Device differs from the last one used",
  user_new_device: "First time this account used this device",
  user_location_changed: "Location differs from the last one used",
  user_new_location: "First time this account used this location",
  user_distinct_devices_7d: "Devices used by the account in 7 days",
  user_distinct_locations_7d: "Locations used by the account in 7 days",
  card_txn_count_1h: "Card's payments in the last hour",
  card_txn_count_24h: "Card's payments in the last 24 hours",
  card_amount_sum_24h: "Card's spend in the last 24 hours",
  card_distinct_users_7d: "Accounts using this card in 7 days",
  card_distinct_devices_7d: "Devices using this card in 7 days",
  device_txn_count_1h: "Device's payments in the last hour",
  device_txn_count_24h: "Device's payments in the last 24 hours",
  device_distinct_users_7d: "Accounts on this device in 7 days",
  device_distinct_cards_7d: "Cards on this device in 7 days",
  device_user_count_total: "Accounts ever seen on this device",
};

export const featureName = (f) => FEATURE_NAMES[f] || `Bank signal ${f.replace("_code", "")}`;

export const STATUS_NAMES = {
  open: "Open",
  investigating: "Investigating",
  escalated: "Escalated",
  closed: "Closed",
};

export const money = (n) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(n);

export const shortId = (id) => (id.length > 26 ? `${id.slice(0, 25)}…` : id);

export const when = (iso) =>
  new Date(iso).toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  });
