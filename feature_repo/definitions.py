"""Feast definitions. Feature names come from the engine so the two cannot diverge."""
import os
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32
from feast.value_type import ValueType

from fraudlib.features.engine import CARD_FEATURES, DEVICE_FEATURES, USER_FEATURES

DATA = Path(os.environ.get("FEATURE_DATA_DIR", Path(__file__).resolve().parents[1] / "data/features"))
# Every transaction writes its own row, so a training join only needs to look back a moment. Without this
# bound Feast pairs each transaction with all earlier rows of that entity and runs out of memory.
BEHAVIOUR_TTL = timedelta(days=1)
NO_EXPIRY = timedelta(0)            # graph scores stay valid until the next batch replaces them

user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.STRING)
card = Entity(name="card", join_keys=["card_id"], value_type=ValueType.STRING)
device = Entity(name="device", join_keys=["device_id"], value_type=ValueType.STRING)


def view(name: str, entity: Entity, features: list[str], ttl: timedelta) -> FeatureView:
    return FeatureView(
        name=name, entities=[entity], ttl=ttl, online=True,
        schema=[Field(name=f, dtype=Float32) for f in features],
        source=FileSource(name=f"{name}_source", path=str(DATA / f"{name}.parquet"),
                          timestamp_field="event_timestamp"),
    )


user_features = view("user_features", user, USER_FEATURES, BEHAVIOUR_TTL)
card_features = view("card_features", card, CARD_FEATURES, BEHAVIOUR_TTL)
device_features = view("device_features", device, DEVICE_FEATURES, BEHAVIOUR_TTL)
# Graph scores are computed in batch by the GNN and looked up at serving time.
user_graph = view("user_graph", user, ["user_graph_risk"], NO_EXPIRY)
card_graph = view("card_graph", card, ["card_graph_risk"], NO_EXPIRY)
device_graph = view("device_graph", device, ["device_graph_risk"], NO_EXPIRY)

fraud_scoring = FeatureService(
    name="fraud_scoring",
    features=[user_features, card_features, device_features, user_graph, card_graph, device_graph],
)
