import json
import pathlib
import random
from typing import Dict, List

import streamlit as st


BASE = pathlib.Path(__file__).resolve().parents[1]


def load_sample_races() -> List[Dict]:
    """Load sample races from data/raw/ if present, else fallback to a tiny embedded sample.

    Expected JSON format: {"race_id": .., "name": .., "horses": [{"id":.., "name":..}, ...]}
    """
    raw_dir = BASE / "data" / "raw"
    if raw_dir.exists():
        # pick first json file if available
        for p in raw_dir.glob("*.json"):
            try:
                return [json.loads(p.read_text())]
            except Exception:
                continue

    # fallback sample
    return [
        {
            "race_id": 123,
            "name": "Sample Race",
            "horses": [
                {"id": 1, "name": "Horse A"},
                {"id": 2, "name": "Horse B"},
                {"id": 3, "name": "Horse C"},
            ],
        }
    ]


def fake_predict(horses: List[Dict]) -> Dict[str, float]:
    """Placeholder prediction: returns normalized random scores per horse.

    Replace with model inference call (load model from `models/` and call `predict_proba`).
    """
    scores = [random.random() for _ in horses]
    total = sum(scores) or 1.0
    return {h["name"]: round(s / total, 3) for h, s in zip(horses, scores)}


def main():
    st.set_page_config(page_title="Horse Racing Predictions", layout="wide")
    st.title("Horse Racing Predictions")

    st.sidebar.header("Data & Options")
    races = load_sample_races()

    race_names = [r.get("name", str(r.get("race_id"))) for r in races]
    selected = st.sidebar.selectbox("Select race", race_names)
    race = races[race_names.index(selected)]

    st.header(f"{race.get('name')} (ID: {race.get('race_id')})")

    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Runners")
        for h in race.get("horses", []):
            st.write(f"• {h.get('name')} (id={h.get('id')})")

    with cols[1]:
        st.subheader("Model")
        st.write("No model loaded — using placeholder predictions.")
        st.text("Model path: models/model.pkl (not found)")

    if st.button("Predict"):
        horses = race.get("horses", [])
        probs = fake_predict(horses)
        st.subheader("Predicted probabilities")
        for name, p in sorted(probs.items(), key=lambda x: -x[1]):
            st.write(f"{name}: {p:.3f}")

    st.markdown("---")
    st.caption("This is an initial UI. Replace `fake_predict` with your model inference.")


if __name__ == "__main__":
    main()
