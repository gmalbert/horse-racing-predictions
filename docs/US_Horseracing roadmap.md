# US Horse Racing Prediction Engine: Technical Findings

## 1. Data Acquisition Strategy
*   **Primary Source:** Equibase is the mandatory starting point. Their Developer Dataset provides the necessary historical depth for feature training.
*   **Scraping Targets:** ADW sites (TwinSpires/FanDuel) are best for live "Tote" pool data via XHR/JSON endpoint interception rather than DOM scraping.

## 2. Feature Engineering Insights
*   **The Track Variant:** Raw times are deceptive. Normalizing against the "Daily Variant" (the deviation of all races on a card from the track par) is the single most important adjustment for speed figures.
*   **Text Analysis:** Subjective trip notes contain "hidden" data (e.g., a horse blocked at the rail). Local LLM extraction (Ollama) can quantify these into binary features that standard models miss.

## 3. Modeling Methodology
*   **Point-Wise vs. Pair-Wise:** Traditional binary classification (Will it win?) fails because it doesn't account for the strength of the field. A **Pair-Wise Ranker** (XGBoost/LightGBM) is superior because it optimizes for the relative finishing order.
*   **Probability Normalization:** Model outputs must be normalized per race using the formula:
    $$P_{norm}(i) = \frac{P(i)}{\sum_{j=1}^{n} P(j)}$$

## 4. Betting Logic
*   **The Overlay Concept:** Profitability is not found by picking winners, but by identifying discrepancies between $P_{model}$ and $P_{tote}$.
*   **Risk Management:** Using the **Kelly Criterion** ensures bankroll longevity by scaling bets according to the size of the perceived edge:
    $$f^* = \frac{bp - q}{b}$$
    *(Where b = decimal odds, p = model probability, q = 1-p)*

## 5. Technical Stack Recommended
- **Language:** Python 3.11+
- **Hardware:** Local execution on Apple Silicon (M-series) using **MLX** for accelerated training.
- **Inference:** Ollama for local NLP feature extraction.
- **UI:** Streamlit for rapid dashboarding and real-time value visualization.