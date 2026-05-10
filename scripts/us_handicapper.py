import streamlit as st

def run_dashboard(race_data):
    st.title("🏇 AI Handicapper: Value Finder")
    
    for race_id, entries in race_data.groupby('race_id'):
        st.subheader(f"Race {race_id}")
        
        # Calculate Model Probabilities
        entries['model_prob'] = get_predictions(my_model, entries)
        
        # Calculate Implied Probability from Tote Odds (e.g., 2/1 -> 1/(2+1) = 33%)
        entries['tote_prob'] = 1 / (entries['tote_odds'] + 1)
        
        # Calculate Edge
        entries['edge'] = entries['model_prob'] - entries['tote_prob']
        
        # Highlight Overlays
        st.table(entries[['horse_name', 'tote_odds', 'model_prob', 'edge']].style.highlight_max(subset=['edge']))

        # Kelly Criterion Recommendation
        # f* = (bp - q) / b
        for _, horse in entries.iterrows():
            if horse['edge'] > 0:
                b = horse['tote_odds']
                p = horse['model_prob']
                q = 1 - p
                kelly = (b * p - q) / b
                if kelly > 0:
                    st.success(f"Suggest wagering {kelly:.1%} on {horse['horse_name']}")