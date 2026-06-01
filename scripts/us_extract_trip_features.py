import ollama
import json

def extract_trip_features(comment_text):
    """
    Uses a local LLM to quantify subjective race notes.
    """
    prompt = f"""
    Analyze this horse racing comment: "{comment_text}"
    Return ONLY a JSON object with these keys:
    - trouble_at_start (0-1)
    - wide_trip (0-1)
    - strong_finish (0-1)
    """
    
    response = ollama.generate(model='qwen2.5-coder', prompt=prompt)
    try:
        # Simple cleanup to ensure valid JSON
        data = json.loads(response['response'].strip())
        return data
    except Exception:
        return {"trouble_at_start": 0, "wide_trip": 0, "strong_finish": 0}