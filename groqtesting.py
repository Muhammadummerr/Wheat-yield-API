import os
import requests

# Set your Groq API key in environment variable or paste directly
GROQ_API_KEY = "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"

# Model to use (Mistral 7B or Mixtral 8x7B are supported on Groq)
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instructleaf"

def clean_response(response_text):
    """Clean boilerplate or unnecessary text from LLM output."""
    unwanted_phrases = [
        "Here is the treatment", "Here's the treatment", "Sure", 
        "The answer is", "As an AI", "You can", "I can", 
        "Let me explain", "Certainly", "Below is"
    ]
    for phrase in unwanted_phrases:
        response_text = response_text.replace(phrase, "")
    return response_text.strip().lstrip("-").strip()


def get_treatment_from_groq(disease, severity):
    """Send prompt to Groq LLM and return cleaned treatment recommendation."""
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an expert wheat agronomist. 
Your job is to provide **only treatment suggestions** for a given wheat disease and severity.
Avoid introductions, explanations, or any unnecessary text. Only return treatment instructions in clear bullet points if multiple.

Disease: {disease}
Severity: {severity}

Treatment:
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt.strip()}
        ],
        "temperature": 0.5,
        "max_tokens": 200
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    raw_output = result["choices"][0]["message"]["content"]
    return clean_response(raw_output)


if __name__ == "__main__":
    print("🌾 Wheat Crop Disease Treatment Assistant")
    disease_name = input("Enter disease name: ")
    severity_level = input("Enter severity level (e.g. Low, Moderate, Severe): ")

    try:
        treatment = get_treatment_from_groq(disease_name, severity_level)
        print("\nSuggested Treatment:")
        print(treatment)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
