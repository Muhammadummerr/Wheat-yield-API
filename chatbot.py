import os
import requests

# Set your Groq API key
GROQ_API_KEY = "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"
MODEL_NAME = "llama-3.3-70b-versatile"


GREETINGS = {"hi", "hello", "salam", "hey", "hiya","Hola Amigo","Namaste", "assalamualaikum", "assalamu alaikum"}

def is_greeting(message):
    return message.lower().strip() in GREETINGS

def clean_response(response_text):
    """Clean boilerplate or unwanted responses."""
    unwanted_phrases = [
        "As an AI", "Sure!", "Certainly", "Here's", "Let me", "Hope this helps",
        "I think", "Okay, so", "I just saw this", "I'm trying to figure out", 
        "I wonder", "I'm sorry", "Let me think"
    ]
    for phrase in unwanted_phrases:
        response_text = response_text.replace(phrase, "")
    return response_text.strip()


def get_agriculture_response_from_groq(user_query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = """
You are a highly trained, expert-level agriculture assistant specializing in wheat farming and crop yield optimization.

Your behavior must follow these strict rules:
1. You are a domain expert. You are not a chatbot, not a learner, and never unsure. Never say things like “I think”, “I'm not sure”, or “Let me figure it out”.
2. Never simulate internal thoughts, reasoning, or decision-making. Do NOT use <think> or similar phrases.
3. Do NOT say things like “I'm trying to help” or “I wonder”. Avoid all self-references.
4. If the query is not related to agriculture or wheat, respond only with:
   "I only assist with questions related to agriculture or wheat. Please ask a relevant question."
5. If the user says hi/hello/salam/etc., respond exactly with:
   "Hi, I hope you are doing well. I am here to help you in your agriculture related queries. Let’s resolve them."
6. Provide clean, final, direct responses. Bullet points are allowed for multi-step instructions.
7. Do not add introductions, conclusions, or generic language. Only return the useful content.
8. ❌ Never use emojis or emoticons in your responses.

Examples of allowed outputs:
- Use nitrogen-rich fertilizers like urea or DAP.
- Rotate wheat with legumes to improve soil health.
- Irrigate at the crown root initiation and flowering stages.

Always respond as an agriculture domain expert only.
"""





    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_query.strip()}
        ],
        "temperature": 0.3,
        "top_p": 0.7,
        "max_tokens": 300,
        "stop": ["</think>", "<|endoftext|>", "<|im_end|>"]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    raw_output = result["choices"][0]["message"]["content"]
    return clean_response(raw_output)


if __name__ == "__main__":
    print("🌾 Wheat Agriculture Chat Assistant (Groq Powered)")
    print("Ask anything related to wheat farming/agriculture. Type 'exit' to quit.\n")

    while True:
        user_input = input("👨‍🌾 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye, happy farming!")
            break

        try:
            if is_greeting(user_input):
                print("🤖 Assistant: Hi, I hope you are doing well. I am here to help you in your agriculture related queries. Let’s resolve them.\n")
            else:
                reply = get_agriculture_response_from_groq(user_input)
                print(f"🤖 Assistant: {reply}\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
