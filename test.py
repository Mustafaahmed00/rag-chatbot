import os
from groq import Groq

# Use env var
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set")

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "user", "content": "What is Groq?"},
    ],
)

print("✅ Groq works!")
print(response.choices[0].message.content.strip())