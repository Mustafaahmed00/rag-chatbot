import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIG ===
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_name = "llama3-8b-8192"

# === KNOWLEDGE BASE ===
knowledge_base = [
   "Strength training is essential for muscle building. It includes exercises like squats, deadlifts, bench press, and rows.",
    "Protein is important for muscle repair and growth. Good sources include chicken, eggs, tofu, fish, and legumes.",
    "Cardio exercises like running, cycling, and swimming improve heart health and aid in fat loss.",
    "A balanced diet includes carbohydrates, proteins, fats, vitamins, and minerals. Each plays a unique role in health.",
    "Drinking enough water is vital for metabolism, muscle function, and recovery after exercise.",
    "Consuming fewer calories than you burn leads to weight loss, while a calorie surplus promotes weight gain.",
    "Sleep and recovery are just as important as exercise. Aim for 7-9 hours of sleep for optimal performance.",
    "Healthy fats such as those from nuts, seeds, olive oil, and avocado support brain function and hormone health.",
    "Carbs are the body's preferred energy source. Complex carbs like oats, quinoa, and brown rice are healthier than refined carbs.",
    "Micronutrients like iron, calcium, magnesium, and vitamin D are essential for muscle function and overall health.",
]

# === EMBEDDING MODEL ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(knowledge_base)

# === VECTOR INDEX ===
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# === CHATBOT FUNCTION ===
def ask(query, k=2):
    query_vec = embedding_model.encode([query])
    _, I = index.search(np.array(query_vec), k)
    top_chunks = [knowledge_base[i] for i in I[0]]
    context = "\n".join(top_chunks)

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# === RUN CHAT ===
if __name__ == "__main__":
    print("🧠 Fitbot. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = ask(user_input)
        print("FitBot:", answer, "\n")