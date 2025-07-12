import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# === ENV VAR SETUP ===
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_name = "llama3-8b-8192"

# === HARDCODED FITNESS KNOWLEDGE BASE ===
kb_chunks = [
    "Strength training builds muscle and includes squats, deadlifts, and presses.",
    "Protein is essential for muscle recovery and is found in eggs, fish, and tofu.",
    "Cardio improves heart health and burns fat. Examples: running, cycling, HIIT.",
    "Sleep (7–9 hrs) helps recovery and hormone regulation, critical for fitness progress.",
    "Healthy fats (avocados, olive oil, nuts) are essential for brain and hormone health.",
    "Complex carbs (like oats, rice, quinoa) provide long-lasting energy for workouts.",
    "Drinking water helps metabolism, digestion, and physical performance.",
    "Weight loss = calorie deficit; weight gain = calorie surplus. Both require consistency.",
    "Micronutrients like magnesium, iron, and vitamin D support muscle and energy systems.",
    "Supplements like creatine, whey protein, and multivitamins can support fitness goals."
]

# === EMBEDDINGS + VECTOR INDEX (cached) ===
@st.cache_resource
def setup_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index

embedding_model, faiss_index = setup_index(kb_chunks)

# === ASK FUNCTION ===
def ask(query, k=4):
    query_vec = embedding_model.encode([query])
    _, I = faiss_index.search(np.array(query_vec), k)
    top_context = "\n".join([kb_chunks[i] for i in I[0]])

    prompt = f"""Answer the fitness-related question below using the context:

Context:
{top_context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

# === STREAMLIT UI ===
st.title("🏋️ Fitness RAG Chatbot")
st.markdown("Ask about training, food, sleep, recovery, etc.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "")

if user_input:
    with st.spinner("Thinking..."):
        reply = ask(user_input)
        st.session_state.history.append((user_input, reply))

# Display history 
#manav mangukiya
for user, bot in reversed(st.session_state.history):
    st.markdown(f"**You:** {user}")
    st.markdown(f"**Bot:** {bot}")
    st.markdown("---")