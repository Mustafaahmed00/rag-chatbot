import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# === ENV VAR SETUP & VALIDATION ===
@st.cache_data
def validate_api_key():
    """Validate Groq API key and return client if successful"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "❌ GROQ_API_KEY environment variable not set. Please set it and restart the app."
    
    try:
        client = Groq(api_key=api_key)
        # Test the API key with a simple call
        test_response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        return client, "✅ API key validated successfully"
    except Exception as e:
        return None, f"❌ Invalid API key or connection error: {str(e)}"

# Initialize client
client, validation_message = validate_api_key()
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
    """Setup embeddings and FAISS index with error handling"""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        return model, index, None
    except Exception as e:
        return None, None, f"❌ Error setting up embeddings: {str(e)}"

embedding_model, faiss_index, setup_error = setup_index(kb_chunks)

# === ASK FUNCTION ===
def ask(query, k=4):
    """Process query with comprehensive error handling"""
    if not client:
        return "❌ API client not available. Please check your GROQ_API_KEY."
    
    if not embedding_model or not faiss_index:
        return "❌ Embedding model not loaded. Please refresh the page."
    
    if not query.strip():
        return "❌ Please enter a valid question."
    
    try:
        # Get relevant context
        query_vec = embedding_model.encode([query])
        _, I = faiss_index.search(np.array(query_vec), k)
        top_context = "\n".join([kb_chunks[i] for i in I[0]])

        prompt = f"""Answer the fitness-related question below using the context:

Context:
{top_context}

Question: {query}
Answer:"""

        # Call Groq API with timeout handling
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"❌ Sorry, I encountered an error: {str(e)}. Please try again."

# === STREAMLIT UI ===
st.title("🏋️ Fitness RAG Chatbot")
st.markdown("Ask about training, food, sleep, recovery, etc.")

# Show API validation status
if "✅" in validation_message:
    st.success(validation_message)
else:
    st.error(validation_message)
    st.stop()  # Stop execution if API key is invalid

# Show setup errors if any
if setup_error:
    st.error(setup_error)
    st.stop()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Input section
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("You:", "", key="user_input")
with col2:
    if st.button("Clear Chat", type="secondary"):
        st.session_state.history = []
        st.rerun()

# Process input
if user_input:
    if user_input.strip():  # Only process non-empty inputs
        with st.spinner("🤔 Thinking..."):
            reply = ask(user_input)
            st.session_state.history.append((user_input, reply))
        # Clear the input box
        st.rerun()
    else:
        st.warning("Please enter a valid question.")

# Display history in correct order (oldest first)
if st.session_state.history:
    st.markdown("---")
    for user, bot in st.session_state.history:  # Removed 'reversed' for correct order
        st.markdown(f"**You:** {user}")
        st.markdown(f"**Bot:** {bot}")
        st.markdown("---")