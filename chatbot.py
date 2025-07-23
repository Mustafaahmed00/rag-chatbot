import os
import streamlit as st
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import time
import json

# === ENV VAR SETUP ===
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_name = "llama3-8b-8192"

# === ENHANCED FITNESS KNOWLEDGE BASE ===
kb_chunks = [
    "Strength training builds muscle and includes squats, deadlifts, and presses. Progressive overload is key - gradually increase weight or reps.",
    "Protein is essential for muscle recovery and is found in eggs, fish, and tofu. Aim for 1.6-2.2g per kg body weight daily.",
    "Cardio improves heart health and burns fat. Examples: running, cycling, HIIT. Start with 150 minutes moderate or 75 minutes vigorous weekly.",
    "Sleep (7–9 hrs) helps recovery and hormone regulation, critical for fitness progress. Deep sleep is when muscle repair happens.",
    "Healthy fats (avocados, olive oil, nuts) are essential for brain and hormone health. Include 20-35% of daily calories from fats.",
    "Complex carbs (like oats, rice, quinoa) provide long-lasting energy for workouts. Eat 2-3 hours before exercise for best performance.",
    "Drinking water helps metabolism, digestion, and physical performance. Aim for 8-10 glasses daily, more during exercise.",
    "Weight loss = calorie deficit; weight gain = calorie surplus. Both require consistency. Track your progress for best results.",
    "Micronutrients like magnesium, iron, and vitamin D support muscle and energy systems. Consider a multivitamin if deficient.",
    "Supplements like creatine, whey protein, and multivitamins can support fitness goals. Creatine is the most researched supplement.",
    "Rest days are crucial for muscle growth and injury prevention. Overtraining can lead to burnout and decreased performance.",
    "Form is more important than weight. Poor form can cause injuries and limit progress. Start light, perfect your technique.",
    "Consistency beats perfection. Working out 3-4 times weekly consistently is better than sporadic intense sessions.",
    "Mind-muscle connection improves workout effectiveness. Focus on feeling the target muscle work during exercises.",
    "Progressive overload principle: gradually increase resistance, reps, or sets to continue making gains.",
    "Compound exercises (squats, deadlifts, bench press) work multiple muscle groups and are more efficient than isolation exercises.",
    "HIIT (High-Intensity Interval Training) burns more calories in less time and improves cardiovascular fitness.",
    "Flexibility and mobility work prevent injuries and improve performance. Include stretching in your routine.",
    "Nutrition timing matters: eat protein within 30 minutes post-workout for optimal muscle recovery.",
    "Mental health and fitness are connected. Exercise reduces stress, anxiety, and improves mood through endorphin release."
]

# === TF-IDF + SIMILARITY SEARCH (cached) ===
@st.cache_resource
def setup_tfidf(chunks):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = setup_tfidf(kb_chunks)

# === ENHANCED ASK FUNCTION ===
def ask(query, k=4):
    start_time = time.time()
    
    # Transform query using the same vectorizer
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top k most similar chunks
    top_indices = similarities.argsort()[-k:][::-1]
    top_context = "\n".join([kb_chunks[i] for i in top_indices])
    
    # Get similarity scores for visualization
    top_scores = similarities[top_indices]
    
    processing_time = time.time() - start_time

    prompt = f"""You are a knowledgeable fitness expert. Answer the fitness-related question below using ONLY the provided context. 
    Be helpful, encouraging, and provide actionable advice. Keep responses under 150 words.

Context:
{top_context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    
    total_time = time.time() - start_time
    
    return {
        'answer': response.choices[0].message.content.strip(),
        'context_used': [kb_chunks[i] for i in top_indices],
        'similarity_scores': top_scores.tolist(),
        'processing_time': processing_time,
        'total_time': total_time
    }

# === STREAMLIT UI ===
st.set_page_config(
    page_title="🏋️ Advanced Fitness RAG Chatbot",
    page_icon="🏋️",
    layout="wide"
)

# Sidebar for demo controls
with st.sidebar:
    st.title("🎛️ Demo Controls")
    st.markdown("---")
    
    # Model info
    st.subheader("🤖 Model Info")
    st.info(f"**LLM:** Llama 3.1-8B (Groq)")
    st.info(f"**Vector Search:** TF-IDF + Cosine Similarity")
    st.info(f"**Knowledge Base:** {len(kb_chunks)} fitness facts")
    
    # Demo questions
    st.subheader("💡 Demo Questions")
    demo_questions = [
        "How much protein should I eat for muscle growth?",
        "What's the best cardio exercise for fat loss?",
        "How important is sleep for fitness progress?",
        "What supplements should I take?",
        "How often should I work out?",
        "What's the best way to build muscle?"
    ]
    
    for i, question in enumerate(demo_questions):
        if st.button(f"Q{i+1}: {question[:30]}...", key=f"demo_{i}"):
            st.session_state.user_input = question

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🏋️ Advanced Fitness RAG Chatbot")
    st.markdown("**Powered by Retrieval-Augmented Generation (RAG)**")
    st.markdown("Ask about training, nutrition, recovery, supplements, and more!")

with col2:
    st.image("https://img.icons8.com/color/96/000000/dumbbell.png", width=80)
    st.markdown("**Real-time AI-powered fitness advice**")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# Debug toggle
if st.checkbox("🔧 Show Debug Info", value=st.session_state.show_debug):
    st.session_state.show_debug = True
else:
    st.session_state.show_debug = False

# User input
user_input = st.text_input("💬 Ask your fitness question:", value=st.session_state.get("user_input", ""))
st.session_state.user_input = ""

if user_input:
    with st.spinner("🤔 Thinking..."):
        result = ask(user_input)
        st.session_state.history.append((user_input, result))

# Display results
if st.session_state.history:
    st.markdown("---")
    st.subheader("💬 Conversation History")
    
    for i, (user, result) in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Q{i+1}: {user}", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**🤖 Answer:** {result['answer']}")
                
                if st.session_state.show_debug:
                    st.markdown("---")
                    st.markdown("**🔍 Debug Information:**")
                    st.markdown(f"**Processing Time:** {result['processing_time']:.3f}s")
                    st.markdown(f"**Total Time:** {result['total_time']:.3f}s")
                    
                    # Show context used
                    st.markdown("**📚 Context Retrieved:**")
                    for j, (context, score) in enumerate(zip(result['context_used'], result['similarity_scores'])):
                        st.markdown(f"**{j+1}.** (Score: {score:.3f}) {context}")
            
            with col2:
                # Similarity score visualization
                if st.session_state.show_debug:
                    st.markdown("**📊 Similarity Scores:**")
                    for j, score in enumerate(result['similarity_scores']):
                        st.progress(score)
                        st.caption(f"Chunk {j+1}: {score:.3f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit, Groq, and RAG Architecture</p>
    <p>Knowledge Base: 20 curated fitness facts | Vector Search: TF-IDF + Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)