import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "google/flan-t5-small"

faq_data = [
    {"question": "What are your working hours?", "answer": "Our working hours are 9 AM to 6 PM, Monday to Saturday."},
    {"question": "How can I track my order?", "answer": "You can track your order using the tracking link or order ID."},
    {"question": "What is your return policy?", "answer": "Eligible products can be returned within 7 days of delivery."},
    {"question": "How can I get a refund?", "answer": "Please contact customer support with your order ID to request a refund."},
    {"question": "How can I contact support?", "answer": "Please contact our customer support team through the support email listed on our website."},
    {"question": "Do you provide online payment?", "answer": "Yes, we accept online payments through the payment options shown at checkout."},
    {"question": "How long does delivery take?", "answer": "Delivery usually takes 3 to 5 working days, depending on your location."},
    {"question": "Can I cancel my order?", "answer": "Please contact customer support as soon as possible with your order ID to request cancellation."},
]

questions = [item["question"] for item in faq_data]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
question_vectors = vectorizer.fit_transform(questions)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_ai_answer(user_message):
    prompt = (
        "You are a polite customer support assistant. Answer briefly and helpfully. "
        "Do not invent company policies, prices, or order details. If information is missing, "
        "ask the customer to contact the support team. Question: " + user_message
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def get_response(user_message):
    if not user_message or not user_message.strip():
        return "Please enter a question."
    user_vector = vectorizer.transform([user_message])
    scores = cosine_similarity(user_vector, question_vectors)[0]
    best_match = int(scores.argmax())
    if float(scores[best_match]) >= 0.20:
        return faq_data[best_match]["answer"]
    return generate_ai_answer(user_message)

demo = gr.ChatInterface(
    fn=lambda message, history: get_response(message),
    title="AI-Powered Customer Support Chatbot",
    description="Ask questions about orders, delivery, returns, refunds and support.",
    examples=["What are your working hours?", "How can I track my order?", "How can I get a refund?"],
)

if __name__ == "__main__":
    demo.launch()
