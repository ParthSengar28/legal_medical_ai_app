from transformers import pipeline
import gradio as gr

# Load a text2text-generation pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-xl")

# Define the function that generates answers
def answer_question(question):
    prompt = f"Answer the following legal or medical question accurately:\n\n{question}"
    response = generator(prompt, max_length=256, do_sample=True)[0]["generated_text"]
    return response.strip()

# Build the Gradio interface
gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=3, placeholder="Enter your legal or medical question here..."),
    outputs="text",
    title="AI Legal & Medical Assistant",
    description="Ask any legal or medical question and get AI-generated answers. Powered by Flan-T5."
).launch()
