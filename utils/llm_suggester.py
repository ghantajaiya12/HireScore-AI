from transformers import pipeline

# Load the model
generator = pipeline("text-generation", model="gpt2")

def get_resume_suggestions(jd_text, resume_text):
    # Limit input to first 500 characters each
    jd_text = jd_text[:500]
    resume_text = resume_text[:500]

    prompt = f"Job Description: {jd_text}\nResume: {resume_text}\nHow can the resume be improved?"

    result = generator(prompt, num_return_sequences=1, max_new_tokens=100)

    return result[0]["generated_text"]
