import google.generativeai as genai

genai.configure(api_key="AIzaSyA1PUAreQQkSOcFhgLVKRWb4MyPH7R-h-4")

model = genai.GenerativeModel('gemini-2.0-flash')

def generate_text(prompt):
    response = model.generate_content(prompt)
    return response.text
