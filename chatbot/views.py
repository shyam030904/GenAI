from django.shortcuts import render
from .gemini import generate_text

def chatbot_view(request):
    response = ""
    if request.method == "POST":
        prompt = request.POST.get("prompt")
        if prompt:
            response = generate_text(prompt)
    return render(request, "chatbot/chat.html", {"response": response})
