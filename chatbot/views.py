import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from .models import ChatMessage
from .gemini import generate_response


def chatbot_view(request):
    """Renders the main chat UI and loads existing session history."""
    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key
    messages = ChatMessage.objects.filter(session_key=session_key)

    return render(request, 'chatbot/chat.html', {'messages': messages})


@csrf_exempt
@require_POST
def send_message(request):
    """AJAX endpoint: receives a prompt, returns AI response as JSON."""
    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key

    try:
        data = json.loads(request.body)
        prompt = data.get('prompt', '').strip()
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({'error': 'Invalid request body.'}, status=400)

    if not prompt:
        return JsonResponse({'error': 'Prompt cannot be empty.'}, status=400)

    # Build Gemini-compatible history from DB
    past_messages = ChatMessage.objects.filter(session_key=session_key)
    history = [
        {'role': msg.role, 'parts': [msg.content]}
        for msg in past_messages
    ]

    # Get response from Gemini
    reply = generate_response(prompt, history)

    # Persist both messages
    ChatMessage.objects.create(session_key=session_key, role='user', content=prompt)
    ChatMessage.objects.create(session_key=session_key, role='model', content=reply)

    return JsonResponse({'reply': reply})


@csrf_exempt
@require_POST
def clear_chat(request):
    """Clears all messages for the current session."""
    if request.session.session_key:
        ChatMessage.objects.filter(session_key=request.session.session_key).delete()
    return JsonResponse({'status': 'cleared'})

