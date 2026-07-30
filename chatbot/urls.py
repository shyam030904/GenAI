from django.urls import path
from .views import chatbot_view, send_message, clear_chat

urlpatterns = [
    path('', chatbot_view, name='chatbot'),
    path('send/', send_message, name='send_message'),
    path('clear/', clear_chat, name='clear_chat'),
]
