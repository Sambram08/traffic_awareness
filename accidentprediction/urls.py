

from django.urls import path
from .views import accident_detect

urlpatterns = [
    # path('', accident_model, name="welcome"),
    path('predict/',accident_detect, name="predict"),
]
