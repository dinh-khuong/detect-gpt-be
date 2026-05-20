from django.urls import  path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from . import views

urlpatterns = [
    # path('users/', views.get_users, name="user"),
    path('auth/login/', TokenObtainPairView.as_view(), name="Auth login"),
    path('auth/register/', views.RegisterView.as_view(), name="Auth Register"),

    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    path('auth/me/', views.UserMeView.as_view(), name="Auth me"),
    path('detect-ai/', views.detect_ai, name="Detect AI"),
]
