from django.http import JsonResponse
from rest_framework import permissions
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView, status
from .models import User
from .serializers import RegisterSerializer, UserSerializer
from rest_framework import generics, viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from detectgptmodel.fastgpt import get_sampling_discrepancy
from . import gptDetectModel

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.values('id', 'username')
    serializer_class = UserSerializer

# 1. Sign Up View
class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (permissions.AllowAny,)
    serializer_class = RegisterSerializer

# 2. Get Profile View (The "Me" endpoint)
class UserMeView(APIView):
    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def detect_ai(request: Request):
    user_text = request.data.get('text', '')
    if not user_text:
        return Response({'error': "No text provided"}, status=status.HTTP_400_BAD_REQUEST)
    # print(user_text)

    # org_dist, perturbate_dist = gptDetectModel.log_perterbate(user_text, 0.2, [(0.8, 30.0), ])[0]
    # dist = get_sampling_discrepancy(org_dist, perturbate_dist).item()

    return JsonResponse(gptDetectModel.predict(user_text), safe=False)


