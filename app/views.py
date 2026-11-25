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
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.decorators import parser_classes
from detectgptmodel.document_reader import DocumentReadError, SUPPORTED_EXTENSIONS, read_document
from .detectors import DEFAULT_METHOD, METHODS, get_detector, list_methods

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB

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

def _read_upload(upload):
    """Đọc file upload -> (text, source_info). Ném DocumentReadError nếu lỗi."""
    if upload.size > MAX_UPLOAD_BYTES:
        raise DocumentReadError(f"File quá lớn (tối đa {MAX_UPLOAD_BYTES // (1024 * 1024)}MB).")
    doc = read_document(upload.read(), upload.name)
    return doc.text, doc.to_dict()


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def detect_methods(request: Request):
    return Response({
        'default': DEFAULT_METHOD,
        'methods': list_methods(),
        'supported_files': SUPPORTED_EXTENSIONS,
        'max_upload_mb': MAX_UPLOAD_BYTES // (1024 * 1024),
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def extract_text(request: Request):
    upload = request.FILES.get('file')
    if upload is None:
        return Response({'error': "No file provided"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        text, source = _read_upload(upload)
    except DocumentReadError as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    return Response({'text': text, 'source': source})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([JSONParser, MultiPartParser, FormParser])
def detect_ai(request: Request):
    """Nhận JSON {text, method?} hoặc multipart {file | text, method?}.

    method mặc định là ``prob_mean``; ``fastgpt`` là phương pháp cũ.
    """
    method = request.data.get('method') or DEFAULT_METHOD
    if method not in METHODS:
        return Response({'error': f"Unknown method '{method}'", 'methods': list(METHODS)},
                        status=status.HTTP_400_BAD_REQUEST)

    source = None
    upload = request.FILES.get('file')
    if upload is not None:
        try:
            user_text, source = _read_upload(upload)
        except DocumentReadError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    else:
        user_text = request.data.get('text', '')

    if not user_text or not user_text.strip():
        return Response({'error': "No text provided"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        detector = get_detector(method)
    except Exception as e:  # thiếu file mô hình, lỗi tải model...
        return Response({'error': f"Không nạp được phương pháp '{method}': {e}"},
                        status=status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        result = detector.predict(user_text)
    except ValueError as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    if method == 'fastgpt':
        # Giữ nguyên các trường cũ, thêm trường chung để FE hiển thị thống nhất
        result = {**result, 'method': 'fastgpt', 'score': float(result['fast_gpt'])}
        result = {k: float(v) if hasattr(v, 'item') else v for k, v in result.items()}

    if source is not None:
        result['source'] = source
        result['text'] = user_text

    return JsonResponse(result, safe=False)
