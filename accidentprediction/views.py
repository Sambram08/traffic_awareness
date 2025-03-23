import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import logging
from .model_loader import predict_model  # Import the function correctly

logger = logging.getLogger(__name__)

@csrf_exempt
def accident_detect(request):
    if request.method == 'GET':
        return JsonResponse({"message": "Send a POST request with JSON input to get predictions."})

    if request.method == 'POST':
        try:
            # ✅ Parse JSON input correctly
            input_data = json.loads(request.body)

            # ✅ Call the model prediction function
            result = predict_model(input_data)  

            # ✅ Return the result as a JSON response
            return JsonResponse(result)  

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format. Please send valid JSON data."}, status=400)

        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}")
            return JsonResponse({"error": f"Internal Server Error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)
