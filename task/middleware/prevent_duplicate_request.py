import json
from django.core.cache import cache
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin
import hashlib
from config import CACHE_TIME


class PreventDuplicateRequestMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if request.method == 'POST':
            # 获取post的json
            try:
                post_data = json.loads(request.body.decode('utf-8'))
                url_path = request.path
                request_flag = (str(post_data) + url_path).encode('utf-8')
                request_hash = hashlib.sha256(request_flag).hexdigest()
                if cache.get(request_hash):
                    return JsonResponse({
                        'code': 600,
                        'msg': '请求重复',
                    })
                cache.set(request_hash, True, timeout=CACHE_TIME)
            except json.decoder.JSONDecodeError:
                return JsonResponse({
                    'code': 500,
                    'msg': '请求json为空',
                })
