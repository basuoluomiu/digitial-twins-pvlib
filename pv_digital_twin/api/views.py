from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from dashboard.pv_model_adapter import PVModelAdapter


def simulation_data(request):
    """提供仿真数据的API端点"""
    pv_model = PVModelAdapter.get_instance()
    data = pv_model.get_simulation_data()
    return JsonResponse(data)


def daily_energy(request):
    """提供每日能量数据的API端点"""
    pv_model = PVModelAdapter.get_instance()
    data = pv_model.get_daily_energy()
    return JsonResponse(data, safe=False)


def anomaly_data(request):
    """提供异常检测数据的API端点"""
    pv_model = PVModelAdapter.get_instance()
    data = pv_model.get_detected_anomalies()
    return JsonResponse(data, safe=False)


def system_info(request):
    """提供系统信息的API端点"""
    pv_model = PVModelAdapter.get_instance()
    data = pv_model.get_system_info()
    return JsonResponse(data)


def simulation_logs(request):
    """提供仿真日志的API端点"""
    pv_model = PVModelAdapter.get_instance()
    logs = pv_model.get_simulation_logs()
    return JsonResponse({"logs": logs}, safe=False)


@csrf_exempt
def apply_settings(request):
    """应用系统设置的API端点"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            pv_model = PVModelAdapter.get_instance()
            success, message = pv_model.apply_settings(data)
            return JsonResponse(
                {
                    "success": success,
                    "message": message,
                }
            )
        except json.JSONDecodeError:
            return JsonResponse(
                {
                    "success": False,
                    "message": "无效的JSON数据",
                },
                status=400,
            )
        except Exception as e:
            return JsonResponse(
                {
                    "success": False,
                    "message": f"发生错误: {str(e)}",
                },
                status=500,
            )
    else:
        return JsonResponse(
            {
                "success": False,
                "message": "仅支持POST请求",
            },
            status=405,
        )


@csrf_exempt
def reset_simulation(request):
    """重置仿真数据并重新运行仿真的API端点"""
    if request.method == "POST":
        try:
            pv_model = PVModelAdapter.get_instance()
            result = pv_model.reset_simulation()
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse(
                {
                    "status": "error",
                    "message": f"重置仿真时发生错误: {str(e)}",
                },
                status=500,
            )
    else:
        return JsonResponse(
            {
                "status": "error",
                "message": "仅支持POST请求",
            },
            status=405,
        )


@csrf_exempt
def pause_simulation(request):
    """暂停仿真的API端点"""
    if request.method == "POST":
        try:
            pv_model = PVModelAdapter.get_instance()
            result = pv_model.pause_simulation()
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse(
                {
                    "status": "error",
                    "message": f"暂停仿真时发生错误: {str(e)}",
                },
                status=500,
            )
    else:
        return JsonResponse(
            {
                "status": "error",
                "message": "仅支持POST请求",
            },
            status=405,
        )


@csrf_exempt
def resume_simulation(request):
    """恢复仿真的API端点"""
    if request.method == "POST":
        try:
            pv_model = PVModelAdapter.get_instance()
            result = pv_model.resume_simulation()
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse(
                {
                    "status": "error",
                    "message": f"恢复仿真时发生错误: {str(e)}",
                },
                status=500,
            )
    else:
        return JsonResponse(
            {
                "status": "error",
                "message": "仅支持POST请求",
            },
            status=405,
        )


def simulation_status(request):
    """获取仿真状态的API端点"""
    pv_model = PVModelAdapter.get_instance()
    status = pv_model.get_simulation_status()
    return JsonResponse(status)


def capacity_options(request):
    """获取系统可用容量选项的API端点"""
    pv_model = PVModelAdapter.get_instance()
    options = pv_model.get_available_capacity_options()
    return JsonResponse({"options": options}, safe=False)
