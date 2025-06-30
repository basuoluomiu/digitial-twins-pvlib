from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .pv_model_adapter import PVModelAdapter
import os
from .config import SimulationConfig


def dashboard(request):
    """主仪表盘视图"""
    pv_model = PVModelAdapter.get_instance()
    system_info = pv_model.get_system_info()

    # 获取适配器类型信息
    adapter_type = type(pv_model).__name__

    # 仿真模式说明
    simulation_mode_str = "模拟仿真系统"
    print(f"\n当前使用的适配器: {adapter_type}")
    print(f"仿真模式: {simulation_mode_str}")

    # 获取基本系统信息
    context = {
        "installed_capacity": system_info.get("installed_capacity", "N/A"),
        "current_power": system_info.get("current_power", 0),
        "max_power_today": system_info.get("max_power_today", 0),
        "max_ghi_today": system_info.get("max_ghi_today", 0),
        "max_efficiency_today": system_info.get("max_efficiency_today", 0),
        "daily_energy": system_info.get("daily_energy", 0),
        "current_temp_air": system_info.get("current_temp_air", 25),
        "current_temp_cell": system_info.get("current_temp_cell", 25),
        "active_tab": "dashboard",
        # 添加仿真模式信息到上下文
        "simulation_mode": simulation_mode_str,
        "adapter_type": adapter_type,
    }

    return render(request, "dashboard/dashboard.html", context)


def fault_diagnosis(request):
    """故障诊断视图"""
    pv_model = PVModelAdapter.get_instance()
    # 获取自动检测到的异常
    raw_anomalies = pv_model.get_detected_anomalies()

    fault_causes_list = []
    if raw_anomalies:
        for anomaly in raw_anomalies:
            fault_causes_list.append(
                {
                    "timestamp": anomaly.get("timestamp", "N/A"),
                    "type": anomaly.get("type", "未知类型"),
                    "severity": anomaly.get("severity", 0),
                }
            )

    context = {
        # fault_causes 传递给模板的是处理过的列表
        "fault_causes": fault_causes_list,
        "active_tab": "fault_diagnosis",
        # "recommendations" 已在模板中硬编码，如果需要动态的，可以在这里添加
    }

    return render(request, "dashboard/fault_diagnosis.html", context)


def system_settings(request):
    """系统设置视图"""
    pv_model = PVModelAdapter.get_instance()

    # 获取当前设置
    # 系统总容量应该从逆变器参数或通过模块数和单模块功率计算得到
    pv_twin_instance = pv_model.pv_twin

    # 尝试从逆变器参数获取总直流功率
    total_system_capacity_watts = getattr(
        pv_twin_instance, "inverter_parameters_dict", {}
    ).get("pdc0")

    # 获取当前的组串数量
    strings_count = getattr(pv_twin_instance, "strings_per_inverter", 1)

    # 获取每串模块数
    modules_per_string = getattr(pv_twin_instance, "modules_per_string", 10)

    # 如果逆变器参数中没有，则尝试从模块参数计算
    if total_system_capacity_watts is None:
        single_module_pdc0 = getattr(pv_twin_instance, "module_parameters", {}).get(
            "pdc0", 300.0
        )
        total_system_capacity_watts = (
            single_module_pdc0 * modules_per_string * strings_count
        )

    system_capacity_kw = (
        total_system_capacity_watts / 1000.0
        if total_system_capacity_watts is not None
        else 0.3
    )  # 默认0.3kW

    temp_coefficient_decimal = getattr(pv_twin_instance, "module_parameters", {}).get(
        "gamma_pdc", -0.004
    )
    temp_coefficient_percent = temp_coefficient_decimal * 100

    system_loss_value = getattr(pv_twin_instance, "loss_parameters", {}).get(
        "system_loss_input_percentage", 14.0
    )

    anomaly_threshold_value = (
        getattr(pv_model.anomaly_model, "threshold", 200)
        if pv_model.anomaly_model
        else 200
    )

    context = {
        "latitude": pv_twin_instance.location.latitude,
        "longitude": pv_twin_instance.location.longitude,
        "system_capacity": system_capacity_kw,
        "strings_count": strings_count,  # 添加组串数量到上下文
        "modules_per_string": modules_per_string,  # 添加每串模块数到上下文
        "temp_coeff": temp_coefficient_percent,
        "system_loss": system_loss_value,
        "fault_threshold": anomaly_threshold_value,
        "active_tab": "system_settings",
    }

    return render(request, "dashboard/system_settings.html", context)


def simulation_logs(request):
    """仿真日志视图"""
    pv_model = PVModelAdapter.get_instance()
    logs = pv_model.get_simulation_logs()

    context = {
        "logs": logs,
        "active_tab": "simulation_logs",
    }

    return render(request, "dashboard/simulation_logs.html", context)


# 新增视图函数，用于处理计划的故障
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt  # 如果前端不方便发送CSRF token，可以临时用这个，但生产环境要注意安全
def schedule_faults_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            faults_to_schedule = data.get("faults", [])

            if not isinstance(faults_to_schedule, list):
                return JsonResponse(
                    {
                        "status": "error",
                        "message": "无效的数据格式，需要一个故障列表。",
                    },
                    status=400,
                )

            # 验证每个故障项的结构
            for fault in faults_to_schedule:
                if not all(
                    k in fault for k in ["type", "start_hour", "end_hour", "severity"]
                ):
                    return JsonResponse(
                        {"status": "error", "message": "故障项缺少必要字段。"},
                        status=400,
                    )
                if not (
                    0 <= fault["start_hour"] <= 23
                    and 0 <= fault["end_hour"] <= 23
                    and fault["start_hour"] < fault["end_hour"]
                ):
                    return JsonResponse(
                        {"status": "error", "message": "无效的小时范围。"}, status=400
                    )
                if not (0.05 <= fault["severity"] <= 1.0):
                    return JsonResponse(
                        {"status": "error", "message": "无效的严重程度值。"}, status=400
                    )

            pv_model_adapter = PVModelAdapter.get_instance()
            # 假设 PVModelAdapter 有一个方法来处理这些计划的故障
            success, message = pv_model_adapter.schedule_faults_for_next_simulation(
                faults_to_schedule
            )

            if success:
                return JsonResponse({"status": "success", "message": message})
            else:
                return JsonResponse({"status": "error", "message": message}, status=400)

        except json.JSONDecodeError:
            return JsonResponse(
                {"status": "error", "message": "无效的JSON数据。"}, status=400
            )
        except Exception as e:
            # 记录异常 e
            print(f"Error in schedule_faults_view: {e}")
            return JsonResponse(
                {"status": "error", "message": f"服务器内部错误: {str(e)}"}, status=500
            )

    return JsonResponse({"status": "error", "message": "仅支持POST请求。"}, status=405)


def index(request):
    """首页视图"""
    # 获取适配器类型信息
    adapter = PVModelAdapter.get_instance()
    adapter_type = type(adapter).__name__

    # 添加到上下文
    context = {
        "simulation_mode": "模拟仿真系统",
        "adapter_type": adapter_type,
        "title": "PV数字孪生系统",
    }
    return render(request, "dashboard/index.html", context)
