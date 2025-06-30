from django.urls import path
from . import views

app_name = "api"

urlpatterns = [
    # 原有API端点
    path("simulation-data/", views.simulation_data, name="simulation_data"),
    path("daily-energy/", views.daily_energy, name="daily_energy"),
    path("detected-anomalies/", views.anomaly_data, name="anomaly_data"),
    path("system-info/", views.system_info, name="system_info"),
    path("simulation-logs/", views.simulation_logs, name="simulation_logs"),
    path("apply-settings/", views.apply_settings, name="apply_settings"),
    path("reset-simulation/", views.reset_simulation, name="reset_simulation"),
    path("pause-simulation/", views.pause_simulation, name="pause_simulation"),
    path("resume-simulation/", views.resume_simulation, name="resume_simulation"),
    path("simulation-status/", views.simulation_status, name="simulation_status"),
    path("capacity-options/", views.capacity_options, name="capacity_options"),
]
