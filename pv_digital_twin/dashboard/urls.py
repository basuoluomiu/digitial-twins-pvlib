from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("fault/", views.fault_diagnosis, name="fault_diagnosis"),
    path("settings/", views.system_settings, name="system_settings"),
    path("logs/", views.simulation_logs, name="simulation_logs"),
    path("schedule-faults/", views.schedule_faults_view, name="schedule_faults"),
]
