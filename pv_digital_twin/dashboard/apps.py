from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class DashboardConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "dashboard"
    verbose_name = "光伏数字孪生仪表盘"

    def ready(self):
        """Django应用启动时的初始化"""
        logger.info("光伏数字孪生仪表盘应用已初始化")
        logger.info("使用模拟仿真系统")
