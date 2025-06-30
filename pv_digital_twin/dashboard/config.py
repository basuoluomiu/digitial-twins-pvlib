"""
配置管理模块

提供仿真系统的配置管理功能。
只支持模拟仿真系统。
"""

import os
from django.conf import settings


class SimulationConfig:
    """仿真系统配置管理类"""

    @staticmethod
    def use_real_simulation():
        """
        检查是否使用真实仿真系统

        在只支持模拟系统的情况下，始终返回False

        Returns:
            bool: 总是返回False，表示使用模拟系统
        """
        return False

    @staticmethod
    def get_simulation_config():
        """
        获取仿真系统配置

        Returns:
            dict: 仿真系统配置字典
        """
        return getattr(
            settings,
            "SIMULATION_CONFIG",
            {
                "USE_REAL_SIMULATION": False,
                "CACHE_TIMEOUT": 300,  # 5分钟缓存
                "MAX_SIMULATION_POINTS": 180,  # 最大数据点数
                "RESPONSE_TIME_LIMIT": 2.0,  # 响应时间限制（秒）
            },
        )


class ConfigurationError(Exception):
    """配置错误异常"""

    pass


def get_adapter_class():
    """
    获取适当的适配器类 - 只返回模拟适配器

    Returns:
        class: PVModelAdapter类
    """
    from .pv_model_adapter import PVModelAdapter

    return PVModelAdapter
