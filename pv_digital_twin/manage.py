#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    # 将manage.py所在的目录添加到sys.path的最前面
    # 这确保了当Django查找'pv_digital_twin.settings'时，
    # 它会首先在当前目录中查找名为'pv_digital_twin'的子目录（包）。
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pv_digital_twin.settings")

    # 始终使用模拟仿真系统
    os.environ["USE_REAL_SIMULATION"] = "false"

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
