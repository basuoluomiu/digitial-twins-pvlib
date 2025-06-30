# PV Digital Twin Django Application

## 📋 项目概述

这是一个基于Django的光伏数字孪生系统Web应用，提供光伏系统仿真、性能监控、异常检测和数据分析功能。

### 核心功能
- 🔋 光伏系统仿真
- 📊 系统性能监控和数据可视化
- 🔍 异常检测和故障诊断
- 📈 历史数据分析和趋势预测
- 🌐 响应式Web界面
- ⚙️ 灵活的配置管理系统

### 技术栈
- **后端**：Django 4.2
- **前端**：Bootstrap 5、ECharts 5
- **数据处理**：Pandas、NumPy
- **仿真模型**：模拟光伏系统模型
- **异常检测**：基于规则的异常检测
- **机器学习**：简单的统计分析

## 🛠️ 运行方式

### 环境要求
- Python 3.8+
- Django 4.2+
- 其他依赖见 `requirements.txt`

### 快速启动

1. **安装依赖**：
```bash
pip install -r requirements.txt
```

2. **数据库迁移**：
```bash
python manage.py migrate
```

3. **启动开发服务器**：
```bash
python manage.py runserver
```

4. **访问应用**：
打开浏览器访问 http://127.0.0.1:8000

### 高级启动选项

**指定端口和主机**：
```bash
python manage.py runserver 0.0.0.0:8080
```

**生产环境运行**：
```bash
# 设置生产环境变量
export DJANGO_SETTINGS_MODULE=pv_digital_twin.settings_production
python manage.py runserver
```

## ⚙️ 配置选项

### 仿真系统配置

项目使用模拟系统模式：
```bash
# 不需要设置环境变量，系统默认使用模拟仿真
python manage.py runserver
```
- 使用内置的轻量级仿真引擎
- 快速响应，适合开发和演示
- 数据基于简化算法生成

### 配置参数说明

在 `settings.py` 中可以配置以下参数：

```python
# 仿真系统配置
SIMULATION_CONFIG = {
    'CACHE_TIMEOUT': 300,                        # 缓存超时时间（秒）
    'MAX_SIMULATION_POINTS': 180,                # 最大仿真数据点数
    'RESPONSE_TIME_LIMIT': 2.0,                  # API响应时间限制（秒）
}
```

## 🏗️ 系统架构

### 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    Django Web Application                   │
├─────────────────────────────────────────────────────────────┤
│                     API Layer (views.py)                   │
├─────────────────────────────────────────────────────────────┤
│                   Configuration Manager                     │
│                      (config.py)                           │
├─────────────────────────────────────────────────────────────┤
│              Adapter Pattern                               │
│  ┌─────────────────────┐                                    │
│  │   PVModelAdapter    │                                    │
│  │   (模拟系统)         │                                    │
│  └─────────────────────┘                                    │
├─────────────────────────────────────────────────────────────┤
│                      Backend Systems                       │
│  ┌─────────────────────┐                                    │
│  │   模拟仿真引擎       │                                    │
│  │   (内置)            │                                    │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

### 组件关系说明

1. **配置管理器 (Configuration Manager)**
   - 统一管理系统配置

2. **适配器 (Adapter)**
   - 提供统一的API接口
   - 支持运行时配置

3. **后端系统 (Backend Systems)**
   - 模拟仿真引擎：轻量级，快速响应

### 项目结构

```
pv_digital_twin/
├── dashboard/                    # 主要应用模块
│   ├── templates/               # 页面模板
│   ├── config.py               # 配置管理模块
│   └── pv_model_adapter.py     # PV模型适配器
├── api/                        # API应用，提供数据接口
├── static/                     # 静态文件
├── templates/                  # 基础模板
├── pv_digital_twin/           # 项目主设置
│   └── settings.py            # Django设置
├── manage.py                  # Django管理脚本
└── requirements.txt           # 项目依赖
```

## 🌐 API接口

### 核心API端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/simulation-data/` | GET | 获取仿真数据用于图表显示 |
| `/api/system-info/` | GET | 获取系统基本信息和状态 |
| `/api/daily-energy/` | GET | 获取每日能量产量数据 |
| `/api/detected-anomalies/` | GET | 获取检测到的异常信息 |
| `/api/simulation-logs/` | GET | 获取仿真运行日志 |

### API响应格式

#### 仿真数据 (`/api/simulation-data/`)
```json
{
  "timestamps": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
  "ac_power": [1200, 1500],
  "dc_power": [1300, 1600],
  "temp_air": [25, 28],
  "temp_cell": [30, 35],
  "ghi": [800, 900],
  "efficiency": [85.2, 87.1]
}
```

#### 系统信息 (`/api/system-info/`)
```json
{
  "installed_capacity": 5.0,
  "current_power": 1500,
  "max_power_today": 4200,
  "max_ghi_today": 1000,
  "max_efficiency_today": 92.5,
  "daily_energy": 25.6,
  "current_temp_air": 25,
  "current_temp_cell": 30
}
```

#### 异常检测 (`/api/detected-anomalies/`)
```json
[
  {
    "timestamp": "2024-01-01 14:30:00",
    "type": "EFFICIENCY_DROP",
    "severity": 0.85,
    "description": "系统效率显著下降",
    "value": 245.6
  },
  {
    "timestamp": "2024-01-01 15:15:00",
    "type": "TEMPERATURE_HIGH",
    "severity": 0.62,
    "description": "组件温度异常高",
    "value": 75.3
  }
]
```

## 📱 使用说明

### 仪表盘功能

**系统概览**：
- 实时显示系统状态和关键指标
- 自动更新数据，展示最近48小时趋势

**数据可视化**：
- 功率输出趋势图
- 环境条件监控
- 效率分析图表
- 异常检测结果

### 故障诊断

**异常检测**：
- 基于规则的异常检测
- 提供异常描述和建议维护措施

**性能分析**：
- 预期vs实际功率对比
- 效率趋势分析
- 环境因素影响评估

### 系统设置

**参数配置**：
- 系统位置和容量设置
- 温度系数和损耗参数
- 仿真精度和频率配置

**数据管理**：
- 历史数据导出
- 配置备份和恢复
- 系统维护工具

## 🚨 故障排除

### 常见问题

**1. 服务器启动失败**
```bash
# 检查依赖是否正确安装
pip install -r requirements.txt

# 检查数据库迁移
python manage.py migrate

# 检查Django配置
python manage.py check
```

**2. API响应异常**
- 检查适配器是否正确初始化
- 查看Django日志文件
- 验证数据格式转换是否正常

**3. 数据显示异常**
- 确认仿真数据是否正常生成
- 检查前端JavaScript控制台错误
- 验证API端点返回数据格式

### 调试模式

**启用详细日志**：
```python
# 在settings.py中添加
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'dashboard': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}
```

## 📞 技术支持

如遇到问题，请按以下步骤排查：

1. **查看日志**：检查Django控制台输出和仿真日志
2. **验证配置**：确认环境变量和Django设置正确
3. **测试API**：直接访问API端点验证数据
4. **重启服务**：重启Django服务器
