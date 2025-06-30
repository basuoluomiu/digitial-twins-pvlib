# 光伏数字孪生平台

基于 pvlib 库的光伏系统数字孪生平台，用于模拟、分析和优化光伏系统性能。

## 功能特点

- **光伏系统性能模拟**：利用 pvlib 库模拟光伏系统在不同环境条件下的表现
- **实时数据监控和分析**：可视化系统性能数据，提供全面的系统性能视图
- **系统性能预测**：基于天气数据预测未来系统输出
- **设备故障诊断**：使用 LSTM+KAN 模型检测系统异常表现
- **系统优化建议**：根据诊断结果提供优化建议
- **异常工况模拟**：支持热斑效应、部分阴影、组件老化等异常工况模拟

## 项目进度

当前项目整体完成度约为 35%，主要集中在数据层、模型层和展示层的基础功能。

- **数据层**：气象数据模块和异常特征库已完成，历史数据模块进行中(~25%)
- **模型层**：光伏组件模型、逆变器模型基本完成，系统集成模型和异常工况模型完成度约 85-90%
- **仿真层**：时序仿真引擎基本框架已完成(~40-90%)，场景管理器进行中(~10%)
- **优化层**：控制策略库部分功能进行中(~15%)，其他模块尚未开始
- **展示层**：基于 Django 的 Web 界面基本完成(~70-80%)

详细进度请查看`project_progress.md`文件。

## 系统架构

平台采用分层架构设计，主要组件构成：

```
光伏系统数字孪生仿真平台
├── 数据层
│   ├── 历史数据模块
│   ├── 气象数据模块
│   └── 异常特征库
├── 模型层
│   ├── 光伏组件模型
│   ├── 逆变器模型
│   ├── 系统集成模型
│   └── 异常工况模型
├── 仿真层
│   ├── 时序仿真引擎
│   ├── 场景管理器
│   └── 硬件加速接口
├── 优化层
│   ├── 控制策略库
│   ├── 多目标优化引擎
│   └── 决策支持系统
└── 展示层
    ├── 实时可视化模块
    ├── 结果分析模块
    └── Web交互界面
```

具体实现包括：

- **PV 数字孪生模型** (`src/model/pv_model.py`)：基于 pvlib 的光伏系统数学模型，包含详细的组件和系统级建模。
- **逆变器模型** (`src/model/inverter_model.py`)：实现 Sandia 逆变器效率模型和 MPPT 算法模拟。
- **数据层模块** (`src/data_layer/`)：负责历史数据、气象数据的处理和异常特征库的管理。
- **异常工况模型** (`src/model/anomaly_model.py`)：集成深度学习模型 (LSTMKATAutoencoder) 进行异常检测，并参数化模拟异常对数字孪生模型的影响。
- **深度学习模型架构** (`src/model/lstm_kan_architecture.py`)：包含 `LSTMKATAutoencoder` 及其依赖的自定义 PyTorch 层定义。
- **仿真层模块** (`src/simulation_layer/`)：包含时序仿真引擎 (`time_series_simulation_engine.py`)，负责管理和执行仿真事件。
- **Django Web 应用** (`pv_digital_twin/`)：基于 Django 的 Web 应用，提供仪表盘、故障诊断和系统设置功能。
- **工具类模块** (`src/utils/`)：提供数据处理等辅助功能。

## 最近改进

### 模型层优化

- ✅ 修复了 PVDigitalTwin 中 weather_df 未定义的问题
- ✅ 增强了逆变器模型，改进了 Sandia 参数处理，提高了鲁棒性
- ✅ 优化了异常工况模型，改进了参数映射和影响传播机制
- ✅ 增强了系统集成模型，支持更灵活的系统拓扑配置

### 仿真层优化

- ✅ 改进了时序仿真引擎的事件调度机制，提高了稳定性
- ✅ 增强了状态管理系统，支持更完善的模型状态跟踪
- ✅ 优化了仿真运行控制接口，提高了用户体验

### 展示层优化

- ✅ 将 Web 界面从 Dash 迁移到 Django 框架
- ✅ 改进了 Web 界面的响应性和用户体验
- ✅ 增强了数据可视化功能，使用 ECharts 提供更丰富的图表
- ✅ 优化了系统控制面板，增加了更多参数调整选项

## 快速开始

### 环境要求

- Python 3.7+
- 依赖包：见`requirements.txt`

### 安装步骤

1. 克隆此仓库

```bash
git clone <仓库URL>
cd 光伏数字孪生平台
```

2. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 设置故障检测模块
python setup_detect.py
```

3. 启动 Django 应用

```bash
cd pv_digital_twin
python manage.py runserver
```

4. 在浏览器中访问应用：http://127.0.0.1:8000/

## 使用指南

### 主界面

平台主界面分为三个主要标签页：

1. **系统仪表盘**：显示系统的实时性能指标和可视化图表
2. **故障诊断**：异常检测结果、原因分析和优化建议
3. **系统设置**：可调整系统参数，导入/导出数据

### 故障诊断功能

故障诊断模块使用 LSTM+KAN 自编码器模型检测光伏系统异常：

1. 模型通过学习正常运行数据的模式来检测异常
2. 可以识别各种类型的系统异常，包括组件故障、逆变器问题等
3. 提供故障原因分析和优化建议

### 自定义模拟

可以通过系统设置页面修改以下参数：

- 系统位置（经纬度）
- 系统容量
- 组件温度系数
- 系统损耗
- 故障检测阈值
- 异常工况参数（热斑效应、部分阴影等）

## 项目结构

```
光伏数字孪生平台/
├── pv_digital_twin/      # Django应用目录
│   ├── api/              # API应用
│   ├── dashboard/        # 仪表盘应用
│   │   └── pv_model_adapter.py  # PV模型适配器
│   ├── templates/        # HTML模板
│   ├── static/           # 静态文件
│   ├── manage.py         # Django管理脚本
│   └── requirements.txt  # Django依赖
├── src/                  # 源代码目录
│   ├── data_layer/       # 数据层模块
│   │   ├── meteorological_data_module.py  # 气象数据模块
│   │   ├── anomaly_feature_library.py     # 异常特征库
│   │   └── __init__.py
│   ├── model/            # 模型层模块
│   │   ├── pv_model.py   # 光伏系统模型
│   │   ├── inverter_model.py  # 逆变器模型
│   │   ├── anomaly_model.py   # 异常工况模型
│   │   ├── lstm_kan_architecture.py  # LSTM+KAN架构
│   │   └── __init__.py
│   ├── simulation_layer/  # 仿真层模块
│   │   ├── time_series_simulation_engine.py  # 时序仿真引擎
│   │   └── __init__.py
│   └── utils/            # 工具类模块
│       └── data_utils.py # 数据处理工具
├── scripts/              # 脚本文件目录
├── data/                 # 数据目录
├── docs/                 # 文档目录
├── detect/               # 故障检测模型目录
├── requirements.txt      # 依赖包列表
├── README.md             # 项目说明文档
└── project_progress.md   # 项目进度跟踪文档
```

## 下一步计划

### 短期目标 (1-2 周)

1. 完成历史数据模块开发
2. 进一步优化异常工况模型，增强异常检测精度
3. 完善场景管理器的基本功能
4. 增强 Web 界面的交互性和可视化效果

### 中期目标 (3-4 周)

1. 开始控制策略库的开发
2. 实现基本的多目标优化框架
3. 增强仿真引擎的性能和稳定性
4. 开发报告生成功能

## 开发指南

### 添加新功能

1. **添加新的模拟模型**：在 `src/model/pv_model.py` 中扩展 `PVDigitalTwin` 类。
2. **改进故障检测**：在 `src/model/anomaly_model.py` 中修改或扩展异常检测逻辑及效应应用，或在 `src/model/lstm_kan_architecture.py` 中调整模型结构。
3. **添加新的数据源/处理**：在 `src/data_layer/` 中创建或修改相应模块，或在 `src/utils/data_utils.py` 中添加数据加载/生成函数。
4. **扩展仿真功能**：在 `src/simulation_layer/time_series_simulation_engine.py` 中增强事件类型、多时间尺度处理或仿真控制逻辑。
5. **扩展 Django 应用**：在 `pv_digital_twin/dashboard/` 中修改视图或模板，或在 `pv_digital_twin/api/` 中添加新的 API 端点。

### 运行测试

```bash
python scripts/test_system.py
```

## 使用 pvlib

本项目使用[pvlib](https://github.com/pvlib/pvlib-python)库进行光伏系统建模和性能分析。
pvlib Python 是一个广泛使用的光伏系统建模库，提供了一套开放、可靠的光伏系统模型实现。

## 联系与支持

如有问题或建议，请联系项目维护者或提交 Issue。
