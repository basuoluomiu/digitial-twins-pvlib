import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


# 创建模拟类和函数替代原有导入
class Location:
    """模拟位置类"""

    def __init__(
        self, latitude, longitude, altitude=0, tz="Asia/Shanghai", name="PV System"
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.tz = tz
        self.name = name


class PVDigitalTwin:
    """模拟PV数字孪生模型类"""

    def __init__(
        self,
        latitude,
        longitude,
        altitude=0,
        tz="Asia/Shanghai",
        name="PV System",
        module_parameters=None,
        inverter_parameters=None,
        system_loss_dc_ohmic=0.14,
        diode_model="sde",
        temperature_model_parameters=None,
        mppt_algorithm="IDEAL",
        po_step_size=0.01,
        modules_per_string=10,
        strings_per_inverter=2,
        transformer_capacity_kva=50,
        transformer_efficiency=0.985,
    ):
        self.location = Location(latitude, longitude, altitude, tz, name)
        self.module_parameters = module_parameters or {
            "pdc0": 5000,
            "gamma_pdc": -0.004,
        }
        self.inverter_parameters_dict = inverter_parameters or {"pdc0": 5000}
        self.loss_parameters = {
            "system_loss_input_percentage": system_loss_dc_ohmic * 100
        }
        self.diode_model = diode_model
        self.temperature_model_parameters = temperature_model_parameters
        self.modules_per_string = modules_per_string
        self.strings_per_inverter = strings_per_inverter
        self.transformer_capacity_kva = transformer_capacity_kva
        self.transformer_efficiency = transformer_efficiency

        # 模拟自定义逆变器模型
        class CustomInverterModel:
            def __init__(self, mppt_algorithm, po_step_size):
                self.mppt_algorithm = mppt_algorithm
                self.po_step_size = po_step_size

        self.custom_inverter_model = CustomInverterModel(mppt_algorithm, po_step_size)

        # 初始化故障事件管理（模拟版本）
        self.scheduled_fault_events = []  # 存储计划的故障事件
        self.active_faults = []  # 存储当前激活的故障
        self.fault_history = []  # 存储故障历史记录

        # 保存初始参数，用于故障恢复
        self._initial_module_parameters = self.module_parameters.copy()
        self._initial_inverter_parameters = self.inverter_parameters_dict.copy()

    def add_scheduled_fault_event(
        self,
        fault_type,
        fault_day_start_datetime,
        start_hour_offset,
        end_hour_offset,
        severity,
    ):
        """
        添加计划故障事件（模拟版本）
        """
        fault_event = {
            "type": fault_type,
            "day_start": fault_day_start_datetime,
            "start_hour": start_hour_offset,
            "end_hour": end_hour_offset,
            "severity": severity,
            "status": "scheduled",
        }

        self.scheduled_fault_events.append(fault_event)
        print(
            f"[模拟PV] 已添加计划故障事件: {fault_type}, 时间: {start_hour_offset}:00-{end_hour_offset}:00, 严重性: {severity}"
        )

    def clear_all_scheduled_fault_events(self):
        """清除所有计划的故障事件（模拟版本）"""
        # 只清除状态为 'scheduled' 的故障事件，保留 'active' 和 'completed' 的
        scheduled_count = len(
            [f for f in self.scheduled_fault_events if f["status"] == "scheduled"]
        )
        self.scheduled_fault_events = [
            f for f in self.scheduled_fault_events if f["status"] != "scheduled"
        ]
        print(f"[模拟PV] 已清除 {scheduled_count} 个计划故障事件")

    def check_and_apply_faults(self, current_datetime):
        """
        检查并应用当前时间应该激活的故障（模拟版本）
        """
        current_hour = current_datetime.hour
        current_date = current_datetime.date()

        # 检查需要激活的故障
        for fault_event in self.scheduled_fault_events:
            fault_date = fault_event["day_start"].date()

            if fault_event["status"] == "scheduled":
                # 检查是否是故障发生的日期和时间
                if (
                    current_date == fault_date
                    and fault_event["start_hour"]
                    <= current_hour
                    < fault_event["end_hour"]
                ):

                    # 激活故障
                    self._activate_fault(fault_event, current_datetime)
                    fault_event["status"] = "active"

            elif fault_event["status"] == "active":
                # 检查是否应该停用故障
                if (
                    current_date == fault_date
                    and current_hour >= fault_event["end_hour"]
                ):
                    # 故障时间结束，停用故障
                    self._deactivate_fault(fault_event, current_datetime)
                    fault_event["status"] = "completed"

    def _activate_fault(self, fault_event, current_datetime):
        """
        激活故障，应用性能损失（模拟版本）
        """
        fault_type = fault_event["type"]
        severity = fault_event["severity"]

        print(
            f"[模拟PV] 激活故障: {fault_type}, 严重性: {severity}, 时间: {current_datetime}"
        )

        # 根据故障类型应用不同的效果
        if fault_type == "PARTIAL_SHADING":
            # 部分遮挡：降低功率输出
            original_pdc0 = self.module_parameters["pdc0"]
            new_pdc0 = original_pdc0 * (1 - severity)
            self.module_parameters["pdc0"] = new_pdc0
            print(
                f"[模拟PV] 部分遮挡: 模块pdc0从 {original_pdc0:.2f}W 降至 {new_pdc0:.2f}W"
            )

        elif fault_type == "MODULE_DEGRADATION":
            # 组件老化：降低功率输出
            original_pdc0 = self.module_parameters["pdc0"]
            new_pdc0 = original_pdc0 * (1 - severity)
            self.module_parameters["pdc0"] = new_pdc0
            print(
                f"[模拟PV] 组件老化: 模块pdc0从 {original_pdc0:.2f}W 降至 {new_pdc0:.2f}W"
            )

        elif fault_type == "INVERTER_CLIPPING":
            # 逆变器限幅：降低逆变器最大功率
            original_pdc0 = self.inverter_parameters_dict["pdc0"]
            new_pdc0 = original_pdc0 * (1 - severity)
            self.inverter_parameters_dict["pdc0"] = new_pdc0
            print(
                f"[模拟PV] 逆变器限幅: 逆变器pdc0从 {original_pdc0:.2f}W 降至 {new_pdc0:.2f}W"
            )

        else:
            # 默认故障：通用性能损失
            original_pdc0 = self.module_parameters["pdc0"]
            new_pdc0 = original_pdc0 * (1 - severity)
            self.module_parameters["pdc0"] = new_pdc0
            print(
                f"[模拟PV] 通用故障: 模块pdc0从 {original_pdc0:.2f}W 降至 {new_pdc0:.2f}W"
            )

        # 记录激活的故障
        active_fault = {
            "event": fault_event,
            "activated_at": current_datetime,
            "original_parameters": self._initial_module_parameters.copy(),
        }
        self.active_faults.append(active_fault)

    def _deactivate_fault(self, fault_event, current_datetime):
        """
        停用故障，恢复正常参数（模拟版本）
        """
        fault_type = fault_event["type"]
        print(f"[模拟PV] 停用故障: {fault_type}, 时间: {current_datetime}")

        # 从激活故障列表中移除
        self.active_faults = [
            f for f in self.active_faults if f["event"] != fault_event
        ]

        # 记录到历史
        fault_history_entry = {
            "event": fault_event,
            "activated_at": getattr(fault_event, "activated_at", None),
            "deactivated_at": current_datetime,
            "duration_hours": fault_event["end_hour"] - fault_event["start_hour"],
        }
        self.fault_history.append(fault_history_entry)

        # 如果没有其他激活的故障，恢复到初始参数
        if not self.active_faults:
            print("[模拟PV] 恢复到初始系统参数（无激活故障）")
            self.module_parameters = self._initial_module_parameters.copy()
            self.inverter_parameters_dict = self._initial_inverter_parameters.copy()
        else:
            print(f"[模拟PV] 仍有 {len(self.active_faults)} 个故障处于激活状态")

    def get_fault_status(self):
        """
        获取当前故障状态信息（模拟版本）
        """
        return {
            "scheduled_faults": len(self.scheduled_fault_events),
            "active_faults": len(self.active_faults),
            "fault_history": len(self.fault_history),
            "active_fault_details": [
                {
                    "type": f["event"]["type"],
                    "severity": f["event"]["severity"],
                    "activated_at": f["activated_at"],
                }
                for f in self.active_faults
            ],
        }

    def calculate_energy_yield(self, power_values, freq="1h"):
        """计算能量产量"""
        # 简单模拟：功率(kW) * 时间(小时) = 能量(kWh)
        if freq == "1h":
            hours_per_step = 1.0
        elif freq == "1min":
            hours_per_step = 1.0 / 60.0
        elif freq == "1s":
            hours_per_step = 1.0 / 3600.0
        else:
            # 尝试从freq字符串解析，例如 "15min"
            try:
                if "min" in freq:
                    minutes = int(freq.replace("min", ""))
                    hours_per_step = minutes / 60.0
                elif "s" in freq:
                    seconds = int(freq.replace("s", ""))
                    hours_per_step = seconds / 3600.0
                else:
                    hours_per_step = 1  # 默认为1小时
            except ValueError:
                hours_per_step = 1  # 解析失败则默认为1小时

        # 将功率值(W)转换为kW，然后乘以每个时间步的小时数得到kWh
        power_kw = np.array(power_values) / 1000.0
        return np.sum(power_kw) * hours_per_step


class AnomalyModel:
    """模拟异常检测模型类"""

    def __init__(self, model_path=None, pv_digital_twin_model=None):
        self.model_path = model_path
        self.pv_digital_twin_model = pv_digital_twin_model
        self.threshold = 200  # 默认阈值，具体含义视真实模型而定
        self._detected_anomalies_log = []  # 存储检测到的异常
        self.max_log_size = 200  # 增加存储容量，从50条增加到200条
        # 添加用于去重的记录集
        self._anomaly_record_keys = set()

    def set_threshold(self, threshold):
        """设置异常检测阈值"""
        self.threshold = threshold

    def run_detection(self, simulation_result_df):
        """根据仿真结果运行模拟的异常检测逻辑"""
        if self.pv_digital_twin_model is None or simulation_result_df.empty:
            return

        # 模拟检测：例如，如果交流功率远低于基于GHI的预期，则有一定概率产生异常
        pdc0 = self.pv_digital_twin_model.module_parameters.get("pdc0", 5000)

        for index, row in simulation_result_df.iterrows():
            ghi = row.get("ghi", 0)
            ac_power = row.get("ac_power", 0)
            timestamp = row.get("datetime")

            if ghi > 100:  # 只在有一定光照时检测
                # 简单预期功率估算 (非常粗略，未考虑温度、精确效率等)
                expected_power_rough = (ghi / 1000.0) * pdc0 * 0.8  # 假设综合效率80%

                # 如果实际功率远低于预期 (例如低于50%)
                if ac_power < expected_power_rough * 0.5:
                    # 以一定概率生成异常 (例如 30% 的概率)
                    if np.random.rand() < 0.3:
                        anomaly_type = np.random.choice(
                            [
                                "组件阵列输出偏低",
                                "逆变器效率低下",
                                "部分遮挡",
                                "线路故障迹象",
                            ]
                        )
                        severity = np.random.uniform(0.2, 0.8)
                        formatted_timestamp = (
                            timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            if timestamp
                            else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        log_entry = {
                            "timestamp": formatted_timestamp,
                            "type": anomaly_type,
                            "severity": round(severity, 2),
                            "details": f"在GHI {ghi:.0f} W/m²时，AC功率为 {ac_power:.0f}W，低于预期。",
                        }

                        # 创建一个唯一键来识别这条记录，防止重复添加
                        record_key = (
                            f"{formatted_timestamp}_{anomaly_type}_{round(severity, 2)}"
                        )

                        # 如果这条记录不存在，则添加它
                        if record_key not in self._anomaly_record_keys:
                            self._detected_anomalies_log.append(log_entry)
                            self._anomaly_record_keys.add(record_key)
                            # 保持日志大小
                            if len(self._detected_anomalies_log) > self.max_log_size:
                                removed_entry = self._detected_anomalies_log.pop(0)
                                # 同时从去重集合中移除
                                removed_key = f"{removed_entry['timestamp']}_{removed_entry['type']}_{removed_entry['severity']}"
                                if removed_key in self._anomaly_record_keys:
                                    self._anomaly_record_keys.remove(removed_key)

    def get_detected_anomalies_summary(self):
        """获取检测到的异常摘要 (最新的N条)"""
        # 返回日志中的所有记录，或者可以进行切片，例如 self._detected_anomalies_log[-20:] 返回最新的20条
        return sorted(
            self._detected_anomalies_log, key=lambda x: x["timestamp"], reverse=True
        )


class SimulationEngine:
    """模拟仿真引擎类"""

    def __init__(self, pv_digital_twin_model, anomaly_model=None):
        self.pv_model = pv_digital_twin_model
        self.anomaly_model = anomaly_model
        self.weather_data_for_simulation = None
        self.model_states = {}
        self.start_time = 0
        self.end_time = 0
        self.time_step_seconds = 3600

    def configure_simulation(self, start_time, end_time, time_step_seconds):
        """配置仿真参数"""
        self.start_time = start_time
        self.end_time = end_time
        self.time_step_seconds = time_step_seconds

    def start(self):
        """启动仿真"""
        if self.weather_data_for_simulation is None:
            print("错误: 仿真前必须设置天气数据")
            return

        # 模拟仿真过程，生成仿真结果
        for i in range(self.start_time, self.end_time + 1):
            # 从天气数据获取当前时刻的数据
            if i < len(self.weather_data_for_simulation):
                weather_row = self.weather_data_for_simulation.iloc[i]

                # 计算模拟的直流和交流功率
                ghi = weather_row.get("ghi", 0)
                temp_air = weather_row.get("temp_air", 25)

                # 设置最小辐照度阈值，低于此值时不产生电力
                MIN_GHI_THRESHOLD = 20  # W/m² - 低于此值的辐照度不会产生有效发电

                if ghi < MIN_GHI_THRESHOLD:
                    # 低辐照度时设置零功率输出
                    self.model_states[i] = {
                        "dc_power": 0.0,
                        "ac_power": 0.0,
                        "temp_cell": temp_air,  # 低辐照度时组件温度近似等于环境温度
                    }
                else:
                    # 单个模块的直流功率计算
                    single_module_pdc0 = self.pv_model.module_parameters.get(
                        "pdc0", 300.0
                    )  # 获取单模块功率

                    # 简单的模拟计算: 辐照强度(W/m²) * 单模块容量(W) / 1000(W/m²) * 效率因子
                    single_module_dc_power = (
                        ghi
                        * single_module_pdc0
                        / 1000.0
                        * 0.9  # 假设组件在STC下的效率和辐照度转换效率因子
                    )

                    # 温度效应: 温度每升高1°C，功率下降系数百分比
                    temp_coeff = self.pv_model.module_parameters.get(
                        "gamma_pdc", -0.004
                    )
                    temp_cell = temp_air + (ghi / 800) * 25  # 简单估算组件温度
                    single_module_dc_power *= 1 + temp_coeff * (temp_cell - 25)

                    # 计算总模块数量
                    num_modules_total = (
                        self.pv_model.modules_per_string
                        * self.pv_model.strings_per_inverter
                    )

                    # 系统总直流功率
                    dc_power = single_module_dc_power * num_modules_total

                    # 系统损耗 (直流侧)
                    system_loss_percentage = self.pv_model.loss_parameters.get(
                        "system_loss_input_percentage", 14
                    )
                    dc_power_after_losses = dc_power * (
                        1 - system_loss_percentage / 100.0
                    )

                    # 逆变器效率 (简单模拟, 基于逆变器总容量和当前直流输入)
                    # 假设一个简单的与负载率相关的效率模型，或者固定效率
                    inverter_efficiency = 0.96  # 假设固定96%的逆变效率

                    ac_power = dc_power_after_losses * inverter_efficiency

                    # 存储状态
                    self.model_states[i] = {
                        "dc_power": max(0, dc_power),  # 功率不能为负
                        "ac_power": max(0, ac_power),
                        "temp_cell": temp_cell,
                    }


def generate_synthetic_weather_data(
    start_date, end_date, latitude, longitude, freq="1h"
):
    """生成模拟天气数据"""
    # 将日期字符串转换为datetime对象
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    # 生成时间索引
    date_range = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)

    # 创建空的DataFrame
    weather_data = pd.DataFrame(index=date_range)
    weather_data["datetime"] = weather_data.index

    # 生成辐照度数据 (GHI)
    # 模拟日出日落和天气变化
    hours = weather_data.index.hour
    days = (
        (weather_data.index.date - start_datetime.date())
        .astype("timedelta64[D]")
        .astype(int)
    )

    # 基本日辐照模式 (钟形曲线)
    base_ghi = np.zeros(len(hours))
    daytime_mask = (hours >= 6) & (hours <= 18)
    base_ghi[daytime_mask] = 1000 * np.sin(np.pi * (hours[daytime_mask] - 6) / 12)

    # 添加日变化 (天气)
    daily_factor = np.array([0.7 + 0.3 * np.sin(2 * np.pi * d / 7) for d in days])

    # 添加随机噪声 - 只给白天时段添加
    noise = np.zeros(len(hours))
    noise[daytime_mask] = np.random.normal(0, 50, sum(daytime_mask))

    # 组合所有因素
    weather_data["ghi"] = np.maximum(0, base_ghi * daily_factor + noise)

    # 强制确保非白天时段GHI为0（双重保障）
    weather_data.loc[~daytime_mask, "ghi"] = 0.0

    # 生成温度数据
    base_temp = 25 + 5 * np.sin(2 * np.pi * days / 365)  # 年季节变化
    daily_temp_var = 5 * np.sin(2 * np.pi * (hours - 12) / 24)  # 日变化
    temp_noise = np.random.normal(0, 1, len(hours))
    weather_data["temp_air"] = base_temp + daily_temp_var + temp_noise

    # 生成风速数据
    weather_data["wind_speed"] = 2 + 3 * np.random.rand(len(hours))

    # 生成气压数据
    weather_data["pressure"] = (
        101325
        + 1000 * np.sin(2 * np.pi * days / 30)
        + 100 * np.random.randn(len(hours))
    )

    # 计算DNI和DHI (简化)
    weather_data["dni"] = (
        weather_data["ghi"] * 0.8 * (1 + 0.1 * np.random.randn(len(hours)))
    )
    weather_data["dhi"] = weather_data["ghi"] - weather_data["dni"] * np.cos(
        np.radians(90 - latitude)
    )

    return weather_data


class PVModelAdapter:
    """适配器类，用于整合现有的PV数字孪生模型到Django应用中"""

    _instance = None
    _simulation_logs = []  # 用于存储仿真日志的列表
    _is_paused = False  # 仿真是否暂停
    _is_running = False  # 仿真是否正在运行
    _continuous_simulation_active = False  # 是否启用持续仿真
    _scheduled_faults_for_next_day = []  # 新增：存储为下一天计划的故障
    _fault_application_log = []  # 新增：记录已应用的计划故障
    _last_anomaly_check_time = None  # 最后一次异常检测时间
    _anomaly_cache_duration = 10  # 异常检测缓存时间（秒）

    @classmethod
    def get_instance(cls):
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = PVModelAdapter()
        return cls._instance

    def __init__(self):
        """初始化适配器"""
        # 清空日志
        PVModelAdapter._simulation_logs = []
        PVModelAdapter._scheduled_faults_for_next_day = []  # 初始化新增属性
        PVModelAdapter._fault_application_log = []  # 初始化新增属性
        self._log_message("初始化PV数字孪生模型适配器")

        # 定义默认的初始系统配置
        initial_total_capacity_kw = 5.0  # 初始总系统容量 (kWp)
        initial_single_module_pdc0_w = 300.0  # 标准单模块功率 (W)
        initial_modules_per_string = 10  # 每串模块数

        initial_total_pdc0_w = initial_total_capacity_kw * 1000.0

        if initial_single_module_pdc0_w <= 0:  # 防止除零错误
            initial_single_module_pdc0_w = 300.0  # 默认回退
        if initial_modules_per_string <= 0:
            initial_modules_per_string = 10  # 默认回退

        initial_num_strings = int(
            np.ceil(
                initial_total_pdc0_w
                / (initial_single_module_pdc0_w * initial_modules_per_string)
            )
        )
        if initial_num_strings == 0 and initial_total_pdc0_w > 0:
            initial_num_strings = 1

        # 计算实际的逆变器直流输入功率，基于确定的模块和组串数量
        actual_initial_inverter_pdc0_w = (
            initial_num_strings
            * initial_modules_per_string
            * initial_single_module_pdc0_w
        )

        # 初始化PV数字孪生模型
        self.pv_twin = PVDigitalTwin(
            latitude=39.9,  # 默认位置：北京
            longitude=116.4,
            altitude=50,
            name="PV Digital Twin System",
            module_parameters={  # 单个模块的参数
                "pdc0": initial_single_module_pdc0_w,
                "gamma_pdc": -0.004,  # 温度系数 -0.4%/°C
            },
            inverter_parameters={  # 整个逆变器的参数，pdc0应为系统总直流功率
                "pdc0": actual_initial_inverter_pdc0_w
            },
            modules_per_string=initial_modules_per_string,
            strings_per_inverter=initial_num_strings,
        )
        self._log_message(f"创建PV数字孪生模型: {self.pv_twin.location.name}")
        self._log_message(
            f"位置: 纬度={self.pv_twin.location.latitude}, 经度={self.pv_twin.location.longitude}"
        )
        # Log the actual total capacity configured for the inverter
        self._log_message(
            f"初始系统容量 (逆变器Pdc0): {actual_initial_inverter_pdc0_w / 1000.0:.2f} kWp"
        )
        self._log_message(
            f"  (单模块Pdc0: {initial_single_module_pdc0_w}W, 每串模块数: {initial_modules_per_string}, 组串数: {initial_num_strings})"
        )

        # 初始化异常检测模型
        self._initialize_anomaly_model()

        # 生成天气数据
        self._generate_weather_data()
        self._log_message(
            f"生成初始天气数据: {len(self.weather_data)}条记录 (每小时间隔)"
        )

        # 运行仿真
        self._run_simulation()

        # 处理仿真结果
        self.simulation_results = self._process_simulation_results()
        self._log_message("仿真完成，结果处理完毕")

        # 启动持续仿真
        import threading

        self._continuous_simulation_thread = threading.Thread(
            target=self._continuous_simulation, daemon=True
        )
        self._continuous_simulation_active = True
        self._is_running = True
        self._is_paused = False
        self._continuous_simulation_thread.start()
        self._log_message("持续仿真已启动 (每日更新，每小时间隔数据)")

    def _initialize_anomaly_model(self):
        """初始化异常检测模型"""
        self._log_message("初始化异常检测模型")
        self.anomaly_model = AnomalyModel(pv_digital_twin_model=self.pv_twin)
        self.anomaly_model.set_threshold(200)  # 设置默认异常阈值
        self._log_message(f"异常检测阈值设置为: {self.anomaly_model.threshold}")

    def _generate_weather_data(self):
        """生成初始天气数据，例如过去48小时的每小时数据"""
        self._log_message("开始生成初始天气数据...")
        # 生成过去48小时的每小时数据
        current_time = datetime.now()
        end_date = current_time.strftime("%Y-%m-%d %H:%M:%S")
        start_date = (current_time - timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S")

        self.weather_data = generate_synthetic_weather_data(
            start_date,
            end_date,
            self.pv_twin.location.latitude,
            self.pv_twin.location.longitude,
            freq="1h",  # 生成每小时间隔数据
        )
        self._log_message(
            f"初始天气数据生成完成: 从{start_date}到{end_date} ({len(self.weather_data)} 条记录)"
        )
        self._log_message(f"平均辐照度: {self.weather_data['ghi'].mean():.2f} W/m²")
        self._log_message(f"平均气温: {self.weather_data['temp_air'].mean():.2f} °C")

    def _run_simulation(self):
        """运行初始仿真 (基于每小时数据)"""
        self._log_message("开始运行初始仿真 (每小时间隔数据)...")
        # 创建仿真引擎
        self.simulation_engine = SimulationEngine(self.pv_twin, self.anomaly_model)

        # 设置天气数据
        self.simulation_engine.weather_data_for_simulation = self.weather_data

        # 配置仿真参数 - 1小时步长
        self.simulation_engine.configure_simulation(
            0, len(self.weather_data) - 1, 3600
        )  # 3600秒

        # 启动仿真
        self.simulation_engine.start()
        self._log_message(
            f"初始仿真完成，共计算{len(self.simulation_engine.model_states)}个时间点"
        )

    def _continuous_simulation(self):
        """持续仿真函数，在单独的线程中运行，每次迭代生成并仿真一天（24小时，每小时间隔）的数据"""
        import time
        from datetime import datetime, timedelta

        self._log_message("持续仿真线程已启动 (每日数据生成，每小时间隔)")

        while self._continuous_simulation_active:
            if self._is_paused:
                time.sleep(1)
                continue
            try:
                # 确定下一天数据的起止时间
                if (
                    hasattr(self, "weather_data")
                    and not self.weather_data.empty
                    and "datetime" in self.weather_data.columns
                ):
                    last_weather_time = pd.to_datetime(
                        self.weather_data["datetime"].iloc[-1]
                    )
                    start_datetime_for_new_day = last_weather_time + timedelta(hours=1)
                else:
                    start_datetime_for_new_day = datetime.now().replace(
                        minute=0, second=0, microsecond=0
                    )  # 从当前小时开始

                end_datetime_for_new_day = start_datetime_for_new_day + timedelta(
                    hours=23
                )  # 生成接下来24小时的数据 (0-23小时)

                start_date_str = start_datetime_for_new_day.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                end_date_str = end_datetime_for_new_day.strftime("%Y-%m-%d %H:%M:%S")

                new_daily_weather_data = generate_synthetic_weather_data(
                    start_date_str,
                    end_date_str,
                    self.pv_twin.location.latitude,
                    self.pv_twin.location.longitude,
                    freq="1h",  # 生成每小时间隔数据
                )

                if new_daily_weather_data.empty:
                    self._log_message("持续仿真: 未能生成新的每日天气数据，稍后重试。")
                    time.sleep(60)  # 如果生成失败，等待较长时间
                    continue

                # 在运行仿真之前，应用计划的故障
                self._apply_scheduled_faults_to_simulation(
                    start_datetime_for_new_day, new_daily_weather_data.copy()
                )

                # 添加到现有天气数据 (DataFrame)
                if hasattr(self, "weather_data") and not self.weather_data.empty:
                    self.weather_data = (
                        pd.concat([self.weather_data, new_daily_weather_data])
                        .drop_duplicates(subset=["datetime"])
                        .sort_values(by="datetime")
                        .reset_index(drop=True)
                    )
                else:
                    self.weather_data = new_daily_weather_data.reset_index(drop=True)

                # 保留例如最近 7.5 天 (180小时) 的天气数据
                max_weather_points = 180
                if len(self.weather_data) > max_weather_points:
                    self.weather_data = self.weather_data.iloc[
                        -max_weather_points:
                    ].reset_index(drop=True)

                # 创建或获取仿真引擎
                if not hasattr(self, "simulation_engine") or not self.simulation_engine:
                    self.simulation_engine = SimulationEngine(
                        self.pv_twin, self.anomaly_model
                    )

                # 设置天气数据为新生成的这一天的数据
                self.simulation_engine.weather_data_for_simulation = (
                    new_daily_weather_data.reset_index(drop=True)
                )

                # 配置仿真参数（仅仿真新添加的这24小时数据）
                self.simulation_engine.configure_simulation(
                    0, len(new_daily_weather_data) - 1, 3600
                )  # 3600秒步长

                # 启动仿真
                self.simulation_engine.start()

                new_day_sim_results_df = self._process_simulation_results()

                if not new_day_sim_results_df.empty:
                    if (
                        hasattr(self, "simulation_results")
                        and isinstance(self.simulation_results, pd.DataFrame)
                        and not self.simulation_results.empty
                    ):
                        self.simulation_results = (
                            pd.concat([self.simulation_results, new_day_sim_results_df])
                            .drop_duplicates(subset=["datetime"])
                            .sort_values(by="datetime")
                            .reset_index(drop=True)
                        )
                    else:
                        self.simulation_results = new_day_sim_results_df.reset_index(
                            drop=True
                        )

                    # 保留例如最近 7.5 天 (180小时) 的仿真结果
                    max_sim_points = 180
                    if len(self.simulation_results) > max_sim_points:
                        self.simulation_results = self.simulation_results.iloc[
                            -max_sim_points:
                        ].reset_index(drop=True)

                    self._log_message(
                        f"持续仿真: 添加了 {len(new_day_sim_results_df)} 个新数据点 (每日更新，每小时间隔)。总仿真点: {len(self.simulation_results)}"
                    )

                    # 在新的仿真数据产生后运行异常检测
                    if self.anomaly_model and hasattr(
                        self.anomaly_model, "run_detection"
                    ):
                        # 首先对新生成的数据进行异常检测
                        self.anomaly_model.run_detection(new_day_sim_results_df)
                        self._log_message(
                            f"对新的 {len(new_day_sim_results_df)} 个数据点运行了异常检测。"
                        )

                        # 同时对最近7天的历史数据也进行异常检测，确保故障检测图表能获取完整数据
                        if (
                            hasattr(self, "simulation_results")
                            and isinstance(self.simulation_results, pd.DataFrame)
                            and not self.simulation_results.empty
                        ):
                            # 获取最近7天（或现有天数）的数据
                            recent_days = 7
                            unique_dates = self.simulation_results["date"].unique()
                            if len(unique_dates) > 0:
                                # 如果历史数据少于7天，则使用全部历史数据
                                days_to_check = min(recent_days, len(unique_dates))
                                recent_dates = sorted(unique_dates)[-days_to_check:]
                                recent_data = self.simulation_results[
                                    self.simulation_results["date"].isin(recent_dates)
                                ]

                                # 对这部分历史数据再次运行异常检测
                                self.anomaly_model.run_detection(recent_data)
                                self._log_message(
                                    f"同时对最近 {days_to_check} 天的历史数据（{len(recent_data)}个点）重新运行了异常检测，确保图表更新。"
                                )
                else:
                    self._log_message("持续仿真: 未生成新的一天的仿真结果。")

                # 每隔一段时间 (例如5秒) 添加新的一天的数据
                # 实际应用中可能希望这个间隔更长，比如每小时或每天才真正添加新的一天数据
                time.sleep(5)
            except Exception as e_continuous_sim:
                self._log_message(
                    f"错误: _continuous_simulation 循环中捕获到异常: {e_continuous_sim}"
                )
                import traceback

                detailed_error_info = traceback.format_exc()
                self._log_message(detailed_error_info)
                self._log_message("持续仿真线程因错误暂停60秒...")
                time.sleep(60)  # 遇到错误时等待60秒
                # 如果需要，可以在这里添加逻辑来决定是否应该停止持续仿真，例如:
                # if isinstance(e_continuous_sim, SystemExit):
                #     self._continuous_simulation_active = False
                #     break # 退出循环

    def _log_message(self, message):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "message": message}
        PVModelAdapter._simulation_logs.append(log_entry)
        print(f"[{timestamp}] {message}")  # 同时输出到控制台

    def get_simulation_logs(self):
        """获取仿真日志"""
        return PVModelAdapter._simulation_logs

    def _process_simulation_results(self):
        """处理当前仿真引擎运行的仿真结果"""
        # 获取当前仿真引擎中模型状态的时间戳/索引
        # 这些键对应于 self.simulation_engine.weather_data_for_simulation 的索引
        sim_engine_model_states = self.simulation_engine.model_states
        if not sim_engine_model_states:
            print("警告: 本次仿真步骤未产生任何模型状态。")
            return pd.DataFrame()  # 返回空的DataFrame

        sim_timestamps_indices = sorted(sim_engine_model_states.keys())

        # 从模型状态创建结果列表
        results_list = [sim_engine_model_states[t] for t in sim_timestamps_indices]
        simulation_results_df_raw = pd.DataFrame(results_list)

        # 从当前仿真引擎使用的天气数据 (weather_data_for_simulation) 获取相关部分
        current_sim_weather_data = self.simulation_engine.weather_data_for_simulation
        if current_sim_weather_data is None or current_sim_weather_data.empty:
            print("警告: _process_simulation_results 中仿真引擎的天气数据为空。")
            return pd.DataFrame()

        # 确保索引在范围内
        valid_indices = [
            idx for idx in sim_timestamps_indices if idx < len(current_sim_weather_data)
        ]
        if not valid_indices:
            print("警告: _process_simulation_results 中没有有效的仿真时间戳索引。")
            return pd.DataFrame()

        relevant_weather_df = current_sim_weather_data.iloc[valid_indices].reset_index(
            drop=True
        )
        # 确保 simulation_results_df_raw 也只包含有效索引对应的结果
        simulation_results_df_raw = simulation_results_df_raw.iloc[
            [sim_timestamps_indices.index(i) for i in valid_indices]
        ].reset_index(drop=True)

        # 重置仿真结果的索引 (这一步可能不再需要，因为上面已经 reset_index)
        # simulation_results_df_processed = simulation_results_df_raw.reset_index(drop=True)
        simulation_results_df_processed = (
            simulation_results_df_raw  # 使用已重置索引的raw
        )

        # 移除任何在天气数据中已存在的列，以避免合并时的列名冲突
        # （但通常我们希望保留天气数据的 datetime, ghi, temp_air 等）
        # 这里的逻辑是，如果 model_states 返回了与 weather_data 中同名的列（除了索引相关的），则移除它们以避免重复。
        # 实际上，我们更可能希望 model_states 只包含模型计算出的新列 (dc_power, ac_power, temp_cell)。
        # 假设 model_states 的列名与 relevant_weather_df 没有冲突 (除了索引)

        # 合并天气数据和仿真结果
        # 确保 relevant_weather_df 和 simulation_results_df_processed 的行数一致
        if len(relevant_weather_df) != len(simulation_results_df_processed):
            print(
                f"警告: _process_simulation_results 中天气数据和仿真结果行数不匹配。天气: {len(relevant_weather_df)}, 结果: {len(simulation_results_df_processed)}"
            )
            # 尝试取两者中较小的长度进行合并，但这可能导致数据不一致
            min_len = min(
                len(relevant_weather_df), len(simulation_results_df_processed)
            )
            relevant_weather_df = relevant_weather_df.iloc[:min_len]
            simulation_results_df_processed = simulation_results_df_processed.iloc[
                :min_len
            ]
            if min_len == 0:
                return pd.DataFrame()

        df = pd.concat([relevant_weather_df, simulation_results_df_processed], axis=1)

        # 确保必要的列存在
        required_cols = [
            "datetime",
            "ghi",
            "temp_air",
            "dc_power",
            "ac_power",
            "temp_cell",
        ]
        for col in required_cols:
            if col not in df.columns:
                if col == "datetime" and isinstance(df.index, pd.DatetimeIndex):
                    df[col] = df.index.to_series(name="datetime")
                elif col in ["dc_power", "ac_power"]:
                    df[col] = 0.0
                elif col == "temp_cell":
                    temp_air_default = 25.0
                    if "temp_air" in df.columns:
                        temp_air_values = df["temp_air"].to_numpy()
                        if (
                            len(temp_air_values) > 0
                            and not np.isnan(temp_air_values).all()
                        ):
                            temp_air_default = np.nanmean(temp_air_values)
                    df[col] = temp_air_default
                elif col in ["ghi", "temp_air"]:
                    df[col] = np.nan
                else:
                    df[col] = np.nan

        # 确保datetime列存在且格式正确
        if "datetime" not in df.columns:
            print("警告: 'datetime'列在合并后缺失。将尝试重新创建。")
            if len(sim_timestamps_indices) > 0 and len(self.weather_data) > 0:
                datetime_series_from_weather = (
                    self.weather_data.iloc[sim_timestamps_indices]
                    .index.to_series()
                    .reset_index(drop=True)
                )
                df["datetime"] = pd.to_datetime(datetime_series_from_weather)
            else:
                df["datetime"] = pd.to_datetime([datetime.now()])
        elif not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])

        # 添加日期列用于按日分组
        df["date"] = df["datetime"].dt.date

        return df

    def get_daily_energy(self):
        """计算每日能量产量"""
        if (
            "date" in self.simulation_results.columns
            and "ac_power" in self.simulation_results.columns
            and len(self.simulation_results) > 0
        ):
            try:
                # 检查是否有缓存的每日能量数据
                if not hasattr(self, "_daily_energy_cache"):
                    self._daily_energy_cache = {}

                # 计算需要处理的日期列表
                all_dates = set(
                    self.simulation_results["date"]
                    .apply(lambda x: x.strftime("%Y-%m-%d"))
                    .unique()
                )
                cached_dates = set(self._daily_energy_cache.keys())

                # 找出新增的日期（需要计算）
                new_dates = all_dates - cached_dates

                # 为新的日期计算能量
                if new_dates:
                    for date_str in new_dates:
                        date_obj = pd.to_datetime(date_str).date()
                        date_mask = self.simulation_results["date"].apply(
                            lambda x: pd.to_datetime(x).date() == date_obj
                        )
                        date_group = self.simulation_results[date_mask]
                        if not date_group.empty:
                            energy_kwh = self.pv_twin.calculate_energy_yield(
                                date_group["ac_power"].values,
                                freq="1h",  # 使用1h频率计算
                            )
                            # 存入缓存
                            self._daily_energy_cache[date_str] = energy_kwh

                # 转换为JSON友好格式
                return [
                    {
                        "date": date_str,
                        "energy_kwh": energy_kwh,
                    }
                    for date_str, energy_kwh in sorted(self._daily_energy_cache.items())
                ]
            except Exception as e:
                print(f"计算每日能量产量时出错: {e}")
                return []
        return []

    def get_simulation_data(self):
        """获取仿真数据用于图表"""
        if not hasattr(self, "simulation_results") or self.simulation_results.empty:
            return {}

        # 获取最近48小时的数据 (如果每小时一个点)
        recent_points = 48
        recent_data = (
            self.simulation_results.iloc[-recent_points:].copy()
            if len(self.simulation_results) > recent_points
            else self.simulation_results.copy()
        )

        # 计算系统效率
        # 效率 = (AC功率 / (GHI * 有效面积)) * 100
        # 由于没有直接的面积，我们用 pdc0 (系统在1000W/m^2下的直流功率) 来估算
        # 有效输入功率约等于 (GHI / 1000) * pdc0 (这是理论最大DC功率)
        # 然后 AC功率 / 理论最大DC功率 可近似为一种系统综合效率（包括逆变器效率和各种损失）

        # 获取 pdc0，确保是数值
        # pdc0_watts = self.pv_twin.module_parameters.get("pdc0", 5000)  # 这是单个模块的PDC0，错误！

        # 正确获取系统总PDC0 (通常是逆变器的额定输入直流功率)
        total_pdc0_watts_inverter = getattr(
            self.pv_twin, "inverter_parameters_dict", {}
        ).get("pdc0")
        if total_pdc0_watts_inverter is None or total_pdc0_watts_inverter <= 0:
            # 如果逆变器参数中没有或无效，则从模块参数计算总容量
            single_module_pdc0 = getattr(self.pv_twin, "module_parameters", {}).get(
                "pdc0", 300.0
            )
            num_modules_total = getattr(
                self.pv_twin, "modules_per_string", 10
            ) * getattr(self.pv_twin, "strings_per_inverter", 1)
            total_pdc0_watts_inverter = single_module_pdc0 * num_modules_total

        if (
            not isinstance(total_pdc0_watts_inverter, (int, float))
            or total_pdc0_watts_inverter <= 0
        ):
            total_pdc0_watts_inverter = 5000  # 最后的默认回退值

        efficiency_list = []
        for index, row in recent_data.iterrows():
            ac_power = row.get("ac_power", 0)
            ghi = row.get("ghi", 0)

            # 夜间或低辐照条件 (GHI < 100) 或无发电时，设置效率为0
            if ghi < 100 or ac_power <= 0 or total_pdc0_watts_inverter <= 0:
                efficiency_list.append(0)  # 夜间或GHI过低时效率为0
                continue

            # 计算系统效率
            theoretical_total_dc_input_power_at_current_ghi = (
                ghi / 1000.0
            ) * total_pdc0_watts_inverter

            if theoretical_total_dc_input_power_at_current_ghi > 0:
                # 效率为实际AC输出与该理论DC输入之比
                eff = (ac_power / theoretical_total_dc_input_power_at_current_ghi) * 100
                # 限制在合理范围内(0-100%)
                efficiency_list.append(min(max(eff, 0), 100))
            else:
                efficiency_list.append(0)  # 理论输入功率为0时，效率也为0

        # 转换为JSON格式
        result = {
            "timestamps": recent_data["datetime"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .tolist(),
            "ac_power": recent_data["ac_power"].tolist(),
            "dc_power": recent_data["dc_power"].tolist(),
            "temp_air": recent_data["temp_air"].tolist(),
            "temp_cell": recent_data["temp_cell"].tolist(),
            "ghi": recent_data["ghi"].tolist(),
            "efficiency": efficiency_list,  # 添加效率数据
        }

        return result

    def get_system_info(self):
        """获取系统信息"""
        pv_twin_instance = self.pv_twin

        # 1. 获取总装机容量 (kWp)
        # 直接使用逆变器pdc0参数作为系统总装机容量
        total_system_capacity_watts = getattr(
            pv_twin_instance, "inverter_parameters_dict", {}
        ).get("pdc0")

        # 如果无法从逆变器参数获取，则从模块配置计算
        if total_system_capacity_watts is None or total_system_capacity_watts <= 0:
            single_module_pdc0 = getattr(pv_twin_instance, "module_parameters", {}).get(
                "pdc0", 300.0
            )
            modules_per_string = getattr(pv_twin_instance, "modules_per_string", 10)
            strings_per_inverter = getattr(pv_twin_instance, "strings_per_inverter", 1)
            total_system_capacity_watts = (
                single_module_pdc0 * modules_per_string * strings_per_inverter
            )
            self._log_message(
                f"警告：从模块配置计算系统容量: {total_system_capacity_watts/1000.0:.2f}kWp (模块:{single_module_pdc0}W, 每串:{modules_per_string}, 组串:{strings_per_inverter})"
            )
        else:
            # 记录日志用于调试
            self._log_message(
                f"信息：使用逆变器pdc0参数作为系统容量: {total_system_capacity_watts/1000.0:.2f}kWp"
            )

        installed_capacity_kw = (
            total_system_capacity_watts / 1000.0
            if total_system_capacity_watts is not None
            and total_system_capacity_watts > 0
            else 0.0
        )

        current_power_value = 0
        max_power_today = 0  # 添加当天最高功率变量
        max_ghi_today = 0  # 添加当天最高辐照度变量
        max_efficiency_today = 0  # 添加当天最高系统效率变量
        today_energy = 0
        latest_temp_air = getattr(
            pv_twin_instance.location, "default_temp_air", 25
        )  # 默认值
        latest_temp_cell = getattr(
            pv_twin_instance.location, "default_temp_cell", 25
        )  # 默认值

        if hasattr(self, "simulation_results") and not self.simulation_results.empty:
            # 获取最新数据点
            latest_data = self.simulation_results.iloc[-1]
            current_power_value = latest_data.get("ac_power", 0)
            latest_temp_air = latest_data.get(
                "temp_air", latest_temp_air
            )  # 如果仿真结果中有，则覆盖默认值
            latest_temp_cell = latest_data.get("temp_cell", latest_temp_cell)

            # 计算今日能量和今日最高功率
            if (
                "date" in self.simulation_results.columns
                and "ac_power" in self.simulation_results.columns
                and "ghi" in self.simulation_results.columns
            ):
                # 确保 'date' 列是 datetime.date 对象或可以安全转换为此类进行比较
                try:
                    # 获取当前日期
                    current_system_date = pd.to_datetime(datetime.now()).date()

                    # 获取当天的数据
                    today_data_mask = self.simulation_results["date"].apply(
                        lambda x: pd.to_datetime(x).date() == current_system_date
                    )
                    today_data = self.simulation_results[today_data_mask]

                    if not today_data.empty:
                        today_energy = self.pv_twin.calculate_energy_yield(
                            today_data["ac_power"].values,
                            freq="1h",  # 假设数据是每小时间隔
                        )
                        # 获取当天的最高功率值
                        max_power_today = today_data["ac_power"].max()

                        # 获取当天的最高辐照度值
                        max_ghi_today = today_data["ghi"].max()

                        # 计算当天最高系统效率
                        # 为了避免除零错误，确保 ghi 和 total_system_capacity_watts 有效
                        # 首先创建包含有效辐照度(>100)和功率的数据子集
                        valid_efficiency_data = today_data[
                            (today_data["ghi"] > 100) & (today_data["ac_power"] > 0)
                        ].copy()

                        if (
                            not valid_efficiency_data.empty
                            and total_system_capacity_watts > 0
                        ):
                            # 计算每个有效时间点的效率，类似 get_simulation_data 方法中的逻辑
                            valid_efficiency_data["efficiency"] = (
                                valid_efficiency_data["ac_power"]
                                / (
                                    (valid_efficiency_data["ghi"] / 1000.0)
                                    * total_system_capacity_watts
                                )
                            ) * 100

                            # 限制效率在合理范围内(0-100%)
                            valid_efficiency_data["efficiency"] = valid_efficiency_data[
                                "efficiency"
                            ].apply(lambda x: min(max(x, 0), 100))

                            # 获取最高效率
                            max_efficiency_today = valid_efficiency_data[
                                "efficiency"
                            ].max()
                except Exception as e:
                    self._log_message(
                        f"计算今日能量或最高指标时出错 (get_system_info): {e}"
                    )
        else:
            # 如果没有仿真结果，使用默认值 (已在上面设置)
            pass

        return {
            "installed_capacity": installed_capacity_kw,
            "current_power": current_power_value,
            "max_power_today": max_power_today,
            "max_ghi_today": max_ghi_today,  # 添加当天最高辐照度
            "max_efficiency_today": max_efficiency_today,  # 添加当天最高系统效率
            "daily_energy": today_energy,
            "current_temp_air": latest_temp_air,
            "current_temp_cell": latest_temp_cell,
        }

    def get_detected_anomalies(self):
        """获取检测到的异常，带缓存机制优化性能"""
        if self.anomaly_model is None:
            self._log_message("警告: 尝试获取异常检测结果，但异常检测模型未初始化")
            return []

        try:
            from datetime import datetime, timedelta

            current_time = datetime.now()

            # 检查是否需要重新运行异常检测（基于缓存时间）
            should_run_detection = True
            if PVModelAdapter._last_anomaly_check_time is not None:
                time_since_last_check = (
                    current_time - PVModelAdapter._last_anomaly_check_time
                )
                if (
                    time_since_last_check.total_seconds()
                    < PVModelAdapter._anomaly_cache_duration
                ):
                    should_run_detection = False

            # 如果需要重新检测且有仿真数据
            if should_run_detection and (
                hasattr(self, "simulation_results")
                and not self.simulation_results.empty
            ):
                # 获取最近的数据点（例如过去24小时）进行检测
                recent_data = (
                    self.simulation_results.iloc[-24:]
                    if len(self.simulation_results) > 24
                    else self.simulation_results
                )
                if not recent_data.empty:
                    self.anomaly_model.run_detection(recent_data)
                    PVModelAdapter._last_anomaly_check_time = current_time
                    self._log_message(
                        f"对最近 {len(recent_data)} 个数据点运行了异常检测（缓存{PVModelAdapter._anomaly_cache_duration}秒）"
                    )
                else:
                    self._log_message("没有可用的仿真数据进行异常检测")
            elif not should_run_detection:
                self._log_message("使用缓存的异常检测结果")

            # 获取检测结果
            anomalies = self.anomaly_model.get_detected_anomalies_summary()
            return anomalies
        except Exception as e:
            import traceback

            self._log_message(f"获取异常检测摘要时出错: {e}")
            self._log_message(traceback.format_exc())
            return []

    def apply_settings(self, settings):
        """应用新的系统设置"""
        try:
            self._log_message(f"开始应用新设置: {settings}")
            # 解析设置
            current_pv_twin = self.pv_twin
            latitude = float(
                settings.get("latitude", current_pv_twin.location.latitude)
            )
            longitude = float(
                settings.get("longitude", current_pv_twin.location.longitude)
            )

            # 保存每日能量缓存，以便在重置后恢复
            daily_energy_cache = (
                getattr(self, "_daily_energy_cache", {})
                if hasattr(self, "_daily_energy_cache")
                else {}
            )

            # 有两种方式设置系统容量:
            # 1. capacity_strings: 直接指定组串数量（优先）
            # 2. system_capacity: 指定系统容量(kWp)，然后计算组串数

            # 检查是否提供了capacity_strings参数
            capacity_strings = settings.get("capacity_strings")
            if capacity_strings is not None:
                # 直接使用指定的组串数
                try:
                    num_strings = int(capacity_strings)
                    self._log_message(f"使用用户直接指定的组串数: {num_strings}")
                except (ValueError, TypeError):
                    self._log_message(
                        f"无效的组串数值 '{capacity_strings}'，将使用系统容量计算"
                    )
                    capacity_strings = None

            # 如果没有提供capacity_strings或转换失败，则使用system_capacity
            if capacity_strings is None:
                # 总系统容量 (kWp)
                capacity_kw = float(
                    settings.get(
                        "system_capacity",
                        # 尝试从当前逆变器Pdc0获取总容量 (W)，然后转kW
                        getattr(current_pv_twin, "inverter_parameters_dict", {}).get(
                            "pdc0", 5000
                        )
                        / 1000.0,
                    )
                )

                # 验证参数
                if capacity_kw <= 0:
                    self._log_message(f"设置应用失败: 系统容量 {capacity_kw}kWp 无效。")
                    return False, "系统容量必须大于0"

                EFFECTIVE_SINGLE_MODULE_PDC0_W = 300.0  # 使用一个标准值，如300Wp
                target_total_system_pdc0_w = capacity_kw * 1000.0

                # 保留用户可能已设置的 modules_per_string，如果它是合理的
                modules_per_string_val = getattr(
                    current_pv_twin, "modules_per_string", 10
                )
                if (
                    not isinstance(modules_per_string_val, int)
                    or modules_per_string_val <= 0
                ):
                    modules_per_string_val = 10  # 默认回退

                if EFFECTIVE_SINGLE_MODULE_PDC0_W <= 0:
                    self._log_message(
                        f"设置应用失败: 单模块功率 {EFFECTIVE_SINGLE_MODULE_PDC0_W}W 无效。"
                    )
                    return False, "内部配置错误：单模块功率无效"

                single_string_pdc0_w = (
                    EFFECTIVE_SINGLE_MODULE_PDC0_W * modules_per_string_val
                )

                if single_string_pdc0_w <= 0:
                    num_strings = 0
                else:
                    ideal_num_strings_float = (
                        target_total_system_pdc0_w / single_string_pdc0_w
                    )
                    num_strings = int(round(ideal_num_strings_float))

                # 如果目标容量大于0，但计算出的组串数为0 (因为 round 到了0，例如目标0.1kW, 每串3kW), 则至少分配1个组串
                if (
                    target_total_system_pdc0_w > 0
                    and num_strings == 0
                    and ideal_num_strings_float > 0
                ):
                    num_strings = 1
                # 如果目标容量为0或负数，则组串数也应为0
                if target_total_system_pdc0_w <= 0:
                    num_strings = 0

            # 确保num_strings为一个合理的整数值
            if num_strings < 0:
                num_strings = 0
            elif num_strings > 100:  # 设置一个上限，防止极端值
                num_strings = 100

            # 基于确定的组串数计算实际系统容量
            modules_per_string_val = getattr(current_pv_twin, "modules_per_string", 10)
            if (
                not isinstance(modules_per_string_val, int)
                or modules_per_string_val <= 0
            ):
                modules_per_string_val = 10

            EFFECTIVE_SINGLE_MODULE_PDC0_W = 300.0
            actual_configured_pdc0_w = (
                num_strings * modules_per_string_val * EFFECTIVE_SINGLE_MODULE_PDC0_W
            )

            self._log_message(
                f"最终确定组串数: {num_strings} (每串 {modules_per_string_val} 个模块, 单模块 {EFFECTIVE_SINGLE_MODULE_PDC0_W}W)"
            )
            self._log_message(
                f"实际配置容量: {actual_configured_pdc0_w / 1000.0:.2f} kWp ({actual_configured_pdc0_w:.0f}W)"
            )

            # 读取其他设置参数
            temp_coeff_percent = float(
                settings.get(
                    "temp_coeff",
                    # 从当前模块参数获取gamma_pdc，它已经是小数形式，乘以100得到百分比
                    getattr(current_pv_twin, "module_parameters", {}).get(
                        "gamma_pdc", -0.004
                    )
                    * 100,
                )
            )
            system_loss_percent = float(
                settings.get(
                    "system_loss",
                    getattr(current_pv_twin, "loss_parameters", {}).get(
                        "system_loss_input_percentage", 14
                    ),
                )
            )
            fault_threshold = float(
                settings.get(
                    "fault_threshold",
                    (
                        200
                        if self.anomaly_model is None
                        else getattr(self.anomaly_model, "threshold", 200)
                    ),
                )
            )

            # 模块参数 (单个模块)
            current_module_params = getattr(
                current_pv_twin, "module_parameters", {}
            ).copy()
            current_module_params["pdc0"] = EFFECTIVE_SINGLE_MODULE_PDC0_W
            current_module_params["gamma_pdc"] = temp_coeff_percent / 100.0
            final_module_params = current_module_params

            # 逆变器参数 (系统总直流功率)
            current_inverter_params = getattr(
                current_pv_twin, "inverter_parameters_dict", {}
            ).copy()
            current_inverter_params["pdc0"] = actual_configured_pdc0_w
            final_inverter_params = current_inverter_params

            self._log_message(f"传递给 PVDigitalTwin 的模块参数: {final_module_params}")
            self._log_message(
                f"传递给 PVDigitalTwin 的逆变器参数: {final_inverter_params}"
            )
            self._log_message(
                f"传递给 PVDigitalTwin 的 modules_per_string: {modules_per_string_val}, strings_per_inverter: {num_strings}"
            )

            try:
                new_pv_twin = PVDigitalTwin(
                    latitude=latitude,
                    longitude=longitude,
                    altitude=getattr(current_pv_twin.location, "altitude", 50),
                    tz=getattr(current_pv_twin.location, "tz", "Asia/Shanghai"),
                    name=getattr(
                        current_pv_twin.location, "name", "PV Digital Twin System"
                    ),
                    module_parameters=final_module_params,
                    inverter_parameters=final_inverter_params,
                    system_loss_dc_ohmic=system_loss_percent / 100.0,
                    diode_model=getattr(current_pv_twin, "diode_model", "sde"),
                    temperature_model_parameters=getattr(
                        current_pv_twin, "temperature_model_parameters", None
                    ),
                    mppt_algorithm=getattr(
                        current_pv_twin.custom_inverter_model, "mppt_algorithm", "IDEAL"
                    ),
                    po_step_size=getattr(
                        current_pv_twin.custom_inverter_model, "po_step_size", 0.01
                    ),
                    modules_per_string=modules_per_string_val,
                    strings_per_inverter=num_strings,
                    transformer_capacity_kva=getattr(
                        current_pv_twin, "transformer_capacity_kva", 50
                    ),
                    transformer_efficiency=getattr(
                        current_pv_twin, "transformer_efficiency", 0.985
                    ),
                )
                self.pv_twin = new_pv_twin
                self._log_message("成功创建并替换了 PVDigitalTwin 实例。")

            except Exception as pv_twin_error:
                self._log_message(f"创建新PVDigitalTwin实例失败: {pv_twin_error}")
                import traceback

                traceback.print_exc()
                return False, f"无法创建新的光伏系统模型: {str(pv_twin_error)}"

            if self.anomaly_model:
                try:
                    if hasattr(self.anomaly_model, "set_threshold"):
                        self.anomaly_model.set_threshold(fault_threshold)
                    self.anomaly_model.pv_digital_twin_model = self.pv_twin
                    self._log_message("异常检测模型已更新。")
                except Exception as anomaly_error:
                    self._log_message(f"更新异常检测模型失败: {anomaly_error}")

            try:
                if hasattr(self, "simulation_engine") and self.simulation_engine:
                    self.simulation_engine.pv_model = self.pv_twin
                    self.simulation_engine.anomaly_model = self.anomaly_model
                    self._log_message("仿真引擎模型引用已更新。")
                else:
                    self._log_message("仿真引擎不存在，将在下次仿真运行时创建。")
            except Exception as engine_error:
                self._log_message(f"更新仿真引擎失败: {engine_error}")

            self._log_message("准备重新运行仿真以应用新设置...")

            # 保存当前日期和原来的仿真结果，以便在重新仿真后合并回今天的数据
            current_system_date = pd.to_datetime(datetime.now()).date()
            today_data = None

            # 如果有现有的仿真结果，保存当天的部分
            if (
                hasattr(self, "simulation_results")
                and not self.simulation_results.empty
                and "date" in self.simulation_results.columns
            ):
                today_data_mask = self.simulation_results["date"].apply(
                    lambda x: pd.to_datetime(x).date() == current_system_date
                )
                if today_data_mask.any():
                    today_data = self.simulation_results[today_data_mask].copy()
                    self._log_message(
                        f"已保存当天({current_system_date})的 {len(today_data)} 条仿真记录，稍后将合并"
                    )

            # 重新运行仿真
            self._run_simulation()
            new_simulation_results = self._process_simulation_results()

            # 如果有保存的今天数据，且新的仿真结果不为空，则合并数据
            if (
                today_data is not None
                and not today_data.empty
                and not new_simulation_results.empty
            ):
                # 将原始的今日数据合并到新的仿真结果中
                # 首先，删除新结果中与今天重叠的数据（如果有的话）
                new_today_mask = new_simulation_results["date"].apply(
                    lambda x: pd.to_datetime(x).date() == current_system_date
                )
                if new_today_mask.any():
                    # 删除新结果中今天的数据
                    new_simulation_results = new_simulation_results[~new_today_mask]

                # 然后，合并原始的今天数据
                self.simulation_results = pd.concat(
                    [new_simulation_results, today_data]
                )
                # 排序并重置索引
                self.simulation_results = self.simulation_results.sort_values(
                    by="datetime"
                ).reset_index(drop=True)
                self._log_message(
                    f"合并完成：新的仿真结果包含 {len(self.simulation_results)} 条记录，其中含有今天的 {len(today_data)} 条原始记录"
                )
            else:
                # 如果没有可合并的今天数据，直接使用新的仿真结果
                self.simulation_results = new_simulation_results

            # 恢复每日能量缓存
            self._daily_energy_cache = daily_energy_cache
            self._log_message(
                f"已恢复每日能量缓存，包含 {len(daily_energy_cache)} 天的历史数据"
            )

            self._log_message("新设置已应用，并重新运行了初始仿真。")

            return True, "设置已成功应用！系统将基于新配置进行仿真。"
        except Exception as e:
            import traceback

            self._log_message(f"应用设置时发生严重错误: {e}")
            traceback.print_exc()
            return False, f"设置应用失败: {str(e)}"

    def reset_simulation(self):
        """重置仿真，重新生成数据"""
        # 清空日志
        PVModelAdapter._simulation_logs = []
        self._log_message("重置仿真开始")

        # 保存每日能量缓存
        daily_energy_cache = (
            getattr(self, "_daily_energy_cache", {})
            if hasattr(self, "_daily_energy_cache")
            else {}
        )

        # 重新生成天气数据
        self._generate_weather_data()

        # 重新运行仿真
        self._run_simulation()

        # 重新处理仿真结果
        self.simulation_results = self._process_simulation_results()

        # 恢复每日能量缓存
        self._daily_energy_cache = daily_energy_cache
        self._log_message(
            f"恢复了每日能量缓存，包含 {len(daily_energy_cache)} 条历史记录"
        )

        self._log_message("仿真重置完成")

        return {"status": "success", "message": "仿真已重置"}

    def pause_simulation(self):
        """暂停仿真"""
        if not self._is_paused:
            self._is_paused = True
            self._log_message("仿真已暂停")
            return {"status": "success", "message": "仿真已暂停"}
        return {"status": "info", "message": "仿真已经处于暂停状态"}

    def resume_simulation(self):
        """恢复仿真"""
        if self._is_paused:
            self._is_paused = False
            self._log_message("仿真已恢复")
            return {"status": "success", "message": "仿真已恢复"}
        return {"status": "info", "message": "仿真已在运行中"}

    def get_simulation_status(self):
        """获取仿真状态"""
        return {
            "is_running": self._is_running,
            "is_paused": self._is_paused,
            "continuous_simulation_active": self._continuous_simulation_active,
        }

    # ==================== 阶段四兼容性方法 ====================
    # 为基础适配器添加阶段四方法的基础实现，确保API兼容性

    def pause_simulation_advanced(self):
        """高级仿真暂停功能（基础实现）"""
        try:
            result = self.pause_simulation()
            return {
                "success": result.get("status") == "success",
                "message": result.get("message", ""),
                "method": "basic",
            }
        except Exception as e:
            return {"success": False, "message": f"暂停失败: {e}", "method": "error"}

    def resume_simulation_advanced(self):
        """高级仿真恢复功能（基础实现）"""
        try:
            result = self.resume_simulation()
            return {
                "success": result.get("status") == "success",
                "message": result.get("message", ""),
                "method": "basic",
            }
        except Exception as e:
            return {"success": False, "message": f"恢复失败: {e}", "method": "error"}

    def update_simulation_parameters(self, params):
        """动态更新仿真参数（基础实现）"""
        try:
            # 基础适配器不支持动态参数更新，但提供兼容接口
            return {
                "success": False,
                "message": "基础适配器不支持动态参数更新，请使用真实仿真系统",
                "updated_params": [],
            }
        except Exception as e:
            return {"success": False, "message": f"参数更新失败: {e}"}

    def change_time_scale(self, time_scale):
        """切换仿真时间尺度（基础实现）"""
        try:
            # 基础适配器不支持时间尺度切换，但提供兼容接口
            return {
                "success": False,
                "message": "基础适配器不支持时间尺度切换，请使用真实仿真系统",
                "time_scale": time_scale,
            }
        except Exception as e:
            return {"success": False, "message": f"时间尺度切换失败: {e}"}

    def get_advanced_simulation_status(self):
        """获取高级仿真状态信息（基础实现）"""
        try:
            # 获取基础状态
            status = self.get_simulation_status()

            # 添加阶段四兼容信息
            status.update(
                {
                    "phase4_features": {
                        "simulation_controller": False,
                        "data_persistence": False,
                        "advanced_control_enabled": False,
                        "note": "使用基础适配器，高级功能不可用",
                    }
                }
            )

            return status
        except Exception as e:
            return {"error": f"获取状态失败: {e}"}

    def query_historical_data(
        self, start_time=None, end_time=None, data_type=None, limit=100
    ):
        """查询历史仿真数据（基础实现）"""
        try:
            # 基础适配器不支持历史数据查询，但提供兼容接口
            return {
                "success": False,
                "message": "基础适配器不支持历史数据查询，请使用真实仿真系统",
                "data": [],
                "query_params": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "data_type": data_type,
                    "limit": limit,
                },
            }
        except Exception as e:
            return {"success": False, "message": f"查询失败: {e}", "data": []}

    def cleanup_old_data(self):
        """清理过期数据（基础实现）"""
        try:
            # 基础适配器不需要数据清理，但提供兼容接口
            return {"success": True, "message": "基础适配器无需数据清理"}
        except Exception as e:
            return {"success": False, "message": f"清理失败: {e}"}

    def get_available_capacity_options(self):
        """
        计算并返回系统可用的容量选项列表，基于单模块功率和每串模块数。
        返回结果为按容量递增顺序排列的字典列表，包含容量值、组串数和详细描述。
        """
        # 使用标准单模块功率
        EFFECTIVE_SINGLE_MODULE_PDC0_W = 300.0

        # 使用当前配置的每串模块数，如果不存在则使用默认值10
        current_pv_twin = self.pv_twin
        modules_per_string_val = getattr(current_pv_twin, "modules_per_string", 10)
        if not isinstance(modules_per_string_val, int) or modules_per_string_val <= 0:
            modules_per_string_val = 10

        # 计算单串功率 (kW)
        single_string_pdc0_kw = (
            EFFECTIVE_SINGLE_MODULE_PDC0_W * modules_per_string_val
        ) / 1000.0

        # 生成1到20串的容量选项 (可根据需要调整范围)
        capacity_options = []
        for num_strings in range(1, 21):  # 1到20串
            capacity_kw = single_string_pdc0_kw * num_strings
            capacity_options.append(
                {
                    "value": capacity_kw,  # 容量值 (kW)
                    "num_strings": num_strings,  # 组串数
                    "description": f"{capacity_kw:.1f} kWp ({num_strings}串 × {modules_per_string_val}模块/串 × {EFFECTIVE_SINGLE_MODULE_PDC0_W/1000.0:.1f}kW/模块)",
                }
            )

        # 返回容量选项列表，已按容量递增排序
        self._log_message(
            f"计算了{len(capacity_options)}个可用系统容量选项，基于每串{modules_per_string_val}个模块"
        )
        return capacity_options

    def schedule_faults_for_next_simulation(self, fault_configs):
        """
        接收前端配置的故障列表，计划在下一次（通常是下一天）的仿真中应用。
        """
        if not isinstance(fault_configs, list):
            self._log_message("错误：计划的故障必须是一个列表。")
            return False, "无效的数据格式，故障配置必须是列表。"

        PVModelAdapter._scheduled_faults_for_next_day = []  # 清空之前的计划
        valid_faults_added = 0
        for config in fault_configs:
            # 可以在这里进行更严格的验证，但基本验证已在视图中完成
            if all(k in config for k in ["type", "start_hour", "end_hour", "severity"]):
                PVModelAdapter._scheduled_faults_for_next_day.append(config)
                valid_faults_added += 1
            else:
                self._log_message(f"警告：跳过无效的故障配置: {config}")

        self._log_message(f"已计划 {valid_faults_added} 个故障将在下一次仿真日应用。")
        if valid_faults_added > 0:
            return (
                True,
                f"已成功计划 {valid_faults_added} 个故障，将在下一天的仿真中模拟。",
            )
        elif not fault_configs:  # 如果传入的是空列表
            return True, "已清除所有计划故障。"
        else:  # 如果传入列表非空，但没有一个是有效的
            return False, "没有有效的故障被添加到计划中。"

    def _apply_scheduled_faults_to_simulation(
        self, current_simulation_day_datetime, weather_df_for_day
    ):
        """
        在为某一天运行仿真之前，检查并应用计划的故障。
        这会直接修改 self.pv_twin 的参数或状态，或调用其方法来模拟故障。
        """
        if not PVModelAdapter._scheduled_faults_for_next_day:
            # 如果没有计划故障，确保PVDigitalTwin内部可能存在的旧计划被清除（如果适用）
            if hasattr(self.pv_twin, "clear_all_scheduled_fault_events"):
                self.pv_twin.clear_all_scheduled_fault_events()
            return

        faults_to_process_now = PVModelAdapter._scheduled_faults_for_next_day
        PVModelAdapter._scheduled_faults_for_next_day = (
            []
        )  # 清空列表，表示这些故障已被"处理"或"发送"给PV模型

        self._log_message(
            f"开始为日期 {current_simulation_day_datetime.strftime('%Y-%m-%d')} 应用 {len(faults_to_process_now)} 个计划故障。"
        )

        # 首先，如果 PVDigitalTwin 有清除旧计划的方法，调用它
        if hasattr(self.pv_twin, "clear_all_scheduled_fault_events"):
            self.pv_twin.clear_all_scheduled_fault_events()
            self._log_message("  已清除PVDigitalTwin中旧的计划故障事件。")

        for fault_config in faults_to_process_now:
            fault_type = fault_config["type"]
            start_hour = fault_config["start_hour"]
            end_hour = fault_config["end_hour"]
            severity = fault_config["severity"]

            # 获取当天的第一个时间戳作为基准，来确定故障的绝对起止时间
            if "datetime" in weather_df_for_day.columns:
                day_start_datetime_from_weather = weather_df_for_day["datetime"].min()
            elif isinstance(weather_df_for_day.index, pd.DatetimeIndex):
                day_start_datetime_from_weather = weather_df_for_day.index.min()
            else:
                self._log_message(
                    f"警告：天气数据中缺少datetime信息，无法应用计划故障 {fault_type}"
                )
                continue

            # 确保使用与 weather_df_for_day 中时间戳相同的日期部分
            fault_start_datetime = day_start_datetime_from_weather.replace(
                hour=start_hour, minute=0, second=0, microsecond=0
            )
            # 结束时间应该是该小时的最后一刻，或者下一个小时的开始
            # 如果 end_hour 是 10，意味着故障持续到 10:59:59。或者说，从 start_hour:00:00 到 (end_hour+1):00:00 之前
            fault_end_datetime = day_start_datetime_from_weather.replace(
                hour=end_hour, minute=0, second=0, microsecond=0
            )
            # 如果希望故障包含整个 end_hour，则 fault_end_datetime 应为 end_hour + 1
            # 或者，如之前设计，持续到 end_hour 的最后一秒
            # fault_end_datetime = day_start_datetime_from_weather.replace(hour=end_hour, minute=59, second=59, microsecond=999999)
            # 若 start=8, end=10, 表示 8:00 至 10:00 (不含10:00) 或者 8:00 至 10:59:59
            # 当前JS侧是 start_hour <= time < end_hour。所以 end_hour 是开区间。
            # 因此，如果end_hour是10，那么故障应该在10:00之前结束。
            # 所以 fault_end_datetime 设置为 end_hour:00:00 是可以的，比较时用 current_time < fault_end_datetime
            # 或者，PVDigitalTwin内部处理时，如果current_time.hour == end_hour，则故障结束。
            # 为了 PVDigitalTwin 的 add_scheduled_fault_event 更清晰，我们传递 end_hour 本身。

            self._log_message(
                f"  故障: {fault_type}, 时间: {start_hour}:00 至 {end_hour}:00 (不含), 严重性: {severity}"
            )

            if hasattr(self.pv_twin, "add_scheduled_fault_event"):
                try:
                    self.pv_twin.add_scheduled_fault_event(
                        fault_type=fault_type,
                        # PVDigitalTwin 将根据 day_start_datetime 和 offset hours 来确定绝对时间
                        fault_day_start_datetime=day_start_datetime_from_weather.replace(
                            hour=0, minute=0, second=0, microsecond=0
                        ),
                        start_hour_offset=start_hour,
                        end_hour_offset=end_hour,  # 传递 end_hour, PVDigitalTwin 内部判断 current_hour < end_hour_offset
                        severity=severity,
                    )
                    self._log_message(
                        f"    已将故障 {fault_type} 传递给PVDigitalTwin进行调度。"
                    )
                    PVModelAdapter._fault_application_log.append(
                        {
                            "applied_on_day": current_simulation_day_datetime.strftime(
                                "%Y-%m-%d"
                            ),
                            "fault_config": fault_config,
                            "status": "Scheduled in PVDigitalTwin",
                        }
                    )
                except Exception as e:
                    self._log_message(
                        f"    应用计划故障 {fault_type} 到 PVDigitalTwin 失败: {e}"
                    )
                    PVModelAdapter._fault_application_log.append(
                        {
                            "applied_on_day": current_simulation_day_datetime.strftime(
                                "%Y-%m-%d"
                            ),
                            "fault_config": fault_config,
                            "status": f"Error scheduling: {e}",
                        }
                    )
            else:
                self._log_message(
                    f"    警告：PVDigitalTwin 没有 add_scheduled_fault_event 方法。无法应用计划故障 {fault_type}。"
                )
                PVModelAdapter._fault_application_log.append(
                    {
                        "applied_on_day": current_simulation_day_datetime.strftime(
                            "%Y-%m-%d"
                        ),
                        "fault_config": fault_config,
                        "status": "PVDigitalTwin method missing",
                    }
                )

        if len(PVModelAdapter._fault_application_log) > 100:
            PVModelAdapter._fault_application_log = (
                PVModelAdapter._fault_application_log[-100:]
            )
