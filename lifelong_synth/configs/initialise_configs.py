import json
import logging
from pathlib import Path

# Raw configuration data
config = {
    "growing_up_location": {
        "country": {
            "allowed_values": [
                {"code": "china", "label_zh": "中国", "label_en": "China", "m49": "156"},
                {"code": "uk", "label_zh": "英国", "label_en": "United Kingdom", "m49": "826"}
            ]
        },
        "city": {
            "type": "string",
            "description": "City where the agent primarily grew up"
        }
    },
    "current_living_location": {
        "country": {
            "allowed_values": [
                {"code": "china", "label_zh": "中国", "label_en": "China", "m49": "156"},
                {"code": "uk", "label_zh": "英国", "label_en": "United Kingdom", "m49": "826"}
            ]
        },
        "city": {
            "type": "string",
            "description": "Current city of residence"
        }
    },
    "language_options": {
        "allowed_values": [
            {"code": "zh", "label_zh": "中文", "label_en": "Chinese"},
            {"code": "en", "label_zh": "英文", "label_en": "English"}
        ],
        "default": ["zh"]
    },
    "target_education_level": {
        "allowed_values": [
            {"code": "0", "label_zh": "学前教育", "label_en": "Early childhood education"},
            {"code": "1", "label_zh": "小学教育", "label_en": "Primary education"},
            {"code": "2", "label_zh": "初中教育", "label_en": "Lower secondary education"},
            {"code": "3", "label_zh": "高中教育", "label_en": "Upper secondary education"},
            {"code": "4", "label_zh": "中学后非高等教育", "label_en": "Post-secondary non-tertiary"},
            {"code": "5", "label_zh": "短周期高等教育", "label_en": "Short-cycle tertiary"},
            {"code": "6", "label_zh": "本科或同等", "label_en": "Bachelor's or equivalent"},
            {"code": "7", "label_zh": "硕士或同等", "label_en": "Master's or equivalent"},
            {"code": "8", "label_zh": "博士或同等", "label_en": "Doctoral or equivalent"}
        ],
        "standard": "ISCED_2011"
    },
    "target_occupation_group": {
        "allowed_values": [
            {"code": "student", "label_zh": "学生", "label_en": "Student"},
            {"code": "manager_executive", "label_zh": "管理者 / 高管", "label_en": "Manager / Executive"},
            {"code": "professional_finance_law_consulting", "label_zh": "专业服务（金融/法律/咨询）", "label_en": "Professional Services"},
            {"code": "professional_tech_research", "label_zh": "技术 / 研发", "label_en": "Technology / Research"},
            {"code": "professional_health_education", "label_zh": "医疗 / 教育", "label_en": "Healthcare / Education"},
            {"code": "office_admin_support", "label_zh": "办公室 / 行政支持", "label_en": "Office / Administrative Support"},
            {"code": "sales_customer_service", "label_zh": "销售 / 客服", "label_en": "Sales / Customer Service"},
            {"code": "service_hospitality_retail", "label_zh": "服务业 / 餐饮 / 零售", "label_en": "Service / Retail / Hospitality"},
            {"code": "skilled_trades_technical_ops", "label_zh": "技工 / 技术操作", "label_en": "Skilled Trades / Technical Ops"},
            {"code": "manual_logistics_transport", "label_zh": "物流 / 运输 / 体力劳动", "label_en": "Logistics / Transport / Manual Work"},
            {"code": "public_service_military", "label_zh": "公共部门 / 军警", "label_en": "Public Service / Military"},
            {"code": "self_employed_creator", "label_zh": "自由职业 / 创作者 / 创业者", "label_en": "Self-employed / Creator / Entrepreneur"}
        ]
    }
}

def transform_config(data):
    """
    递归遍历字典，将 allowed_values 列表转换为以 code 为键的字典。
    """
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if key == "allowed_values" and isinstance(value, list):
                # 转换逻辑：将列表转换为映射表
                transformed_map = {}
                for item in value:
                    if isinstance(item, dict) and "code" in item:
                        # 拷贝一份避免修改原数据，提取 code 作为 key
                        item_copy = item.copy()
                        code_key = item_copy.pop("code")
                        transformed_map[code_key] = item_copy
                new_data[key] = transformed_map
            else:
                # 递归处理嵌套字典或列表
                new_data[key] = transform_config(value)
        return new_data
    elif isinstance(data, list):
        return [transform_config(item) for item in data]
    else:
        return data

# 执行转换
transformed_config = transform_config(config)

# 保存路径
output_path = Path(__file__).parent / "persona_schema.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

# 写入文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(transformed_config, f, ensure_ascii=False, indent=2)

logging.info(f"✅ 转换完成！配置文件已保存至: {output_path.resolve()}")

# 打印示例结构确认效果
logging.info("\n转换后的 country 示例:")
logging.info(json.dumps(transformed_config["growing_up_location"]["country"]["allowed_values"]["china"], ensure_ascii=False, indent=2))