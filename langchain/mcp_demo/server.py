"""MCP 服务端示例 - 提供天气查询工具"""
from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务端
mcp = FastMCP("天气服务")


# 用装饰器注册工具
@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴天 25°C",
        "上海": "多云 22°C",
        "广州": "小雨 28°C",
    }
    return weather_data.get(city, f"{city} 暂无天气数据")


@mcp.tool()
def get_temperature(city: str) -> str:
    """获取指定城市的温度"""
    temp_data = {
        "北京": "25°C",
        "上海": "22°C",
        "广州": "28°C",
    }
    return temp_data.get(city, f"{city} 暂无温度数据")


if __name__ == "__main__":
    # stdio 模式启动，客户端通过标准输入输出通信
    mcp.run(transport="stdio")
