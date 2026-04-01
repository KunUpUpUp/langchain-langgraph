"""工具层 - 模拟底层能力（相当于 MCP 提供的工具）"""


def read_file(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"文件不存在: {file_path}"


def count_lines(code: str) -> dict:
    """统计代码行数"""
    lines = code.strip().split("\n")
    return {
        "total": len(lines),
        "blank": sum(1 for l in lines if l.strip() == ""),
        "code": sum(1 for l in lines if l.strip() and not l.strip().startswith("#")),
        "comment": sum(1 for l in lines if l.strip().startswith("#")),
    }


def check_style(code: str) -> list[str]:
    """检查代码风格问题"""
    issues = []
    for i, line in enumerate(code.split("\n"), 1):
        if len(line) > 120:
            issues.append(f"第{i}行: 超过120字符")
        if line.rstrip() != line:
            issues.append(f"第{i}行: 行尾有空格")
        if "\t" in line:
            issues.append(f"第{i}行: 使用了tab缩进，建议用空格")
    return issues if issues else ["无风格问题"]


def check_security(code: str) -> list[str]:
    """检查安全隐患"""
    issues = []
    danger_patterns = {
        "eval(": "使用了 eval，存在代码注入风险",
        "exec(": "使用了 exec，存在代码注入风险",
        "os.system(": "使用了 os.system，建议用 subprocess",
        "password": "代码中出现了 password 明文",
        "secret": "代码中出现了 secret 明文",
    }
    for pattern, msg in danger_patterns.items():
        if pattern in code.lower():
            issues.append(msg)
    return issues if issues else ["未发现安全隐患"]
