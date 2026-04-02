"""Agent - 自动扫描加载 Skill，生产级方式"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from tools import read_file, count_lines, check_style, check_security

client = OpenAI()

SKILLS_DIR = Path(__file__).parent / "skills"


# ========== 1. 自动扫描 skills 目录，加载所有 Skill 的元信息 ==========
def load_skills(skills_dir: Path) -> dict:
    """扫描目录，解析 front-matter，只加载 name + description"""
    skills = {}
    for md_file in skills_dir.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # 解析 front-matter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                meta = {}
                for line in parts[1].strip().split("\n"):
                    if ":" in line and not line.strip().startswith("-"):
                        key, val = line.split(":", 1)
                        meta[key.strip()] = val.strip()

                skills[md_file.stem] = {
                    "name": meta.get("name", md_file.stem),
                    "description": meta.get("description", ""),
                    "file": str(md_file),
                }
    return skills


# ========== 2. 工具注册（生产中也可以动态扫描） ==========
TOOLS_MAP = {
    "read_file": read_file,
    "count_lines": count_lines,
    "check_style": check_style,
    "check_security": check_security,
}

OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "读取文件内容", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "count_lines", "description": "统计代码行数", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "check_style", "description": "检查代码风格问题", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "check_security", "description": "检查代码安全隐患", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]


# ========== 3. LLM 根据 description 匹配 Skill ==========
def match_skill(user_input: str, skills: dict) -> str | None:
    skill_list = "\n".join(f"- {k}: {v['description']}" for k, v in skills.items())
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": f"根据用户输入，从以下技能中选择最匹配的，只返回技能key名称，没有匹配的返回none:\n{skill_list}"},
            {"role": "user", "content": user_input},
        ],
    )
    skill_name = response.choices[0].message.content.strip()
    return skill_name if skill_name in skills else None


# ========== 4. 执行 ==========
def run_agent(user_input: str):
    # 启动时扫描加载所有 Skill 元信息
    skills = load_skills(SKILLS_DIR)
    print(f"[已加载 {len(skills)} 个技能: {', '.join(s['name'] for s in skills.values())}]\n")

    # 匹配 Skill
    skill_name = match_skill(user_input, skills)

    if skill_name:
        skill_content = Path(skills[skill_name]["file"]).read_text(encoding="utf-8")
        system_prompt = f"请严格按照以下技能指南执行任务:\n\n{skill_content}"
        print(f"[匹配到技能: {skills[skill_name]['name']}]\n")
    else:
        system_prompt = "你是一个有用的助手。"
        print("[未匹配到技能，使用通用模式]\n")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    while True:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=OPENAI_TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print(msg.content)
            break

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  [调用工具: {name}]")
            result = TOOLS_MAP[name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result,
            })


if __name__ == "__main__":
    # run_agent("帮我审查 test_code.py 这个文件的代码质量")
    # run_agent("帮我审查 test_code2.py 这个文件的代码质量")
    run_agent("帮我解释 test_code.py 这个文件的代码")
