import base64
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def encode_file(file_path: str) -> str:
    """读取文件并返回 base64 编码字符串"""
    with open(file_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


# ========== 1. 图片识别 ==========
def image_recognition(image_path: str):
    model = ChatOpenAI(model="qwen-vl-plus")
    image_data = encode_file(image_path)

    message = HumanMessage(content=[
        {"type": "text", "text": "请描述这张图片的内容"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"},
        },
    ])

    response = model.invoke([message])
    print("【图片识别】", response.content)


# ========== 2. GIF识别 ==========
def gif_recognition(gif_path: str):
    model = ChatOpenAI(model="qwen-vl-plus")
    gif_data = encode_file(gif_path)

    message = HumanMessage(content=[
        {"type": "text", "text": "请描述这个GIF动图的内容"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/gif;base64,{gif_data}"},
        },
    ])

    response = model.invoke([message])
    print("【GIF识别】", response.content)



# ========== 3. PDF识别 ==========
def pdf_recognition(pdf_path: str):
    model = ChatOpenAI(model="qwen-vl-plus")
    pdf_data = encode_file(pdf_path)

    message = HumanMessage(content=[
        {"type": "text", "text": "请总结这个PDF文档的主要内容"},
        {
            "type": "file",
            "base64": pdf_data,
            "mime_type": "application/pdf",
        },
    ])

    response = model.invoke([message])
    print("【PDF识别】", response.content)


# ========== 4. 视频识别 ==========
def video_recognition(video_path: str):
    model = ChatOpenAI(model="qwen-vl-plus")
    video_data = encode_file(video_path)

    message = HumanMessage(content=[
        {"type": "text", "text": "请描述这个视频的内容"},
        {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_data}"},
        },
    ])

    response = model.invoke([message])
    print("【视频识别】", response.content)


if __name__ == "__main__":
    # 修改为你的实际文件路径
    # image_recognition("/mnt/c/Users/zhukunpeng6/Downloads/image.png")
    # gif_recognition("/mnt/c/Users/zhukunpeng6/Downloads/demo.gif")
    pdf_recognition("/mnt/c/Users/zhukunpeng6/Downloads/主机覆盖说明.pdf")
    # video_recognition("/mnt/c/Users/zhukunpeng6/Downloads/video.mp4")
