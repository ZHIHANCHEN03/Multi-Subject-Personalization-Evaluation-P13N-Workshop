import os
from PIL import Image

def resize_and_pad_image(image_input, target_size=(512, 512), pad_color=(255, 255, 255)):
    """
    规范化图片尺寸 (Standardize Image Dimensions):
    1. 如果图片某一边大于 target_size，等比例压缩（不裁剪），使最长边匹配 512。
    2. 如果图片小于 target_size，不放大原图（避免模糊）。
    3. 无论压缩还是原本较小，最后都用 pad_color (默认纯白) 填充到正好 512x512，将原图居中。
    
    参数:
        image_input: 可以是图片的本地路径 (str) 或者 PIL.Image.Image 对象
        target_size: 目标尺寸 tuple (width, height)，默认 (512, 512)
        pad_color: 补白背景色 tuple (R, G, B)，默认纯白 (255, 255, 255)
    返回:
        PIL.Image.Image: 尺寸必定为 target_size 的标准化图片
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    original_width, original_height = img.size
    target_width, target_height = target_size

    # 计算宽和高的缩放比例
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    # 取较小的比例，保证长边不会超出目标尺寸
    scale = min(scale_w, scale_h)

    # 如果原图比目标尺寸小，则不放大（scale 强制设为 1.0）
    if scale > 1.0:
        scale = 1.0

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 执行缩放 (LANCZOS 是一种高质量的降采样算法)
    if scale < 1.0:
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建纯白背景图
    new_img = Image.new("RGB", target_size, pad_color)

    # 计算居中粘贴的坐标
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # 将处理后的原图粘贴到白底图的中心
    new_img.paste(img, (paste_x, paste_y))

    return new_img
