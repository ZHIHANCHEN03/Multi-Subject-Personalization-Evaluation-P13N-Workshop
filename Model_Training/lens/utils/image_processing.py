import os
from typing import Union, Tuple
from PIL import Image, ImageOps

def resize_and_pad_image(
    image_input: Union[str, Image.Image], 
    target_size: Tuple[int, int] = (512, 512), 
    pad_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Standardize Image Dimensions with high-fidelity resampling and padding.
    
    1. Reads image safely (closing file handlers to prevent Too Many Open Files in DataLoaders).
    2. Corrects EXIF orientation metadata to prevent rotated images.
    3. Handles Alpha channels (PNGs) by composing over the pad_color before RGB conversion.
    4. Downscales proportionally if the image is larger than target_size.
    5. Does NOT upscale if the image is smaller than target_size (prevents blur).
    6. Pads the remaining area with pad_color to exactly match target_size.
    
    Args:
        image_input: File path (str) or PIL.Image.Image object.
        target_size: Target dimension tuple (width, height). Default is (512, 512).
        pad_color: Padding background color tuple (R, G, B). Default is pure white.
        
    Returns:
        PIL.Image.Image: The standardized image strictly matching target_size.
    """
    
    # 1. Safe loading & EXIF correction
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        # Use context manager to prevent file handler leaks in multi-worker DataLoaders
        with Image.open(image_input) as f_img:
            img = ImageOps.exif_transpose(f_img)
            img.load()  # Force load into memory so we can safely exit context manager
    else:
        img = ImageOps.exif_transpose(image_input)

    # 2. Handle Alpha Channel (RGBA -> RGB over pad_color)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_background = Image.new("RGB", img.size, pad_color)
        alpha_background.paste(img, mask=img.convert("RGBA").split()[3])
        img = alpha_background
    else:
        img = img.convert("RGB")

    original_width, original_height = img.size
    target_width, target_height = target_size

    # 3. Calculate scaling factors
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)

    # Prevent upscaling
    if scale > 1.0:
        scale = 1.0

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 4. High-quality downsampling
    if scale < 1.0:
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 5. Create canvas and paste to center
    new_img = Image.new("RGB", target_size, pad_color)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_img.paste(img, (paste_x, paste_y))

    return new_img
