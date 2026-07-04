import os
from typing import Union, Tuple
from PIL import Image, ImageOps

def load_image_with_safety(
    image_input: Union[str, Image.Image], 
) -> Image.Image:
    """
    Standardize Image Loading with high-fidelity safety checks (Optimized for VLM dynamic resolution).
    
    1. Reads image safely (closing file handlers to prevent Too Many Open Files in DataLoaders).
    2. Corrects EXIF orientation metadata to prevent rotated images.
    3. Handles Alpha channels (PNGs) by composing over a white background before RGB conversion.
    4. Does NOT resize or pad. Preserves original aspect ratio to allow the VLM's AutoProcessor 
       to utilize its dynamic resolution and patching mechanisms.
    
    Args:
        image_input: File path (str) or PIL.Image.Image object.
        
    Returns:
        PIL.Image.Image: The standardized RGB image preserving original dimensions.
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

    # 2. Handle Alpha Channel (RGBA -> RGB over white background)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_background = Image.new("RGB", img.size, (255, 255, 255))
        alpha_background.paste(img, mask=img.convert("RGBA").split()[3])
        img = alpha_background
    else:
        img = img.convert("RGB")

    return img
