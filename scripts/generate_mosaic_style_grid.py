import os
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps

def create_mosaic_style_grid():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop'
    output_dir = os.path.join(base_dir, 'Paper', 'latex_source', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Simulating a case analysis with reference images on the left, and generated results on the right
    case_prompt_id = '11_interaction'
    seed = 42
    
    models = ['mosaic', 'xverse', 'psr']
    model_titles = ['MOSAIC', 'XVerse', 'PSR']
    
    # Load one generated image to get dimensions
    first_img_path = os.path.join(base_dir, f'results/mosaic/{case_prompt_id}/{seed}.png')
    with Image.open(first_img_path) as img:
        img_w, img_h = img.size
        
    scale_factor = 0.6
    img_w = int(img_w * scale_factor)
    img_h = int(img_h * scale_factor)
    
    # We will draw placeholders for reference images on the left
    # Make the reference boxes perfectly square to match the source images better
    ref_size = int(min(img_w, img_h) * 0.7)
    
    padding_x = 20
    padding_y = 60
    ref_area_w = ref_size + 40
    top_margin = 80
    
    grid_w = ref_area_w + len(models) * (img_w + padding_x) + 20
    
    # Calculate required height based on max content
    total_ref_h = int(min(img_w, img_h) * 0.7) * 2 + 20
    max_content_h = max(img_h, total_ref_h)
    grid_h = top_margin + max_content_h + padding_y + 80
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        prompt_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        prompt_font = ImageFont.load_default()
        
    # Draw reference placeholder column
    draw.text((20, top_margin // 2), "References", font=title_font, fill='black')
    
    # Draw two dummy reference boxes
    # Load actual reference images if available, otherwise use placeholders
    
    # Calculate y positions to center the two squares vertically relative to the generated images
    total_ref_h = ref_size * 2 + 20
    start_y = top_margin + (img_h - total_ref_h) // 2
    
    # We will just draw a nicer "Image" placeholder to represent the inputs
    # Subject A
    ref1_y = start_y
    subj1_path = os.path.join(base_dir, 'val_dataset', 'black_women.jpg')
    try:
        with Image.open(subj1_path) as s_img:
            s_img = ImageOps.fit(s_img, (ref_size, ref_size), Image.Resampling.LANCZOS)
            grid_img.paste(s_img, (20, ref1_y))
    except Exception:
        draw.rectangle([20, ref1_y, 20+ref_size, ref1_y+ref_size], fill='#e6f2ff')
    draw.rectangle([20, ref1_y, 20+ref_size, ref1_y+ref_size], outline='black', width=2)
    
    # Subject B
    ref2_y = start_y + ref_size + 20
    subj2_path = os.path.join(base_dir, 'val_dataset', 'western_woman.jpg')
    try:
        with Image.open(subj2_path) as s_img:
            s_img = ImageOps.fit(s_img, (ref_size, ref_size), Image.Resampling.LANCZOS)
            grid_img.paste(s_img, (20, ref2_y))
    except Exception:
        draw.rectangle([20, ref2_y, 20+ref_size, ref2_y+ref_size], fill='#e6ffe6')
    draw.rectangle([20, ref2_y, 20+ref_size, ref2_y+ref_size], outline='black', width=2)
    
    # Draw column titles
    for i, title in enumerate(model_titles):
        x = ref_area_w + i * (img_w + padding_x) + img_w // 2
        y = top_margin // 2
        bbox = draw.textbbox((0, 0), title, font=title_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        color = '#d62728' if title == 'MOSAIC' else 'black'
        draw.text((x - text_w // 2, y - text_h // 2), title, font=title_font, fill=color)
        
    # Draw generated images
    for c, model in enumerate(models):
        img_path = os.path.join(base_dir, f'results/{model}/{case_prompt_id}/{seed}.png')
        try:
            with Image.open(img_path) as img:
                img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
                x_offset = ref_area_w + c * (img_w + padding_x)
                grid_img.paste(img, (x_offset, top_margin))
                
                # Add border
                border_color = '#d62728' if model == 'mosaic' else 'black'
                border_width = 4 if model == 'mosaic' else 2
                draw.rectangle([x_offset, top_margin, x_offset+img_w, top_margin+img_h], outline=border_color, width=border_width)
                
                if model == 'mosaic':
                    # Add a checkmark or indicator
                    draw.text((x_offset + 10, top_margin + 10), "[Success]", font=text_font, fill='green')
                else:
                    draw.text((x_offset + 10, top_margin + 10), "[Bleeding]", font=text_font, fill='red')
                    
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
    # Add prompt at the bottom
    # We need to make sure the text is placed below both the generated images AND the reference images
    max_y_content = max(top_margin + img_h, start_y + ref_size * 2 + 20)
    
    prompt_text = "Prompt: A black woman and a western woman shaking hands."
    draw.text((20, max_y_content + 20), prompt_text, font=prompt_font, fill='#333333')
            
    output_path = os.path.join(output_dir, 'fig_case_analysis.png')
    grid_img.save(output_path)
    print(f"Saved MOSAIC-style case analysis to {output_path}")

if __name__ == '__main__':
    create_mosaic_style_grid()