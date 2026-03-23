import os
import json
from PIL import Image, ImageDraw, ImageFont

def get_prompts():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop'
    json_path = os.path.join(base_dir, 'eval_outputs', 'mosaic_results.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    prompts = {}
    for item in data['results']:
        if item['prompt_id'] not in prompts:
            prompts[item['prompt_id']] = item['prompt']
    return prompts

def create_image_grid():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop'
    output_dir = os.path.join(base_dir, 'Paper', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    prompts = get_prompts()
    
    cases = [
        ('01_no_interaction', 42, '2 Subjects (No Interaction)'),
        ('06_occlusion', 42, '2 Subjects (Occlusion)'),
        ('11_interaction', 42, '2 Subjects (Interaction)'),
        ('16_no_interaction', 42, '4 Subjects')
    ]
    
    models = ['mosaic', 'xverse', 'psr']
    model_titles = ['MOSAIC', 'XVerse', 'PSR']
    
    # Load first image to get dimensions
    first_img_path = os.path.join(base_dir, f'results/mosaic/{cases[0][0]}/{cases[0][1]}.png')
    with Image.open(first_img_path) as img:
        img_w, img_h = img.size
        
    # We will scale them down slightly to make the grid manageable
    scale_factor = 0.5
    img_w = int(img_w * scale_factor)
    img_h = int(img_h * scale_factor)
    
    padding_x = 20
    padding_y = 60
    text_area_w = 300
    top_margin = 60
    
    grid_w = text_area_w + len(models) * (img_w + padding_x)
    grid_h = top_margin + len(cases) * (img_h + padding_y)
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    # Try to load a nice font, fallback to default
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        prompt_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        prompt_font = ImageFont.load_default()
        
    # Draw column titles
    for i, title in enumerate(model_titles):
        x = text_area_w + i * (img_w + padding_x) + img_w // 2
        y = top_margin // 2
        # Center text
        bbox = draw.textbbox((0, 0), title, font=title_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((x - text_w // 2, y - text_h // 2), title, font=title_font, fill='black')
        
    # Draw rows
    for r, (prompt_id, seed, case_title) in enumerate(cases):
        y_offset = top_margin + r * (img_h + padding_y)
        
        # Draw row title and prompt
        draw.text((20, y_offset + 20), case_title, font=text_font, fill='black')
        
        prompt_text = prompts.get(prompt_id, "")
        
        # Wrap prompt text
        words = prompt_text.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            bbox = draw.textbbox((0, 0), " ".join(current_line), font=prompt_font)
            if bbox[2] - bbox[0] > text_area_w - 40:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]
        lines.append(" ".join(current_line))
        
        for i, line in enumerate(lines):
            draw.text((20, y_offset + 60 + i * 25), line, font=prompt_font, fill='#333333')
            
        # Draw the original Reference Images for this case
        # Instead of just text, we'll draw placeholders representing the reference subjects
        # to match the user's request (similar to MOSAIC paper)
        # We'll put them right below the prompt text
        ref_y = y_offset + 60 + len(lines) * 25 + 10
        ref_box_size = 40
        num_subjects = 2 if '2' in case_title else 4
        
        draw.text((20, ref_y), "Inputs:", font=prompt_font, fill='blue')
        for i in range(num_subjects):
            rx = 20 + 70 + i * (ref_box_size + 10)
            draw.rectangle([rx, ref_y, rx+ref_box_size, ref_y+ref_box_size], fill='#e6f2ff', outline='black')
            draw.text((rx+10, ref_y+10), f"S{i+1}", font=prompt_font, fill='black')
            
        # Draw images
        for c, model in enumerate(models):
            img_path = os.path.join(base_dir, f'results/{model}/{prompt_id}/{seed}.png')
            try:
                with Image.open(img_path) as img:
                    img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
                    x_offset = text_area_w + c * (img_w + padding_x)
                    grid_img.paste(img, (x_offset, y_offset))
                    
                    # Add border
                    border_color = 'black'
                    draw.rectangle([x_offset, y_offset, x_offset+img_w, y_offset+img_h], outline=border_color, width=2)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    output_path = os.path.join(output_dir, 'fig_qualitative_comparison.png')
    grid_img.save(output_path)
    print(f"Saved qualitative comparison grid to {output_path}")

if __name__ == '__main__':
    create_image_grid()
