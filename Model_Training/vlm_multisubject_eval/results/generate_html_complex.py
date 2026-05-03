import json
import random
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent
DATA_DIR = RESULTS_DIR.parents[2].parent / "data"

FILE_25 = RESULTS_DIR / "2_5_merged_sorted.jsonl"
FILE_31 = RESULTS_DIR / "3_1_merged_sorted.jsonl"
OUTPUT_HTML = RESULTS_DIR / "visualization_complex.html"

# Load JSONL
def load_jsonl(path):
    data = {}
    if not path.exists(): return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if "task_id" in obj:
                data[str(obj["task_id"])] = obj
    return data

res_25 = load_jsonl(FILE_25)
res_31 = load_jsonl(FILE_31)

# Find intersection where both models agree
agreed_tasks = []
strict_agreed_tasks = []
for tid, r25 in res_25.items():
    if tid in res_31:
        r31 = res_31[tid]
        w25, w31 = r25.get("winner"), r31.get("winner")
        if w25 == w31 and w25 in ["A", "B"]:
            agreed_tasks.append((tid, r25, r31))
            
            # Check strict agreement on all sub-scores
            scores_match = all(
                r25.get(f"{cand}_{metric}") == r31.get(f"{cand}_{metric}")
                for cand in ["a", "b"]
                for metric in ["existence", "appearance", "interaction"]
            )
            
            if scores_match:
                metadata = r25.get("metadata", {})
                num_subjects = len(metadata.get("people_names", [])) + len(metadata.get("object_names", []))
                
                # Filter out cases where Candidate B has all 0s
                b_all_zeros = all(r25.get(f"b_{metric}", 0) == 0 for metric in ["existence", "appearance", "interaction"])
                
                # Only keep cases with >= 6 subjects
                if not b_all_zeros and num_subjects >= 6:
                    strict_agreed_tasks.append((tid, r25, r31))

# Select a random sample of 50 strict agreed tasks to display
random.seed(42)
sample_tasks = random.sample(strict_agreed_tasks, min(50, len(strict_agreed_tasks)))

# Generate HTML
html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Multi-Subject Generation - LLM-as-Judge Alignment Demo</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.4; color: #333; max-width: 1400px; margin: 0 auto; padding: 10px; background: #fff; font-size: 14px; }}
        h1, h2 {{ text-align: center; color: #2c3e50; margin-bottom: 5px; }}
        .stats {{ text-align: center; background: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #e9ecef; margin-bottom: 20px; font-size: 13px; }}
        .card {{ background: #fff; border-radius: 8px; padding: 15px; margin-bottom: 30px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.05); page-break-inside: avoid; }}
        .prompt-section {{ background: #f1f8ff; padding: 10px 15px; border-radius: 6px; margin-bottom: 15px; font-weight: 500; border-left: 4px solid #3498db; }}
        .refs-section {{ display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px; padding: 10px; background: #fff; border: 1px dashed #ccc; border-radius: 6px; align-items: flex-start; }}
        .ref-item {{ text-align: center; font-size: 11px; color: #666; width: 80px; display: flex; flex-direction: column; align-items: center; }}
        .ref-item img {{ width: 100%; height: 80px; object-fit: cover; border-radius: 4px; border: 1px solid #ddd; }}
        .ref-item-label {{ word-break: break-word; line-height: 1.2; margin-top: 4px; width: 100%; }}
        .split-container {{ display: flex; gap: 20px; }}
        .cand-col {{ flex: 1; display: flex; flex-direction: column; gap: 10px; padding: 10px; border-radius: 6px; background: #fafafa; border: 1px solid #eee; }}
        .cand-col.winner {{ background: #f0fdf4; border: 2px solid #2ecc71; }}
        .cand-header {{ text-align: center; font-size: 1.2em; font-weight: bold; margin-bottom: 5px; }}
        .cand-header .badge {{ font-size: 0.8em; background: #2ecc71; color: white; padding: 2px 8px; border-radius: 12px; vertical-align: middle; margin-left: 5px; }}
        .img-container {{ text-align: center; }}
        .img-container img {{ max-width: 100%; border-radius: 6px; object-fit: contain; max-height: 350px; border: 1px solid #ccc; }}
        .eval-section {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px; font-size: 13px; }}
        .eval-header {{ font-weight: bold; color: #555; margin-bottom: 5px; border-bottom: 1px solid #eee; padding-bottom: 3px; display: flex; justify-content: space-between; }}
        .flaw-log {{ background: #fff5f5; border-left: 3px solid #e74c3c; padding: 6px 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space: pre-wrap; word-break: break-word; margin-bottom: 8px; border-radius: 0 4px 4px 0; }}
        .score-grid {{ display: flex; gap: 10px; justify-content: center; }}
        .score-pill {{ padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 11px; background: #eee; }}
        .score-1 {{ color: #15803d; background: #dcfce7; border: 1px solid #bbf7d0; }}
        .score-0 {{ color: #b91c1c; background: #fee2e2; border: 1px solid #fecaca; }}

    </style>
</head>
<body>

    <h1>SOP-Guided LLM-as-Judge Alignment Demo (>= 6 Subjects)</h1>
    <div class="stats">
        <p><strong>Total Valid 2.5 Tasks:</strong> {len(res_25)} &nbsp;|&nbsp; <strong>Total Valid 3.1 Tasks:</strong> {len(res_31)}</p>
        <p><strong>Preference Agreement:</strong> {len(agreed_tasks)} ({(len(agreed_tasks)/len(res_25)*100):.1f}%) &nbsp;|&nbsp; <strong>Strict Sub-score Agreement (>=6 subjects, B>0):</strong> {len(strict_agreed_tasks)}</p>
        <p>Displaying {len(sample_tasks)} random examples from the <strong>strict agreement & complex scene</strong> subset.</p>
    </div>

"""

def format_scores(res, cand):
    e = res.get(f"{cand}_existence", 0)
    a = res.get(f"{cand}_appearance", 0)
    i = res.get(f"{cand}_interaction", 0)
    return f"""
    <div class="score-grid">
        <div class="score-pill"><span class="score-{e}">Exist: {e}</span></div>
        <div class="score-pill"><span class="score-{a}">App: {a}</span></div>
        <div class="score-pill"><span class="score-{i}">Int: {i}</span></div>
    </div>
    """

def extract_flaw(raw_text, cand):
    # Attempt to extract flaw log from raw response text
    try:
        data = json.loads(raw_text)
        return data.get(f"{cand}_flaw_log", "None")
    except:
        return "Parse Error or None"

for tid, r25, r31 in sample_tasks:
    prompt = r25.get("prompt", "No prompt found.")
    winner = r25.get("winner", "Unknown")
    
    # Extract references
    metadata = r25.get("metadata", {})
    people_names = metadata.get("people_names", [])
    object_names = metadata.get("object_names", [])
    ref_names = people_names + object_names
    
    refs_html = ""
    if ref_names:
        refs_html += '<div class="refs-section">'
        for ref_name in ref_names:
            ref_path_jpg = f"../../../../data/refs/{ref_name}.jpg"
            ref_path_png = f"../../../../data/refs/{ref_name}.png"
            refs_html += f'''
            <div class="ref-item">
                <img src="{ref_path_jpg}" onerror="this.onerror=null; this.src='{ref_path_png}';" alt="{ref_name}">
                <div class="ref-item-label">{ref_name}</div>
            </div>
            '''
        refs_html += '</div>'

    # Use relative paths so it works directly from the HTML file
    img_A_path = f"../../../../data/A/{tid}.png"
    img_B_path_jpg = f"../../../../data/B/{tid}.jpg"
    img_B_path_png = f"../../../../data/B/{tid}.png"


    raw_25 = r25.get("raw_response_text", "{}")
    raw_31 = r31.get("raw_response_text", "{}")

    a_flaw_25 = extract_flaw(raw_25, "a")
    b_flaw_25 = extract_flaw(raw_25, "b")
    a_flaw_31 = extract_flaw(raw_31, "a")
    b_flaw_31 = extract_flaw(raw_31, "b")

    html += f"""
    <div class="card">
        <div style="color: #95a5a6; font-size: 0.85em; margin-bottom: 5px;">Task ID: {tid}</div>
        <div class="prompt-section">{prompt}</div>
        {refs_html}
        
        <div class="split-container">
            <!-- Candidate A Column -->
            <div class="cand-col {'winner' if winner == 'A' else ''}">
                <div class="cand-header">
                    Candidate A
                    {'<span class="badge">Winner</span>' if winner == 'A' else ''}
                </div>
                <div class="img-container">
                    <img src="{img_A_path}" onerror="this.onerror=null; this.src='../../../../data/A/{tid}.jpg';" alt="Image A">
                </div>
                
                <div class="eval-section">
                    <div class="eval-header">Gemini 2.5 Flash</div>
                    <div class="flaw-log">{a_flaw_25}</div>
                    {format_scores(r25, 'a')}
                </div>
                
                <div class="eval-section">
                    <div class="eval-header">Gemini 3.1 Flash Lite</div>
                    <div class="flaw-log">{a_flaw_31}</div>
                    {format_scores(r31, 'a')}
                </div>
            </div>

            <!-- Candidate B Column -->
            <div class="cand-col {'winner' if winner == 'B' else ''}">
                <div class="cand-header">
                    Candidate B
                    {'<span class="badge">Winner</span>' if winner == 'B' else ''}
                </div>
                <div class="img-container">
                    <img src="{img_B_path_jpg}" onerror="this.onerror=null; this.src='{img_B_path_png}';" alt="Image B">
                </div>
                
                <div class="eval-section">
                    <div class="eval-header">Gemini 2.5 Flash</div>
                    <div class="flaw-log">{b_flaw_25}</div>
                    {format_scores(r25, 'b')}
                </div>
                
                <div class="eval-section">
                    <div class="eval-header">Gemini 3.1 Flash Lite</div>
                    <div class="flaw-log">{b_flaw_31}</div>
                    {format_scores(r31, 'b')}
                </div>
            </div>
        </div>
    </div>
    """

html += """
</body>
</html>
"""

OUTPUT_HTML.write_text(html, encoding="utf-8")

print(f"Generated visualization at {OUTPUT_HTML}")
