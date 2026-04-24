# VLM Multi-Subject Eval

这个子目录用于做多主体生成的 `VLM-as-a-Judge` 评测，直接调用 `OpenAI` 或 `Gemini` API，对 `test_v1.json` 里的 A/B 候选图进行成对比较。

## 功能

- 读取现有 `Model_Training/data_v1/test_v1.json`
- 将 `prompt + subject reference images + candidate A/B` 一起发给 VLM
- 输出固定字段：
  - `a_subject_existence`
  - `a_subject_appearance`
  - `a_interaction_alignment`
  - `b_subject_existence`
  - `b_subject_appearance`
  - `b_interaction_alignment`
  - `better_candidate`，只允许 `A` 或 `B`
  - `reason`，限制在 `25` 个词以内
- 自动保存：
  - 逐条结果 `jsonl`
  - 汇总统计 `summary.json`
- 支持断点续跑：默认会跳过已经写入输出文件的 `task_id`

## 文件

- `run_vlm_eval.py`: 主评测脚本
- `requirements.txt`: 最小依赖
- `.env.example`: API key 示例
- `results/`: 默认输出目录，运行后自动创建

## 安装

```bash
cd /Users/bytedance/Documents/multi_subject_generation/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/vlm_multisubject_eval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置 API Key

任选一个 provider：

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

或：

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

`Gemini` 也兼容读取 `GOOGLE_API_KEY`。

## 运行示例

### OpenAI

```bash
python run_vlm_eval.py \
  --provider openai \
  --max-samples 10
```

### Gemini

```bash
python run_vlm_eval.py \
  --provider gemini \
  --max-samples 10
```

### 指定输出路径

```bash
python run_vlm_eval.py \
  --provider openai \
  --output ./results/openai_debug.jsonl \
  --max-samples 5 \
  --overwrite
```

### 只跑一条样本

```bash
python run_vlm_eval.py \
  --provider gemini \
  --only-task-id combo_066
```

## 输出格式

每行 `jsonl` 大致如下：

```json
{
  "task_id": "combo_066",
  "provider": "openai",
  "model": "gpt-5.4-mini",
  "a_subject_existence": 0,
  "a_subject_appearance": 0,
  "a_interaction_alignment": 0,
  "b_subject_existence": 1,
  "b_subject_appearance": 1,
  "b_interaction_alignment": 1,
  "better_candidate": "B",
  "reason": "B preserves all subjects and bindings better.",
  "ground_truth": "B",
  "correct": true,
  "metadata": {
    "ratio_type": "all_human",
    "model_A_name": "mosaic",
    "model_B_name": "nano_banana"
  }
}
```

对应还会生成一个同名的 `*.summary.json`，包含总准确率和按 `subject_count` 分组的准确率。

## 默认路径

- 数据集默认读取：
  - `/Users/bytedance/Documents/multi_subject_generation/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/data_v1/test_v1.json`
- 图片相对路径默认相对于：
  - `/Users/bytedance/Documents/multi_subject_generation/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training`

如果后面你扩展到别的数据集，可以通过参数覆盖：

```bash
python run_vlm_eval.py \
  --provider openai \
  --dataset /path/to/your_eval.json \
  --base-dir /path/to/your_project_root
```

## 注意事项

- 当前模型已写死：
  - `openai -> gpt-5.4-mini`
  - `gemini -> gemini-3.1-flash`
- 这个脚本会把样本中的全部 reference image 都发给 VLM。
- 所有 reference 和 candidate 在发送前都会统一做等比缩放并白底 padding 到 `512x512`。
- `OpenAI` 和 `Gemini` 可能偶尔返回非严格 JSON，脚本已经做了基础提取和规范化，并会强制裁剪 `reason` 到 `25` 个词以内，但仍建议先用 `--max-samples 3` 小规模试跑。
- 目前这是一个通用 pairwise judge baseline，适合快速建立闭源 VLM 上界；如果后面需要更严格的 rubric，可以继续细化 prompt 或拆成多阶段评审。
