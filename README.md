# Kana-Only Japanese Text Generation via GPT-2 Fine-Tuning

Fine-tuned GPT-2 (124M) to generate Japanese text using **only hiragana and katakana — no kanji** — by combining instruction fine-tuning with a novel logit-masking inference constraint.

Built as a final project for an LLM course, based on *Build a Large Language Model From Scratch* (Ch. 7) by Sebastian Raschka.

---

## Motivation

GPT-2 generates Japanese text freely, including kanji. As a beginner Japanese learner at the N5/N4 level, I can read hiragana and katakana but not kanji — so standard GPT-2 output is unreadable to me. This project fine-tunes GPT-2 to output only kana, combining a soft learned preference (fine-tuning) with a hard guaranteed constraint (logit masking).

---

## Approach

Two complementary mechanisms enforce kana-only output:

### 1. Instruction Fine-Tuning (soft constraint)
Fine-tuned GPT-2 on 319 custom hiragana instruction-response pairs in Alpaca format. The model learns to *prefer* kana output from training data.

Each training entry looks like:
```json
{
  "instruction": "ひらがなだけでかいてください：",
  "input": "東京は大きい都市です。",
  "output": "とうきょうはおおきいとしです。"
}
```

Kanji → hiragana conversion handled by `pykakasi`. Prompt tokens are masked with `-100` so gradients only flow from response tokens.

### 2. Logit Masking (hard constraint)
At inference time, all 51 kanji-containing tokens in GPT-2's vocabulary have their logits set to `-inf` before softmax. This gives them exactly 0 probability — the model **cannot** sample kanji regardless of what it has learned.

```python
logits[:, kanji_mask] = float('-inf')  # kanji tokens become impossible
```

51 tokens blocked → 50,206 safe tokens remain.

---

## Results

| Condition | Kanji Ratio | Kana-Only % |
|---|---|---|
| A. Baseline GPT-2 | 0.0244 | 67.7% |
| B. Fine-tuned only | 0.0029 | 96.8% |
| C. Fine-tuned + Logit Mask ★ | 0.0000 | **100.0%** |

- Fine-tuning alone reduced kanji ratio by **88%** vs baseline
- Combined method achieved **perfect 100% kana-only output**
- Training loss converged from **3.0 → 0.47** over 2 epochs with no overfitting (val loss tracked train loss closely throughout)

---

## Sample Outputs

| Input (Kanji) | Baseline | Fine-tuned | FT + Mask ★ |
|---|---|---|---|
| 東京は大きい都市です。 | English code output | とうきょうはまちです。 | とうきょうはおおきいです。 |
| 音楽を聞きます。 | Repeats prompt | おんがくをはいます。 | おんがくをきいます。 |
| 今日は晴れです。 | Random English | きょうははれです。 | きょうははれです。 |

---

## Model & Training Details

| Parameter | Value |
|---|---|
| Base model | GPT-2 Small (124M parameters) |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 0.1 |
| Epochs | 2 |
| Batch size | 8 |
| Training examples | 319 |
| JLPT level | N5/N4 |

---

## Novel Contribution

The `build_kanji_mask()` and `generate_kana()` functions extend the Ch. 7 generation loop with vocabulary-level output control. This logit masking approach is a general inference-time constraint applicable to any token-level generation task where you need hard guarantees on output vocabulary — not just Japanese text generation.

---

## Setup & Usage

This notebook was built in **Google Colab**. Cells 15–16 contain Colab-specific drive mounting code that can be skipped if running locally.

You will need `previous_chapters.py` and `gpt_download.py` from the original book repository:
```
https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/01_main-chapter-code
```

### Dependencies
```bash
pip install torch tiktoken tqdm matplotlib pykakasi
```

---

## Project Structure

```
kana-gpt2/
├── kana_gpt2_project_1.ipynb   # main notebook
└── README.md
```

---

## References

- Raschka, S. (2024). *Build a Large Language Model From Scratch*. Manning Publications.
- OpenAI GPT-2 weights via `gpt_download.py`
- pykakasi: https://github.com/miurahr/pykakasi
