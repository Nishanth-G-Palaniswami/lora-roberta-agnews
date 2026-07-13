# Parameter-Efficient Text Classification — RoBERTa + LoRA on AG News

Fine-tunes **RoBERTa-base** for 4-way news topic classification (AG News) using **LoRA** (Low-Rank Adaptation), training a tiny fraction of the model's weights.

## Result

| Metric | Value |
|---|---|
| Test accuracy | **94.8%** (0.94802 on the 7,600-example AG News test split) |
| Trainable parameters | **999,172 of 125,647,880 — 0.80%** |
| Base model | `roberta-base` |
| Method | LoRA adapters (PEFT) on the attention projections |
| Dataset | AG News (120,000 train / 7,600 test, 4 classes) |

The headline: ~95% accuracy while updating **less than 1%** of the parameters — the rest of RoBERTa stays frozen, so the trained artifact is a few MB of adapter weights instead of a 500 MB full checkpoint.

## Why this matters

Full fine-tuning of a 125M-parameter model updates every weight and produces a full-size checkpoint per task. LoRA freezes the pretrained weights and injects small trainable low-rank matrices into the attention layers, so:
- **Storage:** adapters are ~4 MB vs a ~500 MB full checkpoint.
- **Compute/memory:** far fewer gradients and optimizer states — trains on a single modest GPU.
- **Accuracy:** within a point of full fine-tuning on this task.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook roberta_lora_agnews.ipynb   # run top to bottom
```

The notebook downloads `roberta-base` and AG News from Hugging Face, builds the LoRA/PEFT config, trains, and evaluates. Experiment tracking (Weights & Biases) is **optional** — set `WANDB_API_KEY` in your environment to enable it, or skip it entirely.

## Stack

`transformers` · `peft` · `datasets` · `evaluate` · `torch` · `accelerate`

## Credit

NYU Deep Learning (Spring 2025) course project — group work. This repository is the LoRA text-classification component. Outputs were cleared before publishing; re-run the notebook to reproduce the numbers above.

## License

MIT — see [LICENSE](LICENSE).
