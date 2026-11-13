# Comparing Fine-Tuning Methods for Instruction-Tuning Small Language Models

Code and results for the paper **"A Comparative Study of Fine-Tuning Methods for Instruction-Tuning Small Language Models"** 

We compare **full fine-tuning (FFT)** with four parameter-efficient methods: **LoRA, DoRA, QLoRA and Prompt Tuning**. The comparison covers three small language models and three instruction datasets (45 runs). For each run we record benchmark accuracy, training time, evaluation time and estimated energy.

---

## Experimental design

| Factor | Levels |
|---|---|
| **Models** | `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (base checkpoints, not Instruct) |
| **Datasets** | `databricks/databricks-dolly-15k`, `tatsu-lab/alpaca`, `OpenAssistant/oasst1` |
| **Methods** | FFT, LoRA, DoRA, QLoRA, Prompt Tuning |
| **Benchmarks** | MMLU, GPQA Diamond (0-shot and 5-shot), TruthfulQA MC1, MuSR (via `lm-eval-harness`) |
| **Runs** | 3 × 3 × 5 = 45, one run per configuration |

In the code, FFT is called `SFT` (method id `SFT`).

### Method settings

All methods use `SFTTrainer` (TRL) with the following shared settings:

- per-device batch size 2 and 2 gradient-accumulation steps (effective batch size 4)
- linear LR schedule with 100 warm-up steps
- `paged_adamw_8bit` optimizer
- `bfloat16` precision, `packing=False`
- loss computed on the full text (prompt + response)

| Method | Configuration | Epochs | Max length | LR |
|---|---|---|---|---|
| FFT (`SFT`) | all parameters, unquantized backbone | 1 | 512 | 2e-5 |
| LoRA | r=16, α=16, dropout 0.05, all 7 linear projections | 3 | 512 | 2e-4 |
| DoRA | as LoRA, dropout 0, no gradient checkpointing, fp32 weight-norm patch | 3 | 512 | 2e-4 |
| QLoRA | LoRA settings, NF4 4-bit backbone with double quantization | 3 | 512 | 2e-4 |
| Prompt Tuning | 10 virtual tokens, text init ("You are a helpful assistant."), NF4 backbone | 3 | 256 | 1e-2 |

Gradient checkpointing is on for every method except DoRA. Each dataset is split 80/20 with seed 42; only the training split is used.

---

## Repository structure

```
.
├── AIS_FineTuning_final3_doracheck1.ipynb   # main notebook: preprocessing, training, evaluation, logging
├── lm_eval_worker1.py                       # evaluation worker, run as a subprocess (one benchmark per call)
├── results.csv                              # raw results, one row per run
├── adapters/                                # saved adapters / FFT checkpoints (not committed, large)
└── README.md
```

> The notebook calls `lm_eval_worker1.py` by name, so keep that exact filename in the same folder as the notebook.

---

## Setup

Tested environment: Windows 11, one NVIDIA RTX 4000 Ada Generation (20 GB), Intel Xeon w7-2495X, Python 3.12.3.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.6 trl==1.13.0 peft==0.21.0 bitsandbytes==0.50.2 \
            lm-eval==0.4.13 codecarbon==3.2.2 datasets accelerate torchao langdetect
```

The Llama 3.2 models are gated, so you need a Hugging Face token with access to them. Set it as an environment variable instead of writing it in the notebook:

```bash
# Windows (PowerShell)
$env:HF_TOKEN="hf_..."
# Linux/macOS
export HF_TOKEN=hf_...
```

> **Never commit your token.** Leave the `os.environ["HF_TOKEN"] = ""` cell empty, or remove it.

---

## Running the experiments

1. Open `AIS_FineTuning_final3_doracheck1.ipynb`.
2. Run the setup cells: imports, the DoRA weight-norm patch, HF login, CSV header, dataset formatters, model loading, PEFT/SFT configs and evaluation helpers.
3. In the **experiment-grid cell**, uncomment the datasets, methods and models you want in `DATASETS`, `FINE_TUNING_METHODS` and `MODELS`.
4. Run the **main loop cell**. For each configuration it:
   - loads and formats the dataset and loads the model (4-bit for QLoRA and Prompt Tuning);
   - trains with `SFTTrainer` and saves the adapter or full model to `adapters/exp{ID}_it{N}_{MODEL}_{DATASET}_{METHOD}`;
   - runs each benchmark in a **separate subprocess** (`lm_eval_worker1.py`), so one failing task does not stop the run;
   - appends one row to `results.csv`.

CodeCarbon trackers run separately for training and evaluation.

### Evaluation details

- lm-eval-harness `simple_evaluate` uses the Hugging Face backend with `dtype=float16`, `max_length=1024` and `batch_size=auto:4`.
- FFT is evaluated from the saved checkpoint. PEFT methods are evaluated by attaching the saved adapter to the **unquantized** base model.
- Task IDs and metrics:

| Task ID | Metric |
|---|---|
| `gpqa_diamond_zeroshot` | `acc,none` |
| `gpqa_diamond_n_shot` (5-shot) | `acc,none` |
| `mmlu` (0-shot) | `acc,none` |
| `truthfulqa_mc1` | `acc,none` |
| `leaderboard_musr` | `acc_norm,none` |

- `tinyMMLU` is also requested but returned no scores and is not used.

---

## `results.csv` columns

| Column | Meaning |
|---|---|
| `EXPERIMENT_ID`, `ITERATION_NO` | run identifiers |
| `DATASET_ID`, `MODEL_ID`, `FINE_TUNING_METHOD_ID` | configuration |
| `FINE_TUNING_TIME_S`, `LM_EVAL_TIME_S` | wall-clock training / evaluation time (s) |
| `FINE_TUNING_ENERGY_J`, `LM_EVAL_ENERGY_J` | **see known issues below; do not use directly** |
| `LM_EVAL_*` | benchmark scores |
| `ENTRY_DATA`, `ENTRY_TIME_S` | number of training rows and setup time |

If a configuration appears more than once, the paper uses the **most recent row**.

---

## Known issues (please read before reusing the data)

1. **Energy columns are mislabeled.** `tracker.stop()` returns **kg CO₂e**, not kWh. The code multiplies this by 3.6×10⁶ and stores it as "Joules". To recover the values:
   - emissions: `kgCO2e = stored / 3.6e6`
   - energy: `kWh = kgCO2e / 0.330718` (CodeCarbon's grid intensity for Italy)

   Even after this correction, CodeCarbon 3.2.2 logged a near-constant **5.51 kW**. CodeCarbon 3.3.1 (NVML-based) measured **0.24 kW** on the same workstation, so the logged values are about 23× too high. The paper therefore uses **wall-clock time** as the main cost measure and estimates energy as `time × 0.24 kW`. To log energy correctly, use `tracker.final_emissions_data.energy_consumed` (kWh).

2. **Llama-3.2-3B evaluation failure.**
   - Every DoRA and Prompt Tuning run on Llama-3.2-3B returned degenerate scores. These scores match the harness always choosing the first answer option (e.g. TruthfulQA MC1 = 1.000).
   - The other Llama-3.2-3B scores are close to chance.
   - The suspected cause is overflow in `float16` evaluation (training used `bfloat16`).
   - All Llama-3.2-3B accuracies are excluded from the paper's analysis; their time costs are kept.
   - To re-evaluate, change `"dtype": "float16"` to `"bfloat16"` in `lm_eval_worker1.py`.

3. **Train/eval mismatch for QLoRA and Prompt Tuning.** Both are trained on a 4-bit NF4 backbone but evaluated on the 16-bit backbone.

4. **Truncation.** With `max_length=1024`, many MuSR and 5-shot GPQA prompts are truncated from the left.

5. **Single runs, unmatched schedules.** There is one run per configuration. Repeated runs differed by 0.5–2.2 MMLU points. Schedules differ by method (see the table above), so methods are compared as typically configured, not at equal compute.

6. **Peak GPU memory was not recorded.**

7. **Leftover code and debug cells.**
   - The notebook contains a `QAT_LORA` method (torchao) and a `google/gemma-2-9b` download cell; neither is used in the paper.
   - `run_exp` also runs a one-off debug forward/backward pass before training. It is not part of the measured results, but it falls inside the training timer.

---

## Main results (Llama-3.2-1B and DeepSeek-R1-Distill-Qwen-1.5B)

Mean accuracy over six model–dataset pairs; chance is 0.25 for MMLU/GPQA, 0.226 for TruthfulQA MC1 and 0.371 for MuSR.

| Method | MMLU | GPQA 0-shot | GPQA 5-shot | TruthfulQA | MuSR | Train h/run* | Eval h/run* |
|---|---|---|---|---|---|---|---|
| FFT | 0.380 | 0.247 | 0.249 | 0.264 | 0.355 | 2.46 | 5.42 |
| LoRA | 0.369 | 0.247 | 0.250 | 0.261 | 0.366 | 4.68 | 6.51 |
| DoRA | 0.366 | 0.244 | 0.263 | 0.253 | 0.364 | 18.03 | 8.71 |
| QLoRA | 0.384 | 0.247 | 0.240 | 0.261 | 0.365 | 6.19 | 7.27 |
| Prompt Tuning | 0.361 | 0.253 | 0.247 | 0.272 | 0.353 | 2.73 | 4.20 |

\*Times are averaged over all nine model–dataset combinations, including Llama-3.2-3B.

Key findings:
- Only MMLU separates the methods. The other benchmarks stay near chance.
- Differences between methods are at most 2.3 points, which is comparable to run-to-run variation.
- Only FFT beat LoRA in all six paired comparisons (sign test p = 0.031).
- One epoch of FFT was the cheapest run. DoRA took 3.9× as long as LoRA with no accuracy gain.
- Evaluation took as long as training: 289 h vs 307 h in total.
- The base model mattered more than the fine-tuning method.

---

## Acknowledgements

Built with [PyTorch](https://pytorch.org), [Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [CodeCarbon](https://github.com/mlco2/codecarbon).
