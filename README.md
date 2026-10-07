
# UR2PhD Starter Assignment: SemEval-2027 Task 2 (DiCo-NLI), Track 1

Given an ordered pair of short phrases (text1, text2), a system predicts one of four labels:

| Label | Meaning |
|---|---|
| EQUIVALENCE | text1 and text2 mean the same thing |
| FORWARD_ENTAILMENT | text1 entails text2 |
| BACKWARD_ENTAILMENT | text2 entails text1 |
| NEGATIVE_OTHER | any other relation |

Most pairs also appear in reversed order, and a good system should give compatible answers in both directions. The official scorer reports three metrics:

- **Weighted F1**: label accuracy across all dev items.
- **SoftCons**: share of reversed pairs whose two predictions are compatible with each other (FORWARD with BACKWARD, EQUIVALENCE with EQUIVALENCE), whether or not they are correct.
- **HardCons**: share of reversed pairs where both directions are predicted correctly.

I built two systems: a fine-tuned encoder (System A, DeBERTa-v3-base) and a prompted LLM (System B, Qwen2.5-1.5B-Instruct). All results are on the official dev split (660 items) and come from the official scorer. 

## Repository structure

```
README.md
part1_data_exploration.ipynb        Part 0 (majority baseline) and Part 1 (data exploration)
part2/
  part2_deberta_seed13.ipynb        System A, seed 13
  part2_deberta_seed2026.ipynb      System A, seed 2026 (identical except SEED)
  part2_summary.ipynb               mean and spread of System A over seeds
Part3_llm.ipynb                     System B, zero-shot and few-shot
results-part0/
  majority_dev.csv                  majority baseline predictions
  scores.json, scores.txt           official scorer output
results-part2/
  seed13/                           predictions_deberta_seed13.csv, scores.json, scores.txt, train_loss_seed13.csv
  seed2026/                         predictions_deberta_seed2026.csv, scores.json, scores.txt, train_loss_seed2026.csv
results-part3/
  zeroshot/                         predictions_llm_zeroshot.csv, scores.json, scores.txt, llm_zeroshot_raw.csv
  fewshot/                          predictions_llm_fewshot.csv, scores.json, scores.txt, llm_fewshot_raw.csv
report.pdf                          2-page report (TODO)
```

Every prediction file uses the required format `instance_id,label` and has 660 rows. `scores.json` also contains the confusion matrix and the per-pair consistency results. The `*_raw.csv` files keep the LLM's raw text output next to the parsed label.

## Environment

- Google Colab, T4 GPU
- Python packages: `transformers`, `datasets`, `accelerate`, `sentencepiece`, `scikit-learn`, `pandas`, `torch` (installed in the first cell of each notebook)
- Task data, scorer and starter kit from the official repository: https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI

## How to reproduce

Each notebook is self-contained. It installs the packages, clones the task repository into `/content/`, and writes its outputs to `/content/SemEval-2027-Task-2-DiCo-NLI/results/`.

1. Open the notebook in Google Colab and select **Runtime > Change runtime type > T4 GPU** (Part 1 also runs on CPU).
2. Run **Runtime > Restart session and run all**.
3. Download the `results/` folder when the notebook finishes.

Run order:

1. `part1_data_exploration.ipynb`
2. `part2/part2_deberta_seed13.ipynb` (about 3 minutes of training)
3. `part2/part2_deberta_seed2026.ipynb`
4. `part2/part2_summary.ipynb` (reads `results-part2/seed*/scores.json` from this repository)
5. `Part3_llm.ipynb` (two full passes over dev, a few minutes each)

Every system was scored with the official scorer, run from inside the task repository:

```
python3 -m evaluation_functions \
  --gold final_data/dev/dico_nli_dev_track1_reference.csv \
  --predictions results/<predictions_file>.csv \
  --output-dir results/<system_name>
```

GPU training is not perfectly deterministic, so rerunning Part 2 may change the last decimals of the System A scores. The LLM uses greedy decoding, so Part 3 should reproduce exactly.

## Data

| File | Rows | Use |
|---|---|---|
| `final_data/train/dico_nli_train_track1_participant_labeled.csv` | 3042 | training (System A), few-shot examples (System B) |
| `final_data/dev/dico_nli_dev_track1_reference.csv` | 660 | prediction and scoring only |

Dev was never used for training, for choosing settings, or for choosing prompts or few-shot examples.

## Part 0: majority baseline

Always predicts the most frequent train label. FORWARD_ENTAILMENT and BACKWARD_ENTAILMENT are tied at 881 train rows, and pandas `mode()` returns BACKWARD_ENTAILMENT first (alphabetical order). The baseline confirms the scorer works and gives a floor for the other systems. Its SoftCons is 0 because it predicts BACKWARD_ENTAILMENT in both directions of every pair, which is never a compatible combination.

## Part 1: data exploration (summary)

Label distribution:

| Label | Train | Train % | Dev | Dev % |
|---|---|---|---|---|
| FORWARD_ENTAILMENT | 881 | 29.0 | 190 | 28.8 |
| BACKWARD_ENTAILMENT | 881 | 29.0 | 190 | 28.8 |
| EQUIVALENCE | 802 | 26.4 | 174 | 26.4 |
| NEGATIVE_OTHER | 478 | 15.7 | 106 | 16.1 |

How reversed pairs are linked:

- Both rows of a reversed pair share the same `pair_id`. Their `instance_id`s end in `__original` and `__flipped`.
- In the reference file, each row's `reverse_pair_id` is the `instance_id` of its twin, and the link goes both ways.
- Dev has 277 reversed pairs (554 rows) and 106 rows with no twin. All 106 unpaired rows are NEGATIVE_OTHER, so NEGATIVE_OTHER items count toward weighted F1 but never toward SoftCons or HardCons.
- FORWARD_ENTAILMENT always reverses to BACKWARD_ENTAILMENT, and EQUIVALENCE always stays EQUIVALENCE.

Three hand-picked examples per label, with explanations, are in `part1_data_exploration.ipynb`.

## Part 2: System A, fine-tuned DeBERTa-v3-base

A cross-encoder: both phrases go into the model as one sequence, `[CLS] text1 [SEP] text2 [SEP]`, so the comparison happens inside the model and the order of the phrases is kept. A new 4-way classification head is trained on top, and the pretrained encoder is fine-tuned along with it, using the Hugging Face `Trainer`.

| Setting | Value |
|---|---|
| Model | `microsoft/deberta-v3-base` |
| Label ids | FORWARD_ENTAILMENT=0, BACKWARD_ENTAILMENT=1, EQUIVALENCE=2, NEGATIVE_OTHER=3 |
| Learning rate | 2e-5 |
| Batch size | 16, dynamic padding per batch (`DataCollatorWithPadding`) |
| Epochs | 5 (955 optimizer steps) |
| Weight decay | 0.01 |
| Warmup | none |
| Precision | fp16 mixed precision, model weights loaded in float32 |
| Truncation | none (all phrases are short) |
| Checkpoint | final model after epoch 5, no evaluation during training |
| Seeds | 13 and 2026 |

The seed is the only thing that changes between the two runs. It controls the classification head initialization, the shuffling of training data and dropout. The seed values are arbitrary and were fixed before training. Each seed has its own notebook, so every run starts from a freshly loaded model.

**Number of epochs:** Decided on 5 after trying 3 and 10; at 3 the loss was too high, and at 10 the loss was too low. It was fixed before any dev predictions were made and not changed afterwards.

**Sanity checks:** in both runs the training loss starts near 1.35, close to chance for 4 classes (ln 4 ≈ 1.39), and ends below 0.09 (`train_loss_seed*.csv`). Both runs predict all four labels on dev.

## Part 3: System B, prompted LLM

The LLM is not trained: its weights stay frozen, and only the prompt changes between the two variants.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (small, runs on a T4, no access approval needed) |
| Precision | fp16 |
| Input format | Qwen chat template (`apply_chat_template`): system message with the instructions, user message `text1: ...` / `text2: ...` / `Label:` |
| Decoding | greedy (`do_sample=False`), `max_new_tokens=10` |
| Calls | one generation per dev item, 660 per variant |

**Zero-shot prompt (system message):** defines the four labels and asks for the label only.

```
You classify the relation between two short English phrases, text1 and text2.
Choose exactly one label:
EQUIVALENCE: text1 and text2 mean the same thing.
FORWARD_ENTAILMENT: if text1 is true, text2 must also be true, but not the other way around.
BACKWARD_ENTAILMENT: if text2 is true, text1 must also be true, but not the other way around.
NEGATIVE_OTHER: none of the above.
Reply with the label only.
```

**Few-shot prompt:** the same system message followed by 8 labelled train examples, 2 per label, sampled with `train.groupby("label").sample(2, random_state=0)`. The examples were fixed before running on dev and were not changed after seeing scores. The sample happens to contain two reversed twins ("A boy" / "A young boy" and a bikini example appear as both FORWARD and BACKWARD).

**Parser:** the raw output is uppercased, runs of spaces and hyphens are replaced by `_`, and a regex searches for the first of the four label names. If no label is found, the item gets the fallback label NEGATIVE_OTHER and is counted as unparsed.

**Unparsed outputs:** 0 of 660 for zero-shot and 0 of 660 for few-shot, so the fallback was never used.

## Results (dev, official scorer)

| System | Weighted F1 | SoftCons | HardCons |
|---|---|---|---|
| Majority baseline | 0.129 | 0.000 | 0.000 |
| DeBERTa-v3-base, seed 13 | 0.823 | 0.884 | 0.834 |
| DeBERTa-v3-base, seed 2026 | 0.815 | 0.895 | 0.834 |
| **DeBERTa-v3-base, mean ± std (2 seeds)** | **0.819 ± 0.005** | **0.890 ± 0.008** | **0.834 ± 0.000** |
| Qwen2.5-1.5B-Instruct, zero-shot | 0.234 | 0.260 | 0.123 |
| Qwen2.5-1.5B-Instruct, few-shot (8 examples) | 0.263 | 0.343 | 0.148 |

Spread is the sample standard deviation over the 2 seeds. SoftCons and HardCons are computed over the 277 reversible pairs.

## Runs that did not work

- **Early runs trained on dev by mistake.** While preparing dev, the tokenized dev set was assigned to the variable `tokenized_train`, so a few exploratory runs (3 and 10 epochs, 126 and 420 steps) trained on the 660 dev rows instead of the 3042 train rows. They were discarded, and nothing from them is reported.
- **fp16 error.** Training failed with `Attempting to unscale FP16 gradients` while the weights were in 16-bit. Loading the weights in float32 and keeping `fp16=True` fixed it.
- **Notebook state.** One seed 2026 run kept training a model that had already been trained (its loss started at 0.15 instead of about 1.35), and its predictions came from the seed 13 trainer. It was discarded. The reported runs were made with "Restart session and run all", one notebook per seed.

## Use of AI assistants

> I used Claude (Anthropic) as an assistant during this assignment. I used it to explain concepts (encoders vs decoders, cross-encoder input, tokenization, seeds, the Trainer API, prompting and parsing LLM output), to review code I wrote and point out bugs (for example the overwritten `tokenized_train` variable, out-of-order notebook cells, and passing whole columns instead of single rows to the LLM), and to debug errors such as the fp16 error.
> It also provided some code (the majority baseline script, the prediction and saving cells, the LLM generation function, the label parser and the few-shot example selection) and with a draft of this README. I ran all experiments myself, checked the outputs, and can explain every line of the code.
