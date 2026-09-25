# Model shortlist (licences and parameter counts verified on the Hugging Face API, 2026-09-25)

Rule: each model **and its base model** must be MIT or Apache-2.0, ≤ 8B **total** parameters, offline, fine-tuned only on provided data. Several models are allowed. There is **no time limit**, so compute-for-accuracy trades (ensembles, recursion, test-time augmentation, reasoning) are fair game; the practical budget is GPU hours (Kaggle: 30 h/week per account, T4 × 2).

## Pair scorers (cross-encoders)
| Model | Params | Licence (base) | Why | Status |
|---|---|---|---|---|
| BAAI/bge-reranker-v2-m3 | 0.57B | Apache-2.0 (XLM-R, MIT) | multilingual incl. Indian scripts | zero-shot 0.992 → **fine-tuned 0.9997**, 250 pairs/s |
| Qwen/Qwen3-Reranker-0.6B | 0.60B | Apache-2.0 | yes/no reranker, LoRA | `er-xenc-v1b` |
| Qwen/Qwen3-Reranker-4B | 4.02B | Apache-2.0 | size test (LoRA, fp16 base) | zero-shot 0.9907 → **0.99954 after 18.6k pairs**; 11 pairs/s (uncertain band only) |
| google/byt5-base | ~0.58B | Apache-2.0 | byte-level | **dropped**: NaN logits in fp16 (T5 family overflow; T4 has no bf16); 39 pairs/s |
| Alibaba-NLP/gte-multilingual-reranker-base | 0.31B | Apache-2.0 | small + fast | remote-code CUDA assert → replaced by CANINE |
| mixedbread-ai/mxbai-rerank-large-v2 | 1.54B | Apache-2.0 (Qwen2.5-1.5B, Apache) | strong reranker, different training | next |
| microsoft/mdeberta-v3-base | ~0.28B | MIT | strong NLU encoder, diversity | next |
| google/canine-c | 0.13B | Apache-2.0 | character-level, very cheap | **AUC 0.9992** after 1 epoch (300k pairs, 29 min), 485 pairs/s → v4 with 3x data |
| google/muril-base-cased | ~0.24B | Apache-2.0 | Indian languages + transliteration | next |
| FacebookAI/xlm-roberta-large | 0.56B | MIT | plain multilingual baseline | optional |

## Embedders (bi-encoders: blocking arm, similarity features)
Qwen3-Embedding-0.6B / 4B / 8B (0.60 / 4.02 / 7.57B, Apache) · bge-m3 (MIT) · multilingual-e5-large-instruct (0.56B, MIT) · snowflake-arctic-embed-l-v2.0 (0.57B, Apache) · granite-embedding-278m-multilingual (Apache) · nomic-embed-text-v2-moe (0.48B, Apache).

## Qwen3.5 series (checked 2026-09-25)
Hybrid recurrent architecture: 3 of every 4 layers are **Gated DeltaNet linear attention** (a recurrent state), every 4th is full attention; vision-language models (`Qwen3_5ForConditionalGeneration`, vocab 248k), used text-only. Apache-2.0.
| Model | Params | In limit | Plan |
|---|---|---|---|
| Qwen/Qwen3.5-0.8B | 0.87B | yes | yes/no scorer + LoRA (`er-xenc-v3`) |
| Qwen/Qwen3.5-2B | 2.27B | yes | yes/no scorer + LoRA, fp16 base + checkpointing (`er-xenc-v3`) |
| Qwen/Qwen3.5-4B | 4.66B | yes | next if 2B beats 0.8B |
| Qwen/Qwen3.5-9B | 9.65B | **no** | excluded |
| Qwen3.5-27B / 35B-A3B / 122B, Qwen3.6-27B / 35B, Qwen3-Next-80B | 27–125B | no | excluded |

## Generative judges (listwise over an S1's candidates, uncertain band first)
| Model | Params | Licence (base) | Note |
|---|---|---|---|
| Qwen/Qwen3-4B | 4.02B | Apache-2.0 | thinking mode = test-time reasoning |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7.62B | MIT (Qwen2.5-Math-7B, Apache) | reasoning distill |
| Qwen/Qwen2.5-7B-Instruct | 7.62B | Apache-2.0 | |
| mistralai/Mistral-7B-Instruct-v0.3 | 7.25B | Apache-2.0 | |
| allenai/OLMo-2-1124-7B-Instruct | 7.30B | Apache-2.0 | fully open data |
| microsoft/Phi-4-mini-instruct / Phi-4-mini-flash-reasoning | 3.84B / 3.85B | MIT | flash-reasoning uses a hybrid recurrent decoder |
| HuggingFaceTB/SmolLM3-3B | 3.08B | Apache-2.0 | |
| ibm-granite/granite-3.3-2b-instruct | 2.53B | Apache-2.0 | |

## Recurrent and recursive models (fixed parameters, more compute)
| Model / method | Params | Licence | Idea for us |
|---|---|---|---|
| tomg-group-umd/huginn-0125 | 3.91B | Apache-2.0 | **recurrent depth**: iterates a shared block more times at test time → more reasoning, same parameters |
| RWKV/RWKV7-Goose-World3-2.9B-HF | 2.95B | Apache-2.0 | RNN, linear time, constant memory: listwise judging of long candidate lists |
| Zyphra/Zamba2-2.7B-instruct | 2.66B | Apache-2.0 | hybrid Mamba-2 |
| ibm-granite/granite-4.0-h-micro / h-tiny | 3.19B / 6.94B | Apache-2.0 | hybrid Mamba-2 / transformer |
| state-spaces/mamba2-2.7b | ~2.7B | Apache-2.0 | pure state-space model |
| Tiny Recursive Model (TRM, Samsung SAIL Montréal, 2025) | ~7M | MIT (code) | train from scratch on our data: recursively refine an S1's match set from its candidates' features (set-level decision) |
| **Iterative collective classification** (ours) | — | — | round k uses round k-1 predictions of neighbours (other S1s claiming the record, sibling group, S1 match count) as features; 2–3 rounds |

## Excluded (and why)
Qwen3-8B, Qwen3-Reranker-8B (8.19B) and granite-3.3-8b (8.17B): **over 8B**. bge-reranker-v2-gemma (Gemma base), Gemma 3, t5gemma: Gemma terms. Llama 3.x: Llama licence. jina rerankers: CC-BY-NC. Qwen2.5-3B: Qwen research licence. Falcon3, Falcon-H1, falcon-mamba, LFM2, Nemotron-H: non-Apache licences. granite-4.0-h-small: 32B. MuRIL-large, HRM checkpoints, mamba-2.8b-hf: licence not stated on the card (verify before use).
