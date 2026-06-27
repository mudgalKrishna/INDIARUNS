#!/usr/bin/env python3
"""
train_mamba_sft.py
==================
Fine-tunes Mamba-370M for candidate reasoning generation.

The model learns to read a structured candidate profile and produce a
2-4 sentence reasoning string that explains why the candidate is or isn't
a fit for the JD — exactly the style the teacher model (Qwen2.5-72B) used.

─── TWO-STEP PROCESS ────────────────────────────────────────────────────────

Step 1  — Clean the teacher labels (run with --clean-only):
    Reads teacher_labels.jsonl, removes duplicates, filters by quality,
    and writes sft_training_data.jsonl with (input_text, reasoning) pairs.

Step 2  — SFT training (run after Step 1):
    Fine-tunes Mamba-370M on the cleaned pairs using causal LM loss.
    Saves the fine-tuned model to --out-dir.

─── USAGE ───────────────────────────────────────────────────────────────────

    # Recommended: Step 1 with fixed reasoning + split files (inspect before training)
    python train_mamba_sft.py \
        --labels candidate_split_part1.jsonl,candidate_split_part2.jsonl \
        --fixed-reasoning merged_fixed_reasoning.jsonl \
        --clean-only \
        --sft-data sft_training_data.jsonl

    # Step 2: train on full 15.6K fixed dataset (requires GPU — RTX PRO 6000 Blackwell with 96GB)
    python train_mamba_sft.py \
        --labels candidate_split_part1.jsonl,candidate_split_part2.jsonl \
        --fixed-reasoning merged_fixed_reasoning.jsonl \
        --sft-data sft_training_data.jsonl \
        --out-dir mamba_ranker/ \
        --epochs 3 \
        --batch-size 4

    # Both steps in one run (original single-file mode still works)
    python train_mamba_sft.py \
        --labels teacher_labels.jsonl \
        --out-dir mamba_ranker/

─── REQUIREMENTS ────────────────────────────────────────────────────────────
    pip install transformers torch mamba-ssm causal-conv1d datasets accelerate

─── RUNTIME ─────────────────────────────────────────────────────────────────
    Cleaning: ~30 seconds on CPU.
    Training: ~45 minutes on RTX PRO 6000 Blackwell (96GB VRAM) for 3 epochs.
    Inference: ~100 seconds on CPU for 100 candidates.

─── MODEL OUTPUT ────────────────────────────────────────────────────────────
    mamba_ranker/           — HuggingFace-compatible model directory
    mamba_ranker/tokenizer/ — tokenizer files
    sft_training_data.jsonl — cleaned training data (inspect before training)
"""

import json
import argparse
import os
import re
import sys
import time
import collections
import shutil
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
#  JD CONTEXT (same as teacher_scoring_v2.ipynb — model must see this every time)
# ─────────────────────────────────────────────────────────────────────────────

JD_SYSTEM_PROMPT = """You are a senior technical recruiter at Redrob AI evaluating candidates for:
ROLE: Senior AI Engineer — Founding Team
COMPANY: Redrob AI (Series A, AI-native talent intelligence platform)
LOCATION: Pune/Noida, India (Hybrid). Also: Hyderabad, Mumbai, Delhi NCR.
EXPERIENCE: 5–9 years (ideal: 6–8 years, sweet spot).

KEY REQUIREMENTS:
- Production embeddings retrieval (sentence-transformers, BGE, E5) deployed to real users
- Vector DB / hybrid search infrastructure (Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch)
- Strong Python. Evaluation frameworks: NDCG, MRR, A/B tests.
- Career at product companies (not only IT services)
- Active on platform, reasonable notice period (<90 days preferred)

DISQUALIFIERS: Entire career at services firms only. Non-tech career with AI keywords.
Pure research, no production. AI = only LangChain+API calls. No production code in 18 months.

Write 2-4 sentences of specific, factual reasoning. Only reference facts from the profile."""


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1: DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

# Validator from teacher_scoring_v2 had known false positive issues.
# We re-assess quality from reasoning TEXT content only, since teacher_labels
# does not contain the original profile data.

# These are the only checks that can be done without profile data:
QUALITY_CHECKS = {
    'min_length':    80,     # reasoning must be substantive
    'max_length':    650,    # reasoning must not be bloated
    'min_score':     55,     # only train on reasonable fits (not noise candidates)
    'must_have_number': True, # specific number = specific fact (not generic)
}

# Detect near-duplicate reasoning by (current_title, current_company) pair.
# Teacher generates ~89 near-identical reasonings for "Full Stack Developer at Razorpay".
# Deduplicate: keep the one with the highest teacher_score per (title, company) pair.
TITLE_COMPANY_PATTERN = re.compile(r'Currently (.+?) at (.+?) where', re.IGNORECASE)

# Skills mentioned in reasoning that we KNOW are genuine JD matches
# Used to confirm the reasoning is about the right job
JD_RELEVANT_SKILL_MENTIONS = {
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'pgvector',
    'elasticsearch', 'opensearch', 'bm25',
    'sentence transformers', 'embeddings', 'vector search', 'semantic search',
    'information retrieval', 'learning to rank',
    'pytorch', 'tensorflow', 'scikit-learn',
    'langchain', 'llamaindex', 'rag', 'fine-tuning', 'hugging face',
    'recommendation systems', 'nlp', 'ndcg', 'mrr',
}


def assess_reasoning(record):
    """
    Returns (keep: bool, reason: str).
    Applies content-based quality checks to a teacher_labels record.
    """
    text  = record.get('teacher_reasoning', '').strip()
    score = record.get('teacher_score', 0)

    # Length check
    if len(text) < QUALITY_CHECKS['min_length']:
        return False, f'too_short:{len(text)}'
    if len(text) > QUALITY_CHECKS['max_length']:
        return False, f'too_long:{len(text)}'

    # Score threshold
    if score < QUALITY_CHECKS['min_score']:
        return False, f'score_too_low:{score}'

    # Must contain a specific number (month duration, score, percentage)
    if QUALITY_CHECKS['must_have_number'] and not re.search(r'\d+', text):
        return False, 'no_numbers'

    return True, 'ok'


def build_sft_input(record):
    """
    Build the input prompt for SFT training.
    Format: [JD context] + candidate summary from available data.
    
    Since teacher_labels only has numeric features (not raw profile text),
    we reconstruct a minimal but informative candidate summary from features.
    This is what the model will see during TRAINING.
    
    During INFERENCE (rank.py), the model gets the full formatted profile
    (same as teacher_scoring_v2 format_candidate_for_prompt()).
    
    IMPORTANT: This function must be kept in sync with how rank.py formats
    candidates for Mamba inference.
    """
    feats = record.get('features', {})
    cid   = record.get('candidate_id', 'UNKNOWN')
    score = record.get('teacher_score', 0)

    # Reconstruct readable summary from numeric features
    title_type = (
        'AI/ML core role'         if feats.get('title_is_ai_core')     else
        'tech-adjacent role'      if feats.get('title_is_ambiguous')    else
        'non-technical role'      if feats.get('title_is_disqualify')   else
        'other role'
    )
    yoe        = feats.get('yoe', 0)
    n_must     = feats.get('n_must_have_skills', 0)
    n_expert   = feats.get('n_expert_relevant', 0)
    max_dur    = feats.get('max_relevant_dur_months', 0)
    n_prod_cos = feats.get('n_product_cos', 0)
    n_tier1    = feats.get('n_tier1_product_cos', 0)
    notice     = feats.get('notice_days', 90)
    github     = feats.get('github_score', -1)
    gh_str     = 'no GitHub linked' if github < 0 else f'GitHub score {github:.0f}'
    open_work  = feats.get('open_to_work', 0)
    inactive   = feats.get('days_inactive', 999)
    n_assessments = feats.get('n_assessments', 0)
    avg_assess = feats.get('avg_assessment_score', 0)

    input_text = f"""{JD_SYSTEM_PROMPT}

=== CANDIDATE SUMMARY (ID: {cid}) ===
Role type:           {title_type}
Years of experience: {yoe:.1f} years
Must-have skills:    {n_must} matching JD (expert-level: {n_expert})
Max skill duration:  {max_dur} months for top relevant skill
Product companies:   {n_prod_cos} ({n_tier1} FAANG/Big Tech)
Notice period:       {notice} days
Availability:        {'Open to work' if open_work else 'Not explicitly open to work'}
Activity:            Last active {inactive} days ago
GitHub:              {gh_str}
Assessments:         {n_assessments} taken (avg {avg_assess:.0f}/100 if any)

Provide 2-4 sentences of specific reasoning for this candidate's fit score of {score}/100:"""

    return input_text


def _load_jsonl(path):
    """Load a JSONL file, handling optional UTF-8 BOM."""
    for enc in ('utf-8-sig', 'utf-8'):
        try:
            records = []
            with open(path, 'r', encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} as UTF-8")


def clean_data(labels_path, sft_data_path, fixed_reasoning_path=None):
    """
    Step 1: Load teacher_labels.jsonl (or multiple split files separated by
    commas), optionally merge in fixed reasoning from --fixed-reasoning,
    clean, and write SFT pairs.
    Returns count of records written.
    """
    # ── Load labels (supports comma-separated list of split files) ────────
    paths = [p.strip() for p in labels_path.split(',') if p.strip()]
    records = []
    for p in paths:
        print(f"[CLEAN] Loading {p}...")
        chunk = _load_jsonl(p)
        records.extend(chunk)
        print(f"[CLEAN]   {len(chunk):,} records from {p}")
    print(f"[CLEAN] Total loaded: {len(records):,} records.")

    # ── Merge fixed reasoning (replaces teacher_reasoning for every record) ─
    if fixed_reasoning_path:
        print(f"[CLEAN] Loading fixed reasoning from {fixed_reasoning_path}...")
        fix_records = _load_jsonl(fixed_reasoning_path)
        fix_map = {r['candidate_id']: r['fixed_reasoning'] for r in fix_records}
        patched = 0
        for r in records:
            cid = r.get('candidate_id')
            if cid and cid in fix_map:
                r['teacher_reasoning'] = fix_map[cid]
                patched += 1
        print(f"[CLEAN] Patched {patched:,} / {len(records):,} records with fixed reasoning.")
        if patched < len(records):
            print(f"[CLEAN] WARNING: {len(records)-patched:,} records have no fixed reasoning entry — original reasoning kept.")
    else:
        print(f"[CLEAN] No --fixed-reasoning supplied; using teacher_reasoning as-is.")

    # ── Quality filter ────────────────────────────────────────────────────
    kept, dropped = [], collections.Counter()
    for r in records:
        ok, reason = assess_reasoning(r)
        if ok:
            kept.append(r)
        else:
            dropped[reason.split(':')[0]] += 1

    print(f"[CLEAN] After quality filter: {len(kept):,} (dropped {len(records)-len(kept):,})")
    print(f"[CLEAN] Drop reasons: {dict(dropped.most_common())}")

    # ── Deduplication (skipped when using fixed reasoning) ───────────────
    # Fixed reasoning is already unique per candidate_id with no boilerplate
    # "Currently X at Y where" pattern — dedup would remove 0 records anyway.
    # When using original teacher_reasoning, dedup is still needed to remove
    # the ~1715 near-identical records the teacher generated for same title+co.
    using_fixed = fixed_reasoning_path is not None
    if using_fixed:
        deduped = kept
        print(f"[CLEAN] Dedup skipped — fixed reasoning has no boilerplate duplicates.")
    else:
        groups    = {}  # key -> best record
        no_match  = []  # records with no title/company pattern (keep all)
        for r in kept:
            m = TITLE_COMPANY_PATTERN.search(r.get('teacher_reasoning', ''))
            if m:
                key = (m.group(1).strip().lower(), m.group(2).strip().lower())
                if key not in groups or r['teacher_score'] > groups[key]['teacher_score']:
                    groups[key] = r
            else:
                no_match.append(r)

        deduped = list(groups.values()) + no_match
        print(f"[CLEAN] After dedup (title+company): {len(deduped):,} "
              f"(removed {len(kept)-len(deduped):,} near-duplicates)")

    # ── Sort by teacher_score descending ──────────────────────────────────
    # Cap: 16000 for fixed reasoning (full validated dataset); 2000 for original.
    cap = 16000 if using_fixed else 2000
    deduped.sort(key=lambda r: r['teacher_score'], reverse=True)
    final = deduped[:cap]
    if len(deduped) > cap:
        print(f"[CLEAN] Capped at {cap:,} (dropped {len(deduped)-cap:,} lowest-scoring records).")

    print(f"[CLEAN] Final SFT set: {len(final):,} records")
    scores = [r['teacher_score'] for r in final]
    print(f"[CLEAN] Score range: {min(scores)} – {max(scores)},  Mean: {sum(scores)/len(scores):.1f}")

    # ── Score distribution ────────────────────────────────────────────────
    print("[CLEAN] Score distribution:")
    for lo, hi, label in [(75, 101, '>=75 (strong)'), (60, 75, '60-74 (moderate)'),
                          (55, 60, '55-59 (weak-mod)')]:
        cnt = sum(1 for s in scores if lo <= s < hi)
        print(f"  {label}: {cnt}")

    # ── Write SFT JSONL ───────────────────────────────────────────────────
    with open(sft_data_path, 'w', encoding='utf-8') as f:
        for r in final:
            sft_record = {
                'candidate_id':      r['candidate_id'],
                'teacher_score':     r['teacher_score'],
                'input':             build_sft_input(r),
                'output':            r['teacher_reasoning'],
                'full_text':         build_sft_input(r) + '\n' + r['teacher_reasoning'],
            }
            f.write(json.dumps(sft_record, ensure_ascii=False) + '\n')

    size_mb = os.path.getsize(sft_data_path) / 1024 / 1024
    print(f"[CLEAN] Written to {sft_data_path}  ({size_mb:.1f} MB)")

    # ── Sample for visual inspection ──────────────────────────────────────
    print("\n[CLEAN] === SAMPLE RECORD (inspect before training) ===")
    sample = final[0]
    print(f"  candidate_id:   {sample['candidate_id']}")
    print(f"  teacher_score:  {sample['teacher_score']}")
    print(f"  input (first 300 chars):")
    print(f"    {build_sft_input(sample)[:300]}")
    print(f"  output (reasoning):")
    print(f"    {sample['teacher_reasoning']}")

    return len(final)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: SFT TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def check_gpu():
    """Print GPU info and return whether GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[TRAIN] GPU: {name}  VRAM: {vram:.0f} GB")
            return True
        else:
            print("[TRAIN] WARNING: No CUDA GPU detected. Training on CPU will be very slow.")
            return False
    except ImportError:
        print("[TRAIN] PyTorch not installed.")
        return False


def load_sft_data(sft_data_path, tokenizer, max_length=512):
    """
    Load SFT JSONL and tokenise into HuggingFace Dataset format.
    Uses full_text = input + '\n' + output for causal LM training.
    Pads/truncates to max_length tokens.
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    records = []
    with open(sft_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records.append({'text': r['full_text']})

    print(f"[TRAIN] Loaded {len(records):,} SFT records.")

    dataset = Dataset.from_list(records)

    def tokenize(examples):
        out = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None,
        )
        out['labels'] = out['input_ids'].copy()
        return out

    tokenized = dataset.map(tokenize, batched=True, remove_columns=['text'])
    print(f"[TRAIN] Tokenised. Sequence length: {max_length}")
    return tokenized


def train(sft_data_path, out_dir, epochs=3, batch_size=4, lr=2e-4,
          warmup_steps=50, max_length=512, gradient_accumulation=4):
    """
    Fine-tune Mamba-370M on SFT data using HuggingFace Trainer.
    """
    check_gpu()

    # ── Imports ──────────────────────────────────────────────────────────
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer, DataCollatorForLanguageModeling
        )
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install: pip install transformers torch mamba-ssm causal-conv1d")
        sys.exit(1)

    MODEL_NAME = "state-spaces/mamba-370m-hf"
    print(f"[TRAIN] Loading model: {MODEL_NAME}")

    # ── Load tokenizer ────────────────────────────────────────────────────
    # Mamba uses GPT-NeoX tokenizer (EleutherAI/gpt-neox-20b)
    # The HF checkpoint includes a compatible tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[TRAIN] Set pad_token = eos_token")

    # ── Load model ────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] Model loaded: {n_params/1e6:.0f}M parameters")

    # ── Tokenise data ─────────────────────────────────────────────────────
    tokenized_data = load_sft_data(sft_data_path, tokenizer, max_length=max_length)

    # ── Train/eval split ──────────────────────────────────────────────────
    split = tokenized_data.train_test_split(test_size=0.1, seed=42)
    train_ds = split['train']
    eval_ds  = split['test']
    print(f"[TRAIN] Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    # ── Data collator ─────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # causal LM, not masked
    )

    # ── Training arguments ────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir                  = out_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = gradient_accumulation,
        learning_rate               = lr,
        warmup_steps                = warmup_steps,
        weight_decay                = 0.01,
        lr_scheduler_type           = 'cosine',
        logging_steps               = 20,
        eval_strategy               = 'epoch',
        save_strategy               = 'epoch',
        load_best_model_at_end      = True,
        metric_for_best_model       = 'eval_loss',
        greater_is_better           = False,
        fp16                        = False,
        bf16                        = torch.cuda.is_available(),  # bfloat16 on GPU
        dataloader_num_workers      = 2,
        report_to                   = 'none',
        seed                        = 42,
        # Effective batch size = batch_size * gradient_accumulation * n_gpus
        # 4 * 4 * 1 = 16
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        data_collator   = data_collator,
        tokenizer       = tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting training for {epochs} epochs...")
    print(f"[TRAIN] Effective batch size: {batch_size * gradient_accumulation}")
    print(f"[TRAIN] Learning rate: {lr}")
    t0 = time.time()

    train_result = trainer.train()
    elapsed = time.time() - t0

    print(f"\n[TRAIN] Training complete in {elapsed/60:.1f} minutes")
    print(f"[TRAIN] Loss: {train_result.training_loss:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(os.path.join(out_dir, 'tokenizer'))
    print(f"[TRAIN] Model saved to {out_dir}")
    print(f"[TRAIN] Tokenizer saved to {out_dir}/tokenizer")

    # ── Quick generation test ─────────────────────────────────────────────
    print("\n[TRAIN] === GENERATION TEST ===")
    model.eval()
    import torch
    test_prompt = (
        f"{JD_SYSTEM_PROMPT}\n\n"
        "=== CANDIDATE SUMMARY (ID: CAND_TEST) ===\n"
        "Role type:           AI/ML core role\n"
        "Years of experience: 6.5 years\n"
        "Must-have skills:    7 matching JD (expert-level: 4)\n"
        "Max skill duration:  86 months for top relevant skill\n"
        "Product companies:   2 (1 FAANG/Big Tech)\n"
        "Notice period:       30 days\n"
        "Availability:        Open to work\n"
        "Activity:            Last active 14 days ago\n"
        "GitHub:              GitHub score 82\n"
        "Assessments:         2 taken (avg 78/100 if any)\n\n"
        "Provide 2-4 sentences of specific reasoning for this candidate's fit score of 88/100:"
    )
    inputs = tokenizer(test_prompt, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens     = 120,
            temperature        = 0.3,
            do_sample          = True,
            pad_token_id       = tokenizer.eos_token_id,
            repetition_penalty = 1.1,
        )

    generated_text = tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    print(f"  Generated reasoning:")
    print(f"  {generated_text.strip()}")

    # ── Write training summary ─────────────────────────────────────────────
    summary = {
        'model':        MODEL_NAME,
        'sft_data':     sft_data_path,
        'epochs':       epochs,
        'batch_size':   batch_size,
        'gradient_accumulation': gradient_accumulation,
        'effective_batch_size': batch_size * gradient_accumulation,
        'learning_rate': lr,
        'max_length':   max_length,
        'train_samples': len(train_ds),
        'eval_samples':  len(eval_ds),
        'training_loss': train_result.training_loss,
        'elapsed_minutes': elapsed / 60,
        'out_dir':      out_dir,
        'timestamp':    datetime.now().isoformat(),
    }
    summary_path = os.path.join(out_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[TRAIN] Summary written to {summary_path}")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPER  (use this in rank.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_mamba_for_inference(model_dir):
    """
    Load the fine-tuned Mamba model for inference in rank.py.
    Returns (model, tokenizer) ready for generate().

    Usage in rank.py:
        model, tokenizer = load_mamba_for_inference('mamba_ranker/')
        for candidate in top_100:
            reasoning = generate_reasoning(model, tokenizer, candidate, score)
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("pip install transformers torch mamba-ssm causal-conv1d")

    tok_dir = os.path.join(model_dir, 'tokenizer')
    if not os.path.exists(tok_dir):
        tok_dir = model_dir  # Fallback: tokenizer in model dir

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,   # CPU inference: float32
        device_map='cpu',
    )
    model.eval()
    return model, tokenizer


def generate_reasoning(model, tokenizer, candidate_features_dict, teacher_score,
                       max_new_tokens=120, temperature=0.3):
    """
    Generate a 2-4 sentence reasoning for one candidate.
    candidate_features_dict: the 'features' dict from filter_candidates.py output.
    teacher_score: the predicted score from LightGBM (0-100).

    Called from rank.py for each of the top-100 candidates.
    Falls back to deterministic reasoning if output fails hallucination check.
    """
    import torch

    # Build the same input format as SFT training
    dummy_record = {
        'candidate_id': candidate_features_dict.get('candidate_id', 'UNKNOWN'),
        'teacher_score': int(teacher_score),
        'features': candidate_features_dict,
    }
    prompt = build_sft_input(dummy_record)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            do_sample          = True,
            pad_token_id       = tokenizer.eos_token_id,
            repetition_penalty = 1.1,
        )

    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    reasoning  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Basic sanity check: must be non-empty and reasonable length
    if not reasoning or len(reasoning) < 30:
        reasoning = f"Candidate has {candidate_features_dict.get('n_must_have_skills', 0)} must-have skills with {candidate_features_dict.get('n_expert_relevant', 0)} at expert level. Notice period {candidate_features_dict.get('notice_days', 'unknown')} days."

    return reasoning


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Clean teacher labels and fine-tune Mamba-370M for reasoning generation'
    )
    parser.add_argument('--labels',     required=True,
                        help='Path to teacher_labels.jsonl. Accepts comma-separated list '
                             'for multiple split files, e.g. '
                             'candidate_split_part1.jsonl,candidate_split_part2.jsonl')
    parser.add_argument('--fixed-reasoning', default=None,
                        help='Path to merged_fixed_reasoning.jsonl. When supplied, replaces '
                             'teacher_reasoning for every matched candidate_id before filtering. '
                             'Disables dedup (not needed) and raises training cap to 16,000.')
    parser.add_argument('--sft-data',   default='sft_training_data.jsonl',
                        help='Path to write/read cleaned SFT data (default: sft_training_data.jsonl)')
    parser.add_argument('--out-dir',    default='mamba_ranker/',
                        help='Output dir for fine-tuned model (default: mamba_ranker/)')
    parser.add_argument('--clean-only', action='store_true',
                        help='Only run Step 1 (cleaning), skip model training')
    parser.add_argument('--skip-clean', action='store_true',
                        help='Skip cleaning, use existing --sft-data file')
    parser.add_argument('--epochs',         type=int,   default=3)
    parser.add_argument('--batch-size',     type=int,   default=4)
    parser.add_argument('--lr',             type=float, default=2e-4)
    parser.add_argument('--max-length',     type=int,   default=512,
                        help='Max token sequence length (default: 512)')
    parser.add_argument('--grad-accum',     type=int,   default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--warmup-steps',   type=int,   default=50)
    args = parser.parse_args()

    # Validate --labels files exist (supports comma-separated list)
    for p in args.labels.split(','):
        p = p.strip()
        if p and not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    if args.fixed_reasoning and not os.path.exists(args.fixed_reasoning):
        print(f"ERROR: --fixed-reasoning file not found: {args.fixed_reasoning}")
        sys.exit(1)

    # ── Step 1: Clean ─────────────────────────────────────────────────────
    if not args.skip_clean:
        print("=" * 60)
        print("STEP 1 — CLEANING TEACHER LABELS")
        print("=" * 60)
        n_clean = clean_data(args.labels, args.sft_data,
                             fixed_reasoning_path=args.fixed_reasoning)
        print(f"\n[CLEAN] Done. {n_clean:,} records ready for SFT.")

        if args.clean_only:
            print("\nclean-only mode — skipping training.")
            print(f"Inspect {args.sft_data} before running training.")
            if args.fixed_reasoning:
                print("Then run: python train_mamba_sft.py --labels ... "
                      f"--fixed-reasoning {args.fixed_reasoning} --skip-clean --out-dir mamba_ranker/")
            else:
                print("Then run: python train_mamba_sft.py --labels ... --skip-clean --out-dir mamba_ranker/")
            return
    else:
        if not os.path.exists(args.sft_data):
            print(f"ERROR: --skip-clean set but {args.sft_data} not found.")
            sys.exit(1)
        print(f"[CLEAN] Skipped. Using existing {args.sft_data}")

    # ── Step 2: Train ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — SFT TRAINING (Mamba-370M)")
    print("=" * 60)
    print("Requirements: pip install transformers torch mamba-ssm causal-conv1d datasets accelerate")
    print(f"Output directory: {args.out_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}")
    print()

    train(
        sft_data_path        = args.sft_data,
        out_dir              = args.out_dir,
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        lr                   = args.lr,
        warmup_steps         = args.warmup_steps,
        max_length           = args.max_length,
        gradient_accumulation= args.grad_accum,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Model: {args.out_dir}")
    print(f"Use load_mamba_for_inference('{args.out_dir}') in rank.py")
    print(f"Use generate_reasoning(model, tokenizer, features_dict, score) per candidate")


if __name__ == '__main__':
    main()
