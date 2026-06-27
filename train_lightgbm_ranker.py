#!/usr/bin/env python3
"""
train_lightgbm_ranker.py
========================
Trains a LightGBM LambdaRank model on teacher-scored candidates.
Uses 54 structured features extracted by filter_candidates.py and
teacher_score (0-100) from the teacher model as training labels.

Usage:
    python train_lightgbm_ranker.py \
        --labels teacher_labels.jsonl \
        --out ranker.pkl

    python train_lightgbm_ranker.py \
        --labels teacher_labels.jsonl \
        --out ranker.pkl \
        --no-prescore      # drop prescore from features (ablation)
        --tune             # run Optuna hyperparameter search first

Output:
    ranker.pkl           — trained LightGBM Booster (use with lgb.Booster())
    ranker_features.json — ordered feature name list for inference
    ranker_report.txt    — training summary, NDCG scores, feature importances

Runtime:
    ~60 seconds on CPU. Model file ~2MB.
"""

import json
import argparse
import pickle
import os
import sys
import time
import collections
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# ─── Relevance Grade Mapping ─────────────────────────────────────────────────
# LightGBM LambdaRank requires integer grade labels (0, 1, 2, ...).
# We bin teacher_score (0-100) into 5 tiers based on the JD scoring rubric.
# Grade 4 = exceptional/strong fit (teacher_score >= 75)
# Grade 3 = moderate fit         (60 <= score < 75)
# Grade 2 = weak-moderate        (45 <= score < 60)
# Grade 1 = weak                 (35 <= score < 45)
# Grade 0 = not relevant         (score < 35)
GRADE_THRESHOLDS = [(75, 4), (60, 3), (45, 2), (35, 1), (0, 0)]

# label_gain[g] = gain for grade g — exponential so grade 4 >> grade 3 >> ...
# This makes the model focus hard on getting grade-4 candidates to the very top.
LABEL_GAIN = [0, 1, 3, 7, 15]

# LightGBM hard limit: max candidates per query group = 10,000
MAX_GROUP_SIZE = 9000


def score_to_grade(score):
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return 0


def load_data(labels_path, drop_prescore=False):
    """
    Load teacher_labels.jsonl.
    Returns X (numpy), grades (numpy), raw_scores (numpy), feature_names (list).
    """
    records = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("ERROR: No records loaded.")
        sys.exit(1)

    print(f"Loaded {len(records):,} teacher-scored candidates.")

    # Feature names from first record
    feat_keys = list(records[0]['features'].keys())

    if drop_prescore:
        # Drop prescore (our formula output) and derived sub-scores.
        # Use this for ablation: does teacher add signal beyond the formula?
        drop = {'prescore', 'behavioral_score', 'yoe_fit_score',
                'education_score', 'location_score'}
        feat_keys = [k for k in feat_keys if k not in drop]
        print(f"drop_prescore=True: using {len(feat_keys)} features (dropped formula sub-scores)")
    else:
        print(f"Using all {len(feat_keys)} features including prescore.")

    X = np.array(
        [[r['features'][k] for k in feat_keys] for r in records],
        dtype=np.float32
    )
    raw_scores = np.array([r['teacher_score'] for r in records], dtype=np.float32)
    grades     = np.array([score_to_grade(s) for s in raw_scores], dtype=np.int32)
    cand_ids   = [r['candidate_id'] for r in records]

    # Grade distribution
    print("Grade distribution:")
    for g in range(5):
        cnt = (grades == g).sum()
        pct = cnt / len(grades) * 100
        print(f"  Grade {g}: {cnt:>5,} ({pct:.1f}%)")

    return X, grades, raw_scores, feat_keys, cand_ids


def make_train_val_split(X, grades, raw_scores, cand_ids, val_frac=0.15, seed=42):
    """
    Stratified split on grade labels.
    Ensures grade-4 and grade-3 candidates appear in both train and val.
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(grades)),
        test_size=val_frac,
        random_state=seed,
        stratify=grades
    )
    print(f"\nTrain: {len(train_idx):,}  Val: {len(val_idx):,}")
    for g in range(5):
        tr = (grades[train_idx] == g).sum()
        va = (grades[val_idx]   == g).sum()
        print(f"  Grade {g}: train={tr}, val={va}")

    return train_idx, val_idx


def build_lgb_datasets(X, grades, train_idx, val_idx, feat_keys):
    """
    Build LightGBM Dataset objects with proper query groups.
    LambdaRank requires group= to know which candidates compete against each other.
    All candidates are for the SAME JD (one query), but LightGBM has a per-group
    cap of 10,000 rows. We split train into multiple equal-sized groups.
    """
    X_train, y_train = X[train_idx], grades[train_idx]
    X_val,   y_val   = X[val_idx],   grades[val_idx]

    n_train = len(train_idx)
    # Build groups: chunk train into groups of MAX_GROUP_SIZE
    train_groups = []
    remaining = n_train
    while remaining > 0:
        g = min(remaining, MAX_GROUP_SIZE)
        train_groups.append(g)
        remaining -= g

    # Val is one group (always < 10K since val_frac=0.15 of 15600 = 2340)
    val_groups = [len(val_idx)]

    print(f"\nTrain query groups: {train_groups}")
    print(f"Val query group:    {val_groups}")

    train_ds = lgb.Dataset(
        X_train, label=y_train,
        group=train_groups,
        feature_name=feat_keys,
        free_raw_data=False
    )
    val_ds = lgb.Dataset(
        X_val, label=y_val,
        group=val_groups,
        reference=train_ds,
        free_raw_data=False
    )
    return train_ds, val_ds, X_train, y_train, X_val, y_val


def get_default_params():
    return {
        # Objective
        'objective':                  'lambdarank',
        'metric':                     'ndcg',
        'ndcg_eval_at':               [10, 50, 100],
        'lambdarank_truncation_level': 100,
        'label_gain':                  LABEL_GAIN,

        # Tree structure
        'num_leaves':     63,
        'max_depth':      -1,          # unlimited
        'min_data_in_leaf': 10,

        # Learning
        'learning_rate':   0.05,
        'n_estimators':    500,        # upper bound; early stopping will trim

        # Regularisation
        'lambda_l1':       0.1,
        'lambda_l2':       0.1,
        'min_gain_to_split': 0.0,

        # Subsampling
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq':     5,

        # Misc
        'num_threads':    0,           # use all cores
        'verbose':       -1,
        'seed':           42,
    }


def run_optuna_search(train_ds, val_ds, n_trials=30):
    """
    Optional Optuna hyperparameter search.
    Searches over num_leaves, learning_rate, lambda_l1/l2, feature_fraction.
    Returns best params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna not installed. pip install optuna to enable tuning.")
        return get_default_params()

    def objective(trial):
        params = {
            'objective':    'lambdarank',
            'metric':       'ndcg',
            'ndcg_eval_at': [10],
            'lambdarank_truncation_level': 100,
            'label_gain':   LABEL_GAIN,
            'verbose':     -1,
            'seed':         42,
            'num_leaves':   trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'lambda_l1':    trial.suggest_float('lambda_l1', 1e-4, 1.0, log=True),
            'lambda_l2':    trial.suggest_float('lambda_l2', 1e-4, 1.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 30),
        }
        booster = lgb.train(
            params, train_ds,
            valid_sets=[val_ds],
            num_boost_round=300,
            callbacks=[
                lgb.early_stopping(20, verbose=False),
                lgb.log_evaluation(-1),
            ]
        )
        return booster.best_score['valid_0']['ndcg@10']

    print(f"Running Optuna search ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best NDCG@10: {study.best_value:.4f}")
    best = get_default_params()
    best.update(study.best_params)
    return best


def train_model(train_ds, val_ds, params, num_boost_round=1000, early_stopping=50):
    """Train and return the final LightGBM booster."""
    print(f"\nTraining LightGBM LambdaRank...")
    print(f"  num_boost_round={num_boost_round}, early_stopping={early_stopping}")

    booster = lgb.train(
        params,
        train_ds,
        valid_sets=[val_ds],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=True),
            lgb.log_evaluation(100),
        ]
    )
    return booster


def evaluate(booster, X_val, y_val, raw_val_scores, feat_keys, top_k=(10, 50, 100)):
    """
    Compute NDCG@K metrics and print top-K quality.
    Also shows Precision@K — fraction of top-K that are grade 3 or 4.
    """
    preds  = booster.predict(X_val)
    sorted_idx = np.argsort(preds)[::-1]

    results = {}

    def dcg_at_k(grades_sorted, k):
        grades_k = grades_sorted[:k]
        gains = (2 ** grades_k - 1)
        discounts = np.log2(np.arange(2, k + 2))
        return float(np.sum(gains / discounts))

    def idcg_at_k(grades_available, k):
        ideal = np.sort(grades_available)[::-1][:k]
        gains = (2 ** ideal - 1)
        discounts = np.log2(np.arange(2, len(ideal) + 2))
        return float(np.sum(gains / discounts))

    print("\nEvaluation metrics:")
    for k in top_k:
        top_k_grades = y_val[sorted_idx[:k]]
        dcg  = dcg_at_k(top_k_grades, k)
        idcg = idcg_at_k(y_val, k)
        ndcg = dcg / idcg if idcg > 0 else 0
        prec = (top_k_grades >= 3).sum() / k
        results[f'ndcg@{k}']      = ndcg
        results[f'precision@{k}'] = prec
        print(f"  NDCG@{k}: {ndcg:.4f}  Precision@{k} (grade>=3): {prec:.4f}")

    print(f"\nTop-20 raw teacher scores predicted:")
    top20 = raw_val_scores[sorted_idx[:20]]
    print(f"  {list(top20.astype(int))}")
    print(f"Top-20 grades: {list(y_val[sorted_idx[:20]])}")

    return results, preds, sorted_idx


def feature_importance_report(booster, feat_keys):
    """Return sorted feature importance (gain) as list of (name, score) tuples."""
    importances = booster.feature_importance(importance_type='gain')
    pairs = sorted(zip(feat_keys, importances), key=lambda x: -x[1])
    print("\nTop 20 feature importances (gain):")
    for name, imp in pairs[:20]:
        bar = '█' * int(imp / max(1, pairs[0][1]) * 30)
        print(f"  {name:<35} {imp:>8.1f}  {bar}")
    return pairs


def write_report(path, args, n_records, grade_dist, train_time,
                 metrics, feat_imp_pairs, booster, params):
    lines = []
    div = '=' * 70

    def L(s=''):
        lines.append(str(s))

    L(div)
    L('  LIGHTGBM LAMBDARANK — TRAINING REPORT')
    L(f'  Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    L(div)
    L()
    L('── DATA ────────────────────────────────────────────────────────────')
    L(f'  Input:          {args.labels}')
    L(f'  Total records:  {n_records:,}')
    L(f'  drop_prescore:  {args.no_prescore}')
    L()
    L('  Grade distribution:')
    for g, cnt in grade_dist.items():
        L(f'    Grade {g}: {cnt:,}')
    L()
    L('── TRAINING ────────────────────────────────────────────────────────')
    L(f'  Best iteration:   {booster.best_iteration}')
    L(f'  Training time:    {train_time:.1f} seconds')
    L(f'  Output model:     {args.out}')
    L()
    L('── HYPERPARAMETERS ─────────────────────────────────────────────────')
    for k, v in params.items():
        if k not in ('verbose',):
            L(f'  {k}: {v}')
    L()
    L('── VALIDATION METRICS ──────────────────────────────────────────────')
    for k, v in metrics.items():
        L(f'  {k}: {v:.4f}')
    L()
    L('── FEATURE IMPORTANCES (gain) ───────────────────────────────────────')
    for name, imp in feat_imp_pairs:
        L(f'  {name:<35} {imp:>10.1f}')
    L()
    L(div)
    L('  END OF REPORT')
    L(div)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nReport written to {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train LightGBM LambdaRank on teacher-scored candidates'
    )
    parser.add_argument('--labels',      required=True,
                        help='Path to teacher_labels.jsonl')
    parser.add_argument('--out',         default='ranker.pkl',
                        help='Output path for trained model (default: ranker.pkl)')
    parser.add_argument('--no-prescore', action='store_true',
                        help='Drop prescore from features (ablation experiment)')
    parser.add_argument('--tune',        action='store_true',
                        help='Run Optuna hyperparameter search before final training')
    parser.add_argument('--n-trials',    type=int, default=30,
                        help='Number of Optuna trials (default: 30)')
    parser.add_argument('--val-frac',    type=float, default=0.15,
                        help='Validation fraction (default: 0.15)')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"ERROR: File not found: {args.labels}")
        sys.exit(1)

    # ── 1. Load data ──────────────────────────────────────────────────────
    X, grades, raw_scores, feat_keys, cand_ids = load_data(
        args.labels, drop_prescore=args.no_prescore
    )
    grade_dist = {g: int((grades == g).sum()) for g in range(5)}

    # ── 2. Train/val split ────────────────────────────────────────────────
    train_idx, val_idx = make_train_val_split(
        X, grades, raw_scores, cand_ids, val_frac=args.val_frac, seed=args.seed
    )

    # ── 3. Build LightGBM datasets ────────────────────────────────────────
    train_ds, val_ds, X_train, y_train, X_val, y_val = build_lgb_datasets(
        X, grades, train_idx, val_idx, feat_keys
    )
    raw_val_scores = raw_scores[val_idx]

    # ── 4. Hyperparameters ────────────────────────────────────────────────
    if args.tune:
        params = run_optuna_search(train_ds, val_ds, n_trials=args.n_trials)
    else:
        params = get_default_params()

    # ── 5. Train ──────────────────────────────────────────────────────────
    t0      = time.time()
    booster = train_model(train_ds, val_ds, params, num_boost_round=1000, early_stopping=50)
    train_time = time.time() - t0
    print(f"\nTraining complete in {train_time:.1f}s — best iteration: {booster.best_iteration}")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    metrics, preds_val, sorted_val_idx = evaluate(
        booster, X_val, y_val, raw_val_scores, feat_keys
    )

    # ── 7. Feature importances ────────────────────────────────────────────
    feat_imp_pairs = feature_importance_report(booster, feat_keys)

    # ── 8. Save model ─────────────────────────────────────────────────────
    model_data = {
        'booster':      booster,
        'feat_keys':    feat_keys,
        'grade_thresholds': GRADE_THRESHOLDS,
        'label_gain':   LABEL_GAIN,
        'best_iteration': booster.best_iteration,
        'metrics':      metrics,
    }
    with open(args.out, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")

    # Save feature names separately (for inference scripts that don't load pickle)
    feat_path = args.out.replace('.pkl', '_features.json')
    with open(feat_path, 'w') as f:
        json.dump(feat_keys, f, indent=2)
    print(f"Feature names saved to {feat_path}")

    # ── 9. Write report ───────────────────────────────────────────────────
    report_path = args.out.replace('.pkl', '_report.txt')
    write_report(report_path, args, len(X), grade_dist, train_time,
                 metrics, feat_imp_pairs, booster, params)

    # ── 10. Quick inference demo ──────────────────────────────────────────
    print("\n── Inference demo: scoring 15,600 candidates ──────────────────────")
    t_inf = time.time()
    all_preds = booster.predict(X)
    inf_time = time.time() - t_inf
    top_idx = np.argsort(all_preds)[::-1]
    print(f"  Inference time: {inf_time*1000:.1f}ms for {len(X):,} candidates")
    print(f"  Top-5 candidates by model score:")
    for rank, idx in enumerate(top_idx[:5], 1):
        print(f"    #{rank}  {cand_ids[idx]}  teacher_score={raw_scores[idx]:.0f}  "
              f"grade={grades[idx]}  model_score={all_preds[idx]:.3f}")

    print("\nDone.")
    return booster, feat_keys


if __name__ == '__main__':
    main()
