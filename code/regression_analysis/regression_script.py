#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "correlation_analysis"))
from corr import scale_features_for_correlation
from shared.topic_maps import ASSEMBLY_TOPICS, EXPERT_TOPICS


def drop_high_corr_features(X: pd.DataFrame, threshold: float = 0.90) -> tuple[pd.DataFrame, list[str]]:
    """Drop one feature from each highly-correlated pair (absolute Pearson > threshold)."""
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X.drop(columns=to_drop), to_drop


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor (VIF) for each column in X (must be numeric, no NaNs)."""
    vifs = {}
    cols = list(X.columns)
    for i, col in enumerate(cols):
        y = X[col].values
        X_others = X.drop(columns=[col]).values
        if X_others.shape[1] == 0:
            vifs[col] = 1.0
            continue
        model = LinearRegression()
        model.fit(X_others, y)
        r2 = model.score(X_others, y)
        vifs[col] = 1.0 / max(1e-12, (1.0 - r2))
    return pd.Series(vifs).sort_values(ascending=False)


def drop_high_vif_features(X: pd.DataFrame, vif_threshold: float = 10.0, max_iter: int = 50) -> tuple[pd.DataFrame, list[str]]:
    """Iteratively drop the feature with the highest VIF until all VIF <= threshold."""
    dropped = []
    X_work = X.copy()

    for _ in range(max_iter):
        X_num = X_work.astype(float)
        vifs = compute_vif(X_num)
        worst_feat, worst_vif = vifs.index[0], vifs.iloc[0]
        if worst_vif <= vif_threshold:
            break
        X_work = X_work.drop(columns=[worst_feat])
        dropped.append(worst_feat)

        if X_work.shape[1] <= 1:
            break

    return X_work, dropped


def regression_report(y_true, y_pred, label: str, log_func=print):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    log_func(f"\n[{label}]")
    log_func(f"  RMSE: {rmse:.4f}")
    log_func(f"  MAE : {mae:.4f}")
    log_func(f"  R^2 : {r2:.4f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def main():
    ap = argparse.ArgumentParser(description="Linear regression with feature-specific scaling by topic")
    ap.add_argument("--data_dir", default="data/assemblies/new_feats", help="Base data directory")
    ap.add_argument("--dataset", choices=["assemblies", "experts"], required=True, help="Dataset type")
    ap.add_argument("--model_type", choices=["base", "finetuned"], required=True, help="Model configuration")
    ap.add_argument("--target", default="expertise_llm_score", help="Target column name")
    ap.add_argument("--threshold", type=int, default=5, help="Min non-function words threshold")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corr_thresh", type=float, default=0.90)
    ap.add_argument("--use_vif", action="store_true")
    ap.add_argument("--vif_thresh", type=float, default=10.0)
    ap.add_argument("--also_ridge", action="store_true", help="Fit Ridge as robustness check")
    ap.add_argument("--output_dir", default="code/regression_analysis/results", help="Output directory for reports")
    args = ap.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset / args.model_type / f"threshold_{args.threshold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "regression_report.txt"
    
    # Open report file
    report = open(output_file, 'w')
    
    def log(msg):
        """Print to console and write to report file."""
        print(msg)
        report.write(msg + "\n")

    # Load and combine data
    log(f"\n{'='*80}")
    log(f"REGRESSION ANALYSIS: {args.dataset.upper()} - {args.model_type.upper()}")
    log(f"Threshold: >{args.threshold} non-function words")
    log(f"{'='*80}\n")
    
    base_path = Path(args.data_dir)
    
    # Define topic folders based on dataset
    if args.dataset == "assemblies":
        topic_folders = ASSEMBLY_TOPICS
        data_path = base_path
    else:  # experts
        topic_folders = EXPERT_TOPICS
        data_path = base_path / "experts"
    
    # Load all CSVs
    dfs = []
    for folder, topic_name in topic_folders.items():
        folder_path = data_path / folder
        if not folder_path.exists():
            folder_path = data_path / f"{folder}_csv"
        if not folder_path.exists():
            log(f"⚠️  Skipping {folder}: not found")
            continue
            
        for csv_file in folder_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            df['topic'] = topic_name
            dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No data found in {data_path}")
    
    df = pd.concat(dfs, ignore_index=True)
    log(f"Loaded {len(df)} rows from {len(dfs)} files")
    
    # Filter by content word threshold
    if 'num_non_function_words' in df.columns:
        df['num_non_function_words'] = pd.to_numeric(df['num_non_function_words'], errors='coerce')
        df = df[df['num_non_function_words'] > args.threshold].copy()
        log(f"After filtering: {len(df)} rows (>{args.threshold} content words)")
    
    # Extract target
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns")
    
    y = df[args.target].astype(float)
    
    # Get numeric features (filter by model_type)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude non-feature columns
    excluded = ['text', 'speaker', 'source_file', 'Unnamed', 'id', 'index', 
                'hedge_ratio', 'cluster_', args.target]
    features = [c for c in numeric_cols if c not in excluded and 
                not any(ex in c.lower() for ex in excluded)]
    
    # Filter by model type
    features = [f for f in features if 
                not (('base' in f.lower() and args.model_type != 'base') or
                     ('finetuned' in f.lower() and args.model_type != 'finetuned'))]
    
    log(f"\nSelected {len(features)} features for {args.model_type} model")
    
    # Apply feature-specific scaling BY TOPIC (as in correlation analysis)
    X = df[features + ['topic']].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    
    log("\nApplying feature-specific scaling by topic...")
    X_scaled = scale_features_for_correlation(X, scope='by_topic')
    X_scaled = X_scaled.drop(columns=['topic'])  # Remove topic after scaling
    
    # Align with y after any dropna
    valid_mask = X_scaled.notna().all(axis=1) & y.notna()
    X_final = X_scaled[valid_mask]
    y_final = y[valid_mask]
    
    log(f"Final dataset: {len(X_final)} samples, {X_final.shape[1]} features\n")

    # Multicollinearity handling on scaled features
    X_corr, dropped_corr = drop_high_corr_features(X_final, threshold=args.corr_thresh)
    log(f"Dropped {len(dropped_corr)} highly correlated features (>{args.corr_thresh}): {dropped_corr[:5]}...")

    # Optional VIF pruning
    if args.use_vif:
        X_tmp = X_corr.fillna(X_corr.median())
        X_vif, dropped_vif = drop_high_vif_features(X_tmp, vif_threshold=args.vif_thresh)
        X_model = X_corr[list(X_vif.columns)]
        log(f"Dropped {len(dropped_vif)} high VIF features (>{args.vif_thresh}): {dropped_vif[:5]}...")
    else:
        X_model = X_corr

    log(f"\nFinal feature count: {X_model.shape[1]}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_final, test_size=args.test_size, random_state=args.seed
    )

    # Simple pipeline: impute only (already scaled by topic)
    pipe_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ])

    pipe_lr.fit(X_train, y_train)
    pred_train = pipe_lr.predict(X_train)
    pred_test = pipe_lr.predict(X_test)

    regression_report(y_train, pred_train, "LinearRegression TRAIN", log)
    regression_report(y_test, pred_test, "LinearRegression TEST", log)

    # Coefficients (features already scaled by topic-specific method)
    coef = pipe_lr.named_steps["model"].coef_
    coef_s = pd.Series(coef, index=X_model.columns).sort_values(key=np.abs, ascending=False)
    log("\nTop 15 coefficients (|beta|):")
    log(str(coef_s.head(15)))

    # Permutation importance
    perm = permutation_importance(
        pipe_lr, X_test, y_test, n_repeats=30, random_state=args.seed, 
        scoring="neg_root_mean_squared_error"
    )
    perm_s = pd.Series(perm.importances_mean, index=X_model.columns).sort_values(ascending=False)
    log("\nTop 15 permutation importances:")
    log(str(perm_s.head(15)))

    # Residual diagnostics
    resid = y_test - pred_test
    log("\nResidual diagnostics (TEST):")
    log(f"  mean(resid): {resid.mean():.4f}")
    log(f"  std(resid) : {resid.std(ddof=1):.4f}")
    corr_abs = np.corrcoef(np.abs(resid), pred_test)[0, 1]
    log(f"  corr(|resid|, yhat): {corr_abs:.4f}  (heteroscedasticity indicator)")
    # Optional Ridge
    if args.also_ridge:
        pipe_ridge = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=1.0))
        ])
        pipe_ridge.fit(X_train, y_train)
        pred_ridge = pipe_ridge.predict(X_test)
        regression_report(y_test, pred_ridge, "Ridge TEST (alpha=1.0)", log)

        ridge_coef = pd.Series(pipe_ridge.named_steps["model"].coef_, index=X_model.columns)\
                      .sort_values(key=np.abs, ascending=False)
        log("\nTop 15 Ridge coefficients (|beta|):")
        log(str(ridge_coef.head(15)))
    
    # Save coefficient and importance CSVs
    coef_s.to_csv(output_dir / "coefficients.csv", header=["coefficient"])
    perm_s.to_csv(output_dir / "permutation_importance.csv", header=["importance"])
    
    log(f"\n{'='*80}")
    log(f"Report saved to: {output_file}")
    log(f"Coefficients saved to: {output_dir / 'coefficients.csv'}")
    log(f"Permutation importance saved to: {output_dir / 'permutation_importance.csv'}")
    log(f"{'='*80}")
    
    report.close()


if __name__ == "__main__":
    main()
