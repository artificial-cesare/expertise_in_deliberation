import os
import sys
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.util import batch_to_device

# Import filter function
sys.path.append(str(Path(__file__).parent.parent.parent / "handcrafted_features"))
from filter_function import is_substantive_speech_act

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID   = os.getenv("MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
DATA_DIR   = Path(os.getenv("DATA_DIR", "/home/u933585/thesis/data/assemblies/processed_files/experts"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "qwen3_expert_contrastive_final"))
BSZ        = int(os.getenv("BSZ", "16"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))
EPOCHS     = int(os.getenv("EPOCHS", "10"))
LR         = float(os.getenv("LR", "2e-5"))
MIN_LR     = float(os.getenv("MIN_LR", "1e-6"))
MAX_LEN    = int(os.getenv("MAX_LEN", "1024"))
USE_FA2    = bool(int(os.getenv("USE_FA2", "1")))
GPU_ID     = int(os.getenv("GPU_ID", "1"))
VAL_SPLIT  = float(os.getenv("VAL_SPLIT", "0.15"))
PATIENCE   = int(os.getenv("PATIENCE", "3"))
THRESHOLD  = int(os.getenv("CONTENT_THRESHOLD", "5"))
K_VALUES   = [1, 3, 5, 10, 20]

TOPICS = ["constitution", "future", "gender"]
TOPIC_TO_ID = {topic: idx for idx, topic in enumerate(TOPICS)}

# Setup device
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    print(f"✓ GPU {GPU_ID}: {torch.cuda.get_device_name(0)}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU")

# ============================================================================
# Data Loading
# ============================================================================
def load_speeches(data_dir: Path, topics: List[str], threshold: int) -> Tuple[List[str], np.ndarray]:
    """Load and filter expert speeches by content word count."""
    all_texts, all_labels = [], []
    total_before = total_after = 0
    
    print("=" * 80)
    print(f"Loading speeches (threshold: ≥{threshold} content words)")
    print("=" * 80)
    
    for topic in topics:
        topic_dir = data_dir / topic
        if not topic_dir.exists():
            print(f"⚠️  {topic}: directory not found")
            continue
        
        print(f"\n📂 {topic.upper()}")
        topic_texts = []
        
        for csv_file in sorted(topic_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                
                # Find text column
                text_col = next((c for c in ['text', 'comment'] if c in df.columns), None)
                if not text_col:
                    print(f"   ⚠️  {csv_file.name}: No text column")
                    continue
                
                # Check features exist
                if 'num_non_function_words' not in df.columns:
                    print(f"   ⚠️  {csv_file.name}: Run extract_expert_features.py first!")
                    continue
                
                before = len(df)
                total_before += before
                
                # Filter substantive speeches
                mask = df['num_non_function_words'].apply(
                    lambda x: is_substantive_speech_act(int(x), threshold) if pd.notna(x) else False
                )
                
                speeches = df[mask][text_col].dropna().astype(str).str.strip()
                speeches = speeches[speeches != ''].tolist()
                
                topic_texts.extend(speeches)
                total_after += len(speeches)
                
                retention = (len(speeches) / before * 100) if before > 0 else 0
                print(f"   ✓ {csv_file.name}: {len(speeches)}/{before} ({retention:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ {csv_file.name}: {e}")
        
        all_texts.extend(topic_texts)
        all_labels.extend([TOPIC_TO_ID[topic]] * len(topic_texts))
        print(f"   Total: {len(topic_texts)}")
    
    retention = (total_after / total_before * 100) if total_before > 0 else 0
    print(f"\n✅ Loaded: {total_after:,}/{total_before:,} ({retention:.1f}%)")
    
    return all_texts, np.array(all_labels)

def build_pairs(texts: List[str], labels: List[int]) -> List[InputExample]:
    """Build contrastive pairs from same topic."""
    import random
    
    # Group by topic
    groups = {label: [] for label in set(labels)}
    for text, label in zip(texts, labels):
        groups[label].append(text)
    
    # Create pairs
    examples = []
    for speeches in groups.values():
        if len(speeches) < 2:
            continue
        for i, anchor in enumerate(speeches):
            positives = speeches[:i] + speeches[i+1:]
            if positives:
                examples.append(InputExample(texts=[anchor, random.choice(positives)]))
    
    return examples

# ============================================================================
# Evaluation
# ============================================================================
def compute_retrieval(model: SentenceTransformer, texts: List[str], 
                     labels: np.ndarray, k_values: List[int]) -> Dict[str, float]:
    """Compute Retrieval@K metrics."""
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
    sims = model.similarity(embeddings, embeddings).cpu().numpy()
    
    metrics = {}
    n = len(texts)
    
    for k in k_values:
        if k >= n:
            continue
        
        score = 0
        for i in range(n):
            top_k_idx = np.argsort(sims[i])[::-1][1:k+1]
            score += np.sum(labels[top_k_idx] == labels[i]) / k
        
        metrics[f"retrieval@{k}"] = score / n
    
    return metrics

def compute_separation(model: SentenceTransformer, texts: List[str], 
                      labels: np.ndarray) -> float:
    """Compute intra-topic vs inter-topic similarity ratio."""
    unique_labels = np.unique(labels)
    intra_sims, inter_sims = [], []
    
    # Intra-topic similarities
    for label in unique_labels:
        topic_texts = [texts[i] for i in range(len(texts)) if labels[i] == label]
        if len(topic_texts) < 2:
            continue
        
        embs = model.encode(topic_texts, convert_to_tensor=True, show_progress_bar=False)
        sims = model.similarity(embs, embs).cpu().numpy()
        mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
        intra_sims.extend(sims[mask])
    
    # Inter-topic similarities
    for i, l1 in enumerate(unique_labels):
        for l2 in unique_labels[i+1:]:
            texts1 = [texts[j] for j in range(len(texts)) if labels[j] == l1]
            texts2 = [texts[j] for j in range(len(texts)) if labels[j] == l2]
            
            emb1 = model.encode(texts1, convert_to_tensor=True, show_progress_bar=False)
            emb2 = model.encode(texts2, convert_to_tensor=True, show_progress_bar=False)
            
            inter_sims.extend(model.similarity(emb1, emb2).cpu().numpy().flatten())
    
    if not intra_sims or not inter_sims:
        return 0.0
    
    return (np.mean(intra_sims) - np.mean(inter_sims)) / (np.mean(intra_sims) + 1e-8)

def evaluate(model: SentenceTransformer, texts: List[str], 
            labels: np.ndarray, k_values: List[int]) -> Dict[str, float]:
    """Full model evaluation."""
    model.eval()
    print("   Computing metrics...")
    
    metrics = compute_retrieval(model, texts, labels, k_values)
    metrics['separation_ratio'] = compute_separation(model, texts, labels)
    
    return metrics

# ============================================================================
# Trainer
# ============================================================================
class Trainer:
    def __init__(self, model, train_loader, val_texts, val_labels, 
                 epochs, lr, min_lr, output_dir, k_values, patience, grad_accum):
        self.model = model
        self.train_loader = train_loader
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.epochs = epochs
        self.output_dir = output_dir
        self.k_values = k_values
        self.patience = patience
        self.grad_accum = grad_accum
        
        self.best_metric = -float('inf')
        self.patience_counter = 0
        self.history = []
        
        # Optimizer & scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
        )
        
        total_steps = max(1, math.ceil(len(train_loader) / grad_accum) * epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=min_lr
        )
        
        self.loss_fn = losses.MultipleNegativesRankingLoss(model)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        # Get model device
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = device
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for step, batch in enumerate(pbar):
            # Unpack batch from smart_batching_collate
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2 and isinstance(batch[0], list) and isinstance(batch[0][0], dict):
                    sentence_features, provided_labels = batch[0], batch[1]
                elif len(batch) >= 1 and isinstance(batch[0], dict):
                    sentence_features = list(batch)
                    provided_labels = None
                else:
                    sentence_features = batch[0] if batch else []
                    provided_labels = None
            elif isinstance(batch, dict):
                sentence_features = [batch]
                provided_labels = None
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")
            
            # Validate sentence_features
            if not isinstance(sentence_features, list) or not sentence_features:
                raise ValueError("Invalid sentence_features from collate")
            
            if 'input_ids' not in sentence_features[0]:
                raise KeyError("Missing 'input_ids' in sentence_features")
            
            batch_size = sentence_features[0]['input_ids'].size(0)
            
            # Create labels
            if provided_labels is None or (
                isinstance(provided_labels, torch.Tensor) and provided_labels.numel() != batch_size
            ):
                labels = torch.arange(batch_size, device=model_device, dtype=torch.long)
            else:
                labels = provided_labels
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                labels = labels.to(model_device)
            
            # Move features to correct device
            sentence_features = [batch_to_device(f, model_device) for f in sentence_features]
            
            # Compute loss
            loss = self.loss_fn(sentence_features, labels)
            (loss / self.grad_accum).backward()
            
            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f"{total_loss/(step+1):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Final update
        if len(self.train_loader) % self.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        with torch.no_grad():
            return evaluate(self.model, self.val_texts, self.val_labels, self.k_values)
    
    def train(self):
        """Main training loop with early stopping."""
        print("\n" + "=" * 80)
        print("Training with Early Stopping")
        print("=" * 80)
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            print(f"\n📊 Validation (Epoch {epoch+1}):")
            metrics = self.validate()
            
            for name, value in metrics.items():
                print(f"   {name}: {value:.4f}")
            
            # Save history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                **{k: float(v) for k, v in metrics.items()}
            })
            
            # Check improvement
            current = metrics.get('retrieval@5', 0)
            
            if current > self.best_metric:
                self.best_metric = current
                self.patience_counter = 0
                print(f"\n✅ New best! Retrieval@5: {current:.4f}")
                self.model.save(str(self.output_dir / "best_model"))
            else:
                self.patience_counter += 1
                print(f"\n⏳ No improvement: {self.patience_counter}/{self.patience}")
                
                if self.patience_counter >= self.patience:
                    print("\n🛑 Early stopping!")
                    break
            
            # Save checkpoint
            self.model.save(str(self.output_dir / f"checkpoint_epoch_{epoch+1}"))
        
        # Save history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✅ Training complete! Best Retrieval@5: {self.best_metric:.4f}")
        print(f"📁 Saved to: {self.output_dir}")

# ============================================================================
# Main
# ============================================================================
def main():
    # Load data
    print(f"\n📂 Loading from: {DATA_DIR}")
    texts, labels = load_speeches(DATA_DIR, TOPICS, THRESHOLD)
    
    if len(texts) == 0:
        print("\n❌ No data! Check:")
        print("   1. Data directory exists")
        print("   2. Features extracted (run extract_expert_features.py)")
        print("   3. Threshold not too high")
        sys.exit(1)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=VAL_SPLIT, stratify=labels, random_state=42
    )
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_texts):,} | Val: {len(val_texts):,}")
    for i, topic in enumerate(TOPICS):
        print(f"   {topic.title()}: {np.sum(train_labels==i):,} / {np.sum(val_labels==i):,}")
    
    # Build pairs
    print(f"\n🔄 Building pairs...")
    train_examples = build_pairs(train_texts, train_labels.tolist())
    print(f"   Pairs: {len(train_examples):,}")
    
    # Load model
    print(f"\n📥 Loading: {MODEL_ID}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if USE_FA2 and torch.cuda.is_available():
        model = SentenceTransformer(
            MODEL_ID,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            },
            trust_remote_code=True
        )
        print("✓ Flash Attention 2")
    else:
        model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
        print("✓ Standard attention")
    
    model = model.to(device)
    model.max_seq_length = MAX_LEN
    print(f"✓ Device: {device} | Max length: {MAX_LEN}")
    
    # Setup training
    train_loader = DataLoader(
        train_examples,
        batch_size=BSZ,
        shuffle=True,
        drop_last=True,
        collate_fn=model.smart_batching_collate
    )
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=EPOCHS,
        lr=LR,
        min_lr=MIN_LR,
        output_dir=OUTPUT_DIR,
        k_values=K_VALUES,
        patience=PATIENCE,
        grad_accum=GRAD_ACCUM
    )
    
    trainer.train()
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)
    
    best_model = SentenceTransformer(str(OUTPUT_DIR / "best_model")).to(device)
    final_metrics = evaluate(best_model, val_texts, val_labels, K_VALUES)
    
    print("\n📊 Final Metrics:")
    for name, value in final_metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Save metrics
    with open(OUTPUT_DIR / "final_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in final_metrics.items()}, f, indent=2)

if __name__ == "__main__":
    main()