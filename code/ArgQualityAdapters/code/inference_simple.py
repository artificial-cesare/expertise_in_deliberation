from transformers import AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterTrainer
from data import InferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from inference_parallel import task2identifier, task2label
import argparse
import pandas as pd
import os

def predict(dataloader, model, out_put_path, task2label, dataset, task, device):
    label_num = task2label[task]
    # create a dictionary with a list for each label
    output_dic = {}
    for i in range(label_num):
        output_dic[i] = []
    
    model.eval()
    with torch.no_grad():
        for id, batch in enumerate(tqdm(dataloader, desc="Scoring batches")):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.cpu()  # Move back to CPU for processing
            
            label_num = task2label[task]
            if label_num == 1:
                probs = np.squeeze(predictions.numpy(), axis=1).tolist()
                output_dic[0].extend(probs)
            elif label_num == 2:
                probs = torch.sigmoid(predictions).tolist()
            else:
                probs = F.softmax(predictions, dim=-1).tolist()
            
            if label_num > 1:
                for i in range(label_num):
                    output_dic[i].extend([el[i] for el in probs])
    
    # Load original CSV to preserve all existing columns
    print(f"  → Loading original file: {out_put_path}")
    original_df = pd.read_csv(out_put_path)
    
    # Get the filtered dataset's index to know which rows were scored
    scored_indices = dataset.index.tolist()
    n_predictions = len(output_dic[0])
    n_scored = len(scored_indices)
    
    if n_predictions != n_scored:
        raise ValueError(
            f"Prediction count mismatch: {n_predictions} predictions but {n_scored} valid rows"
        )
    
    # Initialize overall_score column with NaN if it doesn't exist
    if "overall_score" not in original_df.columns:
        original_df["overall_score"] = np.nan
    
    # Check if num_non_function_words column exists
    if "num_non_function_words" not in original_df.columns:
        raise ValueError(
            "Column 'num_non_function_words' not found in CSV. "
            "Please run feature extraction first."
        )
    
    # Filter: only update rows where num_non_function_words > 5
    valid_mask = original_df.loc[scored_indices, "num_non_function_words"] >= 5
    valid_indices = [idx for idx, is_valid in zip(scored_indices, valid_mask) if is_valid]
    valid_scores = [score for score, is_valid in zip(output_dic[0], valid_mask) if is_valid]
    
    n_valid = len(valid_indices)
    n_filtered = n_scored - n_valid
    
    print(f"  → Filtering based on num_non_function_words > 5:")
    print(f"     • Scored rows: {n_scored}")
    print(f"     • Valid rows (>5 content words): {n_valid}")
    print(f"     • Filtered out: {n_filtered}")
    
    # Update only the valid rows
    if label_num == 1:
        original_df.loc[valid_indices, "overall_score"] = valid_scores
    else:
        # For multi-label tasks, use primary label (index 0)
        original_df.loc[valid_indices, "overall_score"] = valid_scores
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_put_path) or '.', exist_ok=True)
    
    # Save to CSV (overwrites original file with updated overall_score)
    print(f"  → Writing {len(original_df)} rows to: {out_put_path}")
    original_df.to_csv(out_put_path, index=False)
    
    # Verify file was written
    if os.path.exists(out_put_path):
        file_size = os.path.getsize(out_put_path)
        print(f"  ✓ File updated successfully ({file_size} bytes)")
        print(f"     • Rows with scores: {n_valid}")
        print(f"     • Rows with NaN: {len(original_df) - n_valid}")
    else:
        raise IOError(f"Failed to write output file: {out_put_path}")


if __name__ == '__main__':
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('testdata', type=str,
                        help='path to the test data')
    parser.add_argument('text_col', type=str, help="column name of text column")
    parser.add_argument('batch_size', type=int)
    parser.add_argument('task', type=str, help="task name or name of quality dimension")
    parser.add_argument("output_path", type=str, help="path to output file")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"ArgQuality Inference")
    print(f"{'='*60}")
    print(f"Input:  {args.testdata}")
    print(f"Column: {args.text_col}")
    print(f"Task:   {args.task}")
    print(f"Batch:  {args.batch_size}")
    print(f"Output: {args.output_path}")
    print(f"{'='*60}\n")
    
    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    # Load tokenizer and model
    print("Loading RoBERTa base model...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoAdapterModel.from_pretrained("roberta-base")
    
    print(f"Loading adapter: {task2identifier[args.task]}")
    adapter_name = model.load_adapter(task2identifier[args.task], source="hf", set_active=True)
    
    # Move model to device
    model.to(device)
    print(f"✓ Model loaded on {device}\n")

    # Load dataset
    print(f"Loading dataset from: {args.testdata}")
    test = InferenceDataset(path_to_dataset=args.testdata, tokenizer=tokenizer, text_col=args.text_col)
    print(f"✓ Loaded {len(test.dataset)} rows\n")
    
    dataloader = DataLoader(test, batch_size=args.batch_size)
    
    # Run prediction
    print("Starting inference...")
    predict(dataloader=dataloader, model=model, out_put_path=args.output_path, task2label=task2label,
            dataset=test.dataset, task=args.task, device=device)
    
    print(f"\n{'='*60}")
    print("✓ Inference complete!")
    print(f"{'='*60}\n")