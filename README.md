# Expertise in Deliberation

A comprehensive research project analyzing expertise levels in Citizens' Assemblies deliberations. This project combines NLP techniques, LLM-based analysis, and embedding methods to study how expertise manifests in deliberative speech.

## Overview

This project investigates expertise signals in deliberative democracy transcripts from Citizens' Assemblies (8th Amendment in Ireland, Gender Equality, and Scottish assemblies). Only the publicly available transcripts from the Irish Citizens' Assemblies and expert testimonies are included here in 'data/raw/', to request access to the
transcripts from the Scottish Assembly, please request it at the [UK Data Archive](https://committees.parliament.uk/publications/oral-evidence/). Original expert files can be saved from [UK Parliament Oral Evidence Archive](https://committees.parliament.uk/publications/oral-evidence/).


## Project Structure

```
expertise_in_deliberation/
├── code/                           # Source code
│   ├── ArgQualityAdapters/         # Argument quality scoring
│   ├── correlation_analysis/       # Statistical analysis
│   ├── embeddings/                 # Embedding-based analysis
│   ├── handcrafted_features/       # Linguistic feature extraction
│   ├── LLM_expertise/              # LLM-based expertise scoring
│   └── preprocessing/              # Data preprocessing
├── data/                           # Data files
│   ├── raw/                        # Raw Assemblies and expert transcripts 
│   └── vocab/                      # Domain vocabulary sources
├── llm_exp_requirements.txt        # LLM analysis dependencies
└── qwen3_requirements.txt          # Embedding model dependencies
```

## Components

### 1. Preprocessing (`code/preprocessing/`)

**`preprocess_llm.py`**: Multi-format speech extraction tool that processes Citizens' Assembly transcripts from PDF, RTF, and HTML files using a two-pass LLM pipeline:
- Pass 1: Cleans OCR errors and standardizes speaker labels
- Pass 2: Extracts structured speaker/speech data
- Outputs CSV files with date, speaker, and text columns

### 2. Handcrafted Features (`code/handcrafted_features/`)

**`new_feats_extract.py`**: Extracts 14 linguistic features from transcripts:
- **Jargon density**: Domain-specific terminology usage
- **Lexical sophistication**: Word rarity based on Zipf frequency
- **Syntactic complexity**: Subordination and nominalization patterns
- **Epistemic markers**: Hedges and boosters (e.g., "may", "certainly")
- **Reasoning markers**: Causal, conditional, and syllogistic patterns
- **Evidential markers**: Citation and evidence language
- **Specificity**: Length, word complexity, numerical content

**`dom_terms_final_exp.py`**: Extracts domain-specific jargon from expert HTML documents with:
- OCR error detection and correction
- Enrichment scoring (corpus frequency vs. general English)
- Multi-word term extraction (n-grams, noun phrases)

**`filter_function.py`**: Filters substantive speech acts (>5 content words)

### 3. LLM Expertise Scoring (`code/LLM_expertise/`)

**`llm_expert_extract_new.py`**: Uses GPT-4o-mini to score expertise on a 1-5 Likert scale based on:
- Domain-specific vocabulary usage
- Structured reasoning patterns
- Evidence and citation markers
- Epistemic calibration
- Syntactic and lexical complexity

### 4. Argument Quality (`code/ArgQualityAdapters/`)

Compared to the original ArgQuality framework, only the inference script was edited to fit the current project structure. Full details about the ArgQuality framework can be found in its [GitHub repository](https://github.com/Blubberli/ArgQualityAdapters/tree/master). 

**`inference_simple.py`**: Runs argument quality inference using adapter-based RoBERTa models from the ArgQuality framework. Scores overall argumentation quality of speech acts.

### 5. Embeddings (`code/embeddings/`)

**`final_finetune.py`**: Fine-tunes Qwen3-Embedding-0.6B using contrastive learning:
- Multiple Negatives Ranking Loss
- Training on expert speech pairs from same topics
- Evaluation via Retrieval@K and topic separation metrics

**`k_means_exp.py`**: K-Means clustering analysis:
- Global PCA dimensionality reduction (100 components)
- Best-k selection based on expert classification accuracy
- Distance calculation to topic centroids
- UMAP visualization of embedding spaces and centroids

**`plotting.py`**: Visualization utilities for embedding analysis

### 6. Correlation Analysis (`code/correlation_analysis/`)

**`corr.py`**: Comprehensive correlation analysis:
- Pearson/Spearman correlations between all features
- Z-score and Min-Max normalization options
- Temporal evolution analysis (start/middle/end of deliberations)
- Output: heatmaps, p-value matrices, JSON data

**`stats_time_final.py`**: Time-series statistical analysis

## Data

### Raw Data (`data/raw/`)
- **8th_amendment/**: Irish Constitutional Convention transcripts 
- **gendereq/**: Gender Equality assembly transcripts
- **experts/**: Expert testimony HTML files organized by topic:
  - `constitution/`: Constitutional reform experts
  - `future/`: Public policy experts
  - `gender/`: Gender equality experts

### Vocabulary Sources (`data/vocab/`)
HTML documents used to build domain-specific vocabulary for each topic.

## Installation

### For LLM-based Analysis
```bash
pip install -r llm_exp_requirements.txt
python -m spacy download en_core_web_sm
```

### For Embedding Analysis (GPU required)
```bash
pip install -r qwen3_requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### 1. Preprocess Transcripts
```bash
export OPENAI_API_KEY='your-key'
python code/preprocessing/preprocess_llm.py \
    --base-dir data/raw \
    --output-dir data/assemblies/processed_files
```

### 2. Extract Domain Vocabulary
```bash
python code/handcrafted_features/dom_terms_final_exp.py \
    --vocab_dir data/vocab \
    --output code/handcrafted_features/domain_terms_clean_exp.json
```

### 3. Extract Linguistic Features
```bash
python code/handcrafted_features/new_feats_extract.py \
    --processed_dir data/assemblies/processed_files
```

### 4. Score Expertise with LLM
```bash
export OPENAI_API_KEY='your-key'
python code/LLM_expertise/llm_expert_extract_new.py \
    --base-dir data/assemblies/processed_files
```

### 5. Run Argument Quality Scoring
```bash
python code/ArgQualityAdapters/code/inference_simple.py \
    <input_csv> text <batch_size> overall <output_csv>
```

### 6. Fine-tune Embeddings
```bash
python code/embeddings/final_finetune.py
```

### 7. Run Clustering Analysis
```bash
python code/embeddings/k_means_exp.py
```

### 8. Run Correlation Analysis
```bash
python code/correlation_analysis/corr.py
```

## Key Features

| Feature | Description | Range |
|---------|-------------|-------|
| `jargon_density` | Domain terminology per token | 0-1 |
| `lexical_sophistication` | Inverse Zipf frequency | 0-1 |
| `syntactic_complexity` | Subordination + nominalization | 0+ |
| `hedge_density` | Uncertainty markers (%) | 0-100 |
| `booster_density` | Certainty markers (%) | 0-100 |
| `epistemic_balance` | Hedge/booster balance | 0-1 |
| `reasoning_markers` | Logical connectives (%) | 0-100 |
| `evidential_markers` | Citation patterns (%) | 0-100 |
| `expertise_llm_score` | LLM expertise rating | 1-5 |
| `overall_score` | Argument quality score | 0-1 |

## Topic Mappings

| Assembly | Expert Domain | Display Name |
|----------|---------------|--------------|
| 8th_amendment | constitution | Constitutional Reform |
| gendereq | gender | Gender Equality |
| scotland | future | Public Policy |

## Output Files

The pipeline generates:
- Processed CSVs with extracted features and scores
- Correlation heatmaps and p-value matrices
- UMAP plots of embedding spaces
- JSON files with domain vocabulary
- Training metrics and model checkpoints

## License

MIT License

Copyright (c) 2025 Cesare Maria Dalbagno

## Authors

Cesare Maria Dalbagno
