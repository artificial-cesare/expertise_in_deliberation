import json
import pandas as pd
from pathlib import Path


def load_sampled_speeches(json_path):
    """Load the sampled speeches from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['speeches']


def load_annotations(csv_dir):
    """Load all annotation CSV files from directory."""
    csv_dir = Path(csv_dir)
    annotations = []

    for csv_file in csv_dir.glob('*.csv'):
        df = pd.read_csv(csv_file)
        annotations.append(df)

    return pd.concat(annotations, ignore_index=True)


def prepare_ratings_matrix(speeches, annotations):
    """Prepare a matrix of ratings for each speech."""
    speech_ids = [speech['speech_id'] for speech in speeches]
    raters = annotations['rater_id'].unique().tolist()
    raters.append('llm')

    ratings_matrix = pd.DataFrame(index=speech_ids, columns=raters, dtype=float)

    for _, row in annotations.iterrows():
        ratings_matrix.loc[row['item_id'], row['rater_id']] = float(row['score'])

    for speech in speeches:
        ratings_matrix.loc[speech['speech_id'], 'llm'] = float(speech['expertise_score'])

    return ratings_matrix
