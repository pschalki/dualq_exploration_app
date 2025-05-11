# data_loader.py
import pandas as pd
import pickle
import ast # For parsing stringified dicts
import os

BASE_PATH = 'results/final_results/'

PARTY_COLORS = {
    'CDU': '#000000',
    'CSU': '#008AC5',
    'SPD': '#EB001F',
    'AfD': '#009EE0',
    'FDP': '#FFEE00',
    'Die Linke': '#BE3075',
    'Grüne': '#64A12D', 
    'DIE LINKE': '#BE3075', 
    'GRÜNE': '#64A12D', 
    'Sonstige': '#AAAAAA' 
}
DEFAULT_COLOR = '#CCCCCC' 

def load_data():
    data = {}
    try:
        data['participant_overall'] = pd.read_csv(os.path.join(BASE_PATH, 'participant_data_overall.csv'))
        data['participant_topic'] = pd.read_csv(os.path.join(BASE_PATH, 'participant_data_per_topic.csv'))
        data['cluster_overall'] = pd.read_csv(os.path.join(BASE_PATH, 'cluster_data_overall.csv'))
        data['cluster_topic'] = pd.read_csv(os.path.join(BASE_PATH, 'cluster_data_per_topic.csv'))

        for df_name in data.keys():
            if 'cluster_id' in data[df_name].columns:
                data[df_name]['cluster_id'] = data[df_name]['cluster_id'].astype(str)
            if 'interview_id' in data[df_name].columns: 
                 data[df_name]['interview_id'] = data[df_name]['interview_id'].astype(str)
        
        # --- MODIFICATION START: Add 'label' to cluster_topic ---
        if 'cluster_overall' in data and 'cluster_topic' in data and \
           'label' in data['cluster_overall'].columns and \
           'cluster_id' in data['cluster_overall'].columns and \
           'cluster_id' in data['cluster_topic'].columns:
            
            cluster_labels = data['cluster_overall'][['cluster_id', 'label']].drop_duplicates()
            data['cluster_topic'] = pd.merge(data['cluster_topic'], cluster_labels, on='cluster_id', how='left')
        # --- MODIFICATION END ---

        with open(os.path.join(BASE_PATH, 'participant_data_overall_embeddings.pkl'), 'rb') as f:
            data['participant_overall_emb'] = pickle.load(f)
        with open(os.path.join(BASE_PATH, 'participant_data_per_topic_embeddings.pkl'), 'rb') as f:
            data['participant_topic_emb'] = pickle.load(f)
        with open(os.path.join(BASE_PATH, 'cluster_data_overall_embeddings.pkl'), 'rb') as f:
            data['cluster_overall_emb'] = pickle.load(f)
        with open(os.path.join(BASE_PATH, 'cluster_data_per_topic_embeddings.pkl'), 'rb') as f:
            data['cluster_topic_emb'] = pickle.load(f)
        with open(os.path.join(BASE_PATH, 'semantic_axes_dict.pkl'), 'rb') as f:
            data['semantic_axes'] = pickle.load(f)

        for df_name in ['cluster_overall', 'cluster_topic']:
            if df_name in data and 'wahlabsicht_verteilung' in data[df_name].columns: # Check if df_name exists
                data[df_name]['wahlabsicht_verteilung_parsed'] = \
                    data[df_name]['wahlabsicht_verteilung'].apply(
                        lambda x: ast.literal_eval(x) if pd.notnull(x) and isinstance(x, str) and x.startswith('{') else {}
                    )
        return data
    except FileNotFoundError as e:
        print(f"ERROR: Data file not found: {e}. Please ensure all data files are in '{BASE_PATH}'.")
        print("If you're running for the first time, you might need to run 'create_dummy_data.py' first.")
        return None 
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None


def get_topic_options(participant_topic_df):
    if participant_topic_df is None or 'topic' not in participant_topic_df.columns:
        return [{"label": "Gesamtes Interview", "value": "Gesamtes Interview"}]
    topics = sorted(list(participant_topic_df['topic'].unique()))
    return [{"label": "Gesamtes Interview", "value": "Gesamtes Interview"}] + \
           [{"label": topic, "value": topic} for topic in topics]

def get_semantic_axis_options(semantic_axes_dict):
    if semantic_axes_dict is None:
        return []
    return [{"label": axis_name, "value": axis_name} for axis_name in semantic_axes_dict.keys()]

def get_coloring_options():
    options = ['cluster_id', 'geschlecht', 'wahlabsicht', 'einkommen'] 
    return [{"label": col.replace("_", " ").title(), "value": col} for col in options]