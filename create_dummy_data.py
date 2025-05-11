# create_dummy_data.py
import pandas as pd
import numpy as np
import pickle
import os

# Create directory if it doesn't exist
DUMMY_DATA_DIR = 'results/final_results/'
os.makedirs(DUMMY_DATA_DIR, exist_ok=True)

# --- Dummy DataFrames ---
# participant_data_overall
p_overall_data = {
    'interview_id': [f'P{i}' for i in range(1, 6)],
    'alter': [25, 35, 45, 30, 50],
    'geschlecht': ['männlich', 'weiblich', 'männlich', 'weiblich', 'männlich'],
    'einkommen': ['1000-2000€', '2000-3000€', '3000-5000€', '1000-2000€', '>5000€'],
    'beruf': ['Student', 'Angestellte', 'Manager', 'Künstler', 'Ingenieur'],
    'wahlabsicht': ['Grüne', 'SPD', 'CDU', 'Die Linke', 'FDP'],
    'cluster_id': ['0', '1', '0', '2', '1'], # String IDs
    'overall_summary': [f'Overall summary for P{i}' for i in range(1, 6)],
    'umap_x': np.random.rand(5) * 10,
    'umap_y': np.random.rand(5) * 10,
}
participant_data_overall_df = pd.DataFrame(p_overall_data)
participant_data_overall_df.to_csv(f'{DUMMY_DATA_DIR}participant_data_overall.csv', index=False)

# participant_data_per_topic
topics = ['Topic A', 'Topic B']
p_topic_data = []
for i in range(1, 6):
    for topic in topics:
        p_topic_data.append({
            'interview_id': f'P{i}', 'alter': p_overall_data['alter'][i-1], # ... other demographics
            'geschlecht': p_overall_data['geschlecht'][i-1],
            'einkommen': p_overall_data['einkommen'][i-1],
            'beruf': p_overall_data['beruf'][i-1],
            'wahlabsicht': p_overall_data['wahlabsicht'][i-1],
            'cluster_id': p_overall_data['cluster_id'][i-1],
            'topic': topic,
            'topic_summary': f'{topic} summary for P{i}',
            'umap_x': np.random.rand() * 10,
            'umap_y': np.random.rand() * 10,
        })
participant_data_per_topic_df = pd.DataFrame(p_topic_data)
participant_data_per_topic_df.to_csv(f'{DUMMY_DATA_DIR}participant_data_per_topic.csv', index=False)

# cluster_data_overall
c_overall_data = {
    'cluster_id': ['0', '1', '2'], # String IDs
    'label': ['Cluster Zero', 'Cluster One', 'Cluster Two'],
    'overall_summary': [f'Overall summary for Cluster {i}' for i in ['0', '1', '2']],
    'umap_x': np.random.rand(3) * 5,
    'umap_y': np.random.rand(3) * 5,
    'anzahl_teilnehmer': [2, 2, 1],
    'frauenanteil': [0.0, 0.5, 0.0],
    'durchschnittsalter': [35, 40, 30],
    'durchschnittseinkommen': [2000, 3500, 1500],
    'wahlabsicht_verteilung': ["{'Grüne': 0.5, 'CDU': 0.5}", "{'SPD': 0.5, 'FDP': 0.5}", "{'Die Linke': 1.0}"]
}
cluster_data_overall_df = pd.DataFrame(c_overall_data)
cluster_data_overall_df.to_csv(f'{DUMMY_DATA_DIR}cluster_data_overall.csv', index=False)

# cluster_data_per_topic
c_topic_data = []
for cid in ['0', '1', '2']:
    for topic in topics:
        c_topic_data.append({
            'cluster_id': cid, 'topic': topic,
            'topic_summary': f'{topic} summary for Cluster {cid}',
            'umap_x': np.random.rand() * 5, 'umap_y': np.random.rand() * 5,
            'anzahl_teilnehmer': c_overall_data['anzahl_teilnehmer'][int(cid)], # simplified
             # ... other demographics from c_overall_data
            'frauenanteil': c_overall_data['frauenanteil'][int(cid)],
            'durchschnittsalter': c_overall_data['durchschnittsalter'][int(cid)],
            'durchschnittseinkommen': c_overall_data['durchschnittseinkommen'][int(cid)],
            'wahlabsicht_verteilung': c_overall_data['wahlabsicht_verteilung'][int(cid)]
        })
cluster_data_per_topic_df = pd.DataFrame(c_topic_data)
cluster_data_per_topic_df.to_csv(f'{DUMMY_DATA_DIR}cluster_data_per_topic.csv', index=False)

# --- Dummy Embeddings (very small dimension for simplicity) ---
EMBEDDING_DIM = 5
participant_data_overall_embeddings = {f'P{i}': np.random.rand(EMBEDDING_DIM) for i in range(1, 6)}
participant_data_per_topic_embeddings = {
    f'P{i}': {topic: np.random.rand(EMBEDDING_DIM) for topic in topics} for i in range(1, 6)
}
cluster_data_overall_embeddings = {str(i): np.random.rand(EMBEDDING_DIM) for i in range(3)}
cluster_data_per_topic_embeddings = {
    str(i): {topic: np.random.rand(EMBEDDING_DIM) for topic in topics} for i in range(3)
}

with open(f'{DUMMY_DATA_DIR}participant_data_overall_embeddings.pkl', 'wb') as f: pickle.dump(participant_data_overall_embeddings, f)
with open(f'{DUMMY_DATA_DIR}participant_data_per_topic_embeddings.pkl', 'wb') as f: pickle.dump(participant_data_per_topic_embeddings, f)
with open(f'{DUMMY_DATA_DIR}cluster_data_overall_embeddings.pkl', 'wb') as f: pickle.dump(cluster_data_overall_embeddings, f)
with open(f'{DUMMY_DATA_DIR}cluster_data_per_topic_embeddings.pkl', 'wb') as f: pickle.dump(cluster_data_per_topic_embeddings, f)

# Dummy semantic_axes_dict
semantic_axes_dict = {
    'Links/Rechts': {
        'X_Name': 'Links', 'Y_Name': 'Rechts',
        'Description_X': 'Beschreibung für Links.', 'Description_Y': 'Beschreibung für Rechts.',
        'Embedding_X': np.random.rand(EMBEDDING_DIM), 'Embedding_Y': np.random.rand(EMBEDDING_DIM)
    },
    'Progressiv/Konservativ': {
        'X_Name': 'Progressiv', 'Y_Name': 'Konservativ',
        'Description_X': 'Beschreibung für Progressiv.', 'Description_Y': 'Beschreibung für Konservativ.',
        'Embedding_X': np.random.rand(EMBEDDING_DIM), 'Embedding_Y': np.random.rand(EMBEDDING_DIM)
    }
}
with open(f'{DUMMY_DATA_DIR}semantic_axes_dict.pkl', 'wb') as f: pickle.dump(semantic_axes_dict, f)

print(f"Dummy data created in {DUMMY_DATA_DIR}")