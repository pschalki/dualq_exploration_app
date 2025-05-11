# app.py
import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go # Added for go.Figure in empty cases

# Import your custom modules
from data_loader import load_data, get_topic_options, get_semantic_axis_options, get_coloring_options
from plotting_utils import create_cluster_plot, create_participant_plot, project_embedding
from layout_components import generate_cluster_details_html, generate_participant_details_html, generate_semantic_axis_description_html

# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
server = app.server

# --- Load Data ---
ALL_DATA = load_data()
if ALL_DATA is None:
    print("Failed to load data. Exiting.")
    # You might want to display an error in the Dash app itself if ALL_DATA is None
    # For now, we exit, but in a production app, handle this more gracefully.
    # app.layout = html.Div("Error loading data. Please check logs.")
    # if __name__ == '__main__':
    #     app.run_server(debug=True)
    exit()

# --- Prepare Dropdown Options ---
TOPIC_OPTIONS = get_topic_options(ALL_DATA.get('participant_topic')) # Use .get for safety
SEMANTIC_AXIS_OPTIONS = get_semantic_axis_options(ALL_DATA.get('semantic_axes'))
COLORING_OPTIONS = get_coloring_options()

# --- App Layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("DualQ Interview Analysis Dashboard"), width=12), className="mb-3 mt-3"),
    
    dcc.Store(id='selected-cluster-store'),
    dcc.Store(id='selected-participant-store'),

    dbc.Tabs(id="main-tabs", active_tab='tab-cluster', children=[
        # --- CLUSTER TAB ---
        dbc.Tab(label='Cluster-Based Visuals', tab_id='tab-cluster', children=[
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='cluster-topic-dropdown', options=TOPIC_OPTIONS, value="Gesamtes Interview", clearable=False), width=4),
                dbc.Col(dcc.Dropdown(id='cluster-x-axis-dropdown', options=SEMANTIC_AXIS_OPTIONS, placeholder="X-Achse wählen (optional)", clearable=True), width=3),
                dbc.Col(dcc.Dropdown(id='cluster-y-axis-dropdown', options=SEMANTIC_AXIS_OPTIONS, placeholder="Y-Achse wählen (optional)", clearable=True), width=3),
                dbc.Col(dbc.Button("Auswahl zurücksetzen", id="cluster-clear-selection-btn", color="secondary", outline=True, size="sm"), width=2, className="d-flex align-items-center")
            ], className="mt-3 mb-3 align-items-center"),
            
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id='participant-scatter-plot')), width=8),
                dbc.Col(dcc.Loading(html.Div(id='participant-details-panel', children="Wählen Sie einen Teilnehmer für Details.")), width=4)
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='cluster-x-axis-desc'), width=6),
                dbc.Col(html.Div(id='cluster-y-axis-desc'), width=6)
            ], className="mt-2")
        ]),

        # --- PARTICIPANT TAB ---
        dbc.Tab(label='Participant-Based Visuals', tab_id='tab-participant', children=[
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='participant-topic-dropdown', options=TOPIC_OPTIONS, value="Gesamtes Interview", clearable=False), width=3),
                dbc.Col(dcc.Dropdown(id='participant-coloring-dropdown', options=COLORING_OPTIONS, placeholder="Einfärbung wählen", value=COLORING_OPTIONS[0]['value'] if COLORING_OPTIONS else None), width=2),
                dbc.Col(dcc.Dropdown(id='participant-x-axis-dropdown', options=SEMANTIC_AXIS_OPTIONS, placeholder="X-Achse wählen (optional)", clearable=True), width=3),
                dbc.Col(dcc.Dropdown(id='participant-y-axis-dropdown', options=SEMANTIC_AXIS_OPTIONS, placeholder="Y-Achse wählen (optional)", clearable=True), width=2),
                dbc.Col(dbc.Button("Auswahl zurücksetzen", id="participant-clear-selection-btn", color="secondary", outline=True, size="sm"), width=2, className="d-flex align-items-center")
            ], className="mt-3 mb-3 align-items-center"),

            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id='participant-scatter-plot')), width=8),
                dbc.Col(dcc.Loading(html.Div(id='participant-details-panel', children="Wählen Sie einen Teilnehmer für Details.")), width=4) # Initial text
            ]),
             dbc.Row([
                dbc.Col(html.Div(id='participant-x-axis-desc'), width=6),
                dbc.Col(html.Div(id='participant-y-axis-desc'), width=6)
            ], className="mt-2")
        ]),
    ])
], fluid=True)


# --- Helper function for plot data prep ---
def get_plot_axes_data(df, emb_dict, selected_topic, sel_x_axis, sel_y_axis, id_col): # Removed data_id_col_in_emb
    x_coords, y_coords = [], []
    x_label, y_label = "UMAP X", "UMAP Y"
    
    # Ensure emb_dict is not None
    if emb_dict is None: emb_dict = {}
    
    use_semantic = sel_x_axis and sel_y_axis and ALL_DATA.get('semantic_axes')
    
    if use_semantic:
        semantic_axes_info = ALL_DATA.get('semantic_axes', {})
        x_axis_info = semantic_axes_info.get(sel_x_axis, {})
        y_axis_info = semantic_axes_info.get(sel_y_axis, {})

        x_label = x_axis_info.get('X_Name', sel_x_axis) + " / " + x_axis_info.get('Y_Name', '')
        y_label = y_axis_info.get('X_Name', sel_y_axis) + " / " + y_axis_info.get('Y_Name', '')
        
        sem_ax_x_emb_x = x_axis_info.get('Embedding_X')
        sem_ax_x_emb_y = x_axis_info.get('Embedding_Y')
        sem_ax_y_emb_x = y_axis_info.get('Embedding_X')
        sem_ax_y_emb_y = y_axis_info.get('Embedding_Y')

        if any(e is None for e in [sem_ax_x_emb_x, sem_ax_x_emb_y, sem_ax_y_emb_x, sem_ax_y_emb_y]):
            use_semantic = False # Fallback to UMAP if embeddings are missing for selected axes
            x_label, y_label = "UMAP X (Axis Emb. Missing)", "UMAP Y (Axis Emb. Missing)"


    for _, row in df.iterrows():
        item_id = str(row[id_col]) 
        embedding = None
        
        if selected_topic == "Gesamtes Interview":
            embedding = emb_dict.get(item_id)
        else: 
            topic_emb_data = emb_dict.get(item_id, {})
            if isinstance(topic_emb_data, dict): # Check if it's a dict of topics
                 embedding = topic_emb_data.get(selected_topic)

        if use_semantic and embedding is not None:
            x_coords.append(project_embedding(embedding, sem_ax_x_emb_x, sem_ax_x_emb_y))
            y_coords.append(project_embedding(embedding, sem_ax_y_emb_x, sem_ax_y_emb_y))
        elif 'umap_x' in row and 'umap_y' in row: # Fallback to UMAP if not using semantic or embedding is None
            x_coords.append(row['umap_x'])
            y_coords.append(row['umap_y'])
        else: # If no UMAP data either, append NaN
            x_coords.append(np.nan)
            y_coords.append(np.nan)
            
    return pd.Series(x_coords, index=df.index), pd.Series(y_coords, index=df.index), x_label, y_label


# --- Callbacks for Cluster Tab ---
@app.callback(
    Output('cluster-scatter-plot', 'figure'),
    Output('cluster-x-axis-desc', 'children'),
    Output('cluster-y-axis-desc', 'children'),
    Output('cluster-details-panel', 'children', allow_duplicate=True),
    Input('cluster-topic-dropdown', 'value'),
    Input('cluster-x-axis-dropdown', 'value'),
    Input('cluster-y-axis-dropdown', 'value'),
    Input('selected-cluster-store', 'data'),
    prevent_initial_call='initial_duplicate' # MODIFIED
)
def update_cluster_view(selected_topic, sel_x_axis, sel_y_axis, selected_cluster_id):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered and ctx.triggered[0]['prop_id'] != '.' else None

    df_overall = ALL_DATA.get('cluster_overall')
    df_topic = ALL_DATA.get('cluster_topic')
    emb_overall = ALL_DATA.get('cluster_overall_emb')
    emb_topic = ALL_DATA.get('cluster_topic_emb')

    if df_overall is None or df_topic is None: # Check if data loaded
        return go.Figure().update_layout(title="Clusterdaten nicht geladen"), "", "", "Datenfehler."

    if selected_topic == "Gesamtes Interview":
        current_df = df_overall.copy()
        current_emb_dict = emb_overall
        title_suffix = " (Gesamtes Interview)"
    else:
        current_df = df_topic[df_topic['topic'] == selected_topic].copy()
        current_emb_dict = emb_topic
        title_suffix = f" ({selected_topic})"
    
    if current_df.empty:
        return go.Figure().update_layout(title="Keine Daten für diese Auswahl"), "", "", "Keine Clusterdaten für diese Auswahl."

    x_coords, y_coords, x_label, y_label = get_plot_axes_data(
        current_df, current_emb_dict, selected_topic, sel_x_axis, sel_y_axis, 'cluster_id'
    )
    current_df['plot_x'] = x_coords
    current_df['plot_y'] = y_coords
    
    current_df.dropna(subset=['plot_x', 'plot_y'], inplace=True) # Drop if projection failed
    if current_df.empty:
         return go.Figure().update_layout(title="Keine plotbaren Daten nach Projektion"), "", "", "Keine plotbaren Clusterdaten."

    fig = create_cluster_plot(
        current_df, 'plot_x', 'plot_y',
        size_col='anzahl_teilnehmer', color_col='cluster_id', text_col='label',
        custom_data_cols=['cluster_id', 'label', 'anzahl_teilnehmer'],
        title="Cluster-Visualisierung" + title_suffix,
        selected_cluster_id=selected_cluster_id,
        hover_name_col='label',
        x_axis_label=x_label, y_axis_label=y_label
    )
    
    x_desc = generate_semantic_axis_description_html(sel_x_axis, ALL_DATA.get('semantic_axes'), 'X') if sel_x_axis and sel_y_axis else ""
    y_desc = generate_semantic_axis_description_html(sel_y_axis, ALL_DATA.get('semantic_axes'), 'Y') if sel_x_axis and sel_y_axis else ""
    
    details_panel_content = no_update
    # If it's an initial call or major filters change, reset the details panel
    if not triggered_id or triggered_id in ['cluster-topic-dropdown', 'cluster-x-axis-dropdown', 'cluster-y-axis-dropdown']:
        details_panel_content = "Wählen Sie einen Cluster für Details."
        if selected_cluster_id and (triggered_id is None or triggered_id == 'selected-cluster-store'): # Restore details if only selection changed on initial load
             # This case might be complex, for now, reset. Click handler will repopulate.
             pass


    return fig, x_desc, y_desc, details_panel_content

@app.callback(
    Output('cluster-details-panel', 'children', allow_duplicate=True), # Keep allow_duplicate
    Output('selected-cluster-store', 'data', allow_duplicate=True), # Keep allow_duplicate
    Output('cluster-scatter-plot', 'figure', allow_duplicate=True),
    Input('cluster-scatter-plot', 'clickData'),
    State('cluster-topic-dropdown', 'value'),
    State('selected-cluster-store', 'data'),
    State('cluster-scatter-plot', 'figure'), 
    prevent_initial_call=True
)
def display_cluster_click_data(clickData, selected_topic, current_selected_id, current_fig_state):
    if clickData is None or not clickData['points']:
        return no_update, no_update, no_update

    clicked_cluster_id = str(clickData['points'][0]['customdata'][0]) 

    if clicked_cluster_id == current_selected_id:
        new_selected_id = None
        details_html = "Wählen Sie einen Cluster für Details."
    else:
        new_selected_id = clicked_cluster_id
        df_overall = ALL_DATA.get('cluster_overall')
        df_topic = ALL_DATA.get('cluster_topic')
        
        cluster_info_df = pd.DataFrame() # Default to empty
        if selected_topic == "Gesamtes Interview" and df_overall is not None:
            cluster_info_df = df_overall[df_overall['cluster_id'] == new_selected_id]
        elif df_topic is not None:
            cluster_info_df = df_topic[(df_topic['cluster_id'] == new_selected_id) & (df_topic['topic'] == selected_topic)]
        
        details_html = generate_cluster_details_html(cluster_info_df, selected_topic)

    updated_fig_dict = no_update
    if current_fig_state:
        updated_fig = go.Figure(current_fig_state) # Create a new figure object from state
        for trace in updated_fig.data:
            if hasattr(trace, 'name'): # Assuming trace.name is cluster_id
                opacity = 0.8 # Default
                if new_selected_id:
                    opacity = 1.0 if trace.name == new_selected_id else 0.2
                if hasattr(trace, 'marker') and hasattr(trace.marker, 'opacity'):
                    trace.marker.opacity = opacity
        updated_fig_dict = updated_fig.to_dict() # Return as dict
    
    return details_html, new_selected_id, updated_fig_dict


@app.callback(
    Output('selected-cluster-store', 'data', allow_duplicate=True),
    Output('cluster-details-panel', 'children', allow_duplicate=True),
    Output('cluster-scatter-plot', 'figure', allow_duplicate=True),
    Input('cluster-clear-selection-btn', 'n_clicks'),
    State('cluster-scatter-plot', 'figure'),
    prevent_initial_call=True
)
def clear_cluster_selection(n_clicks, current_fig_state):
    if n_clicks is None:
        return no_update, no_update, no_update
    
    details_html = "Wählen Sie einen Cluster für Details."
    updated_fig_dict = no_update
    if current_fig_state:
        updated_fig = go.Figure(current_fig_state)
        for trace in updated_fig.data:
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'opacity'):
                 trace.marker.opacity = 0.8 
        updated_fig_dict = updated_fig.to_dict()
            
    return None, details_html, updated_fig_dict

# --- Callbacks for Participant Tab ---
@app.callback(
    Output('participant-scatter-plot', 'figure'),
    Output('participant-x-axis-desc', 'children'),
    Output('participant-y-axis-desc', 'children'),
    Output('participant-details-panel', 'children', allow_duplicate=True),
    Input('participant-topic-dropdown', 'value'),
    Input('participant-coloring-dropdown', 'value'),
    Input('participant-x-axis-dropdown', 'value'),
    Input('participant-y-axis-dropdown', 'value'),
    Input('selected-participant-store', 'data'),
    prevent_initial_call='initial_duplicate' # MODIFIED
)
def update_participant_view(selected_topic, coloring_var, sel_x_axis, sel_y_axis, selected_participant_id):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered and ctx.triggered[0]['prop_id'] != '.' else None

    df_overall = ALL_DATA.get('participant_overall')
    df_topic = ALL_DATA.get('participant_topic')
    emb_overall = ALL_DATA.get('participant_overall_emb')
    emb_topic = ALL_DATA.get('participant_topic_emb')

    if df_overall is None or df_topic is None:
        return go.Figure().update_layout(title="Teilnehmerdaten nicht geladen"), "", "", "Datenfehler."

    current_df = df_overall.copy() # Always start with overall for full participant list & demographics
    current_emb_dict = emb_overall # Default to overall embeddings
    title_suffix = " (Gesamtes Interview)"

    if selected_topic != "Gesamtes Interview":
        topic_data = df_topic[df_topic['topic'] == selected_topic][['interview_id', 'umap_x', 'umap_y', 'topic_summary']]
        # Merge topic-specific UMAP and summary, keeping all participants from overall
        current_df = pd.merge(current_df, topic_data, on='interview_id', how='left', suffixes=('', '_topic_specific'))
        
        # Use topic UMAP if available, else fallback to overall UMAP
        current_df['umap_x'] = current_df['umap_x_topic_specific'].fillna(current_df['umap_x'])
        current_df['umap_y'] = current_df['umap_y_topic_specific'].fillna(current_df['umap_y'])
        # Topic summary will be used in details panel, ensure column exists
        if 'topic_summary' not in current_df.columns and 'topic_summary_topic_specific' in current_df.columns:
             current_df.rename(columns={'topic_summary_topic_specific': 'topic_summary'}, inplace=True)
        elif 'topic_summary' not in current_df.columns: # If still not there
            current_df['topic_summary'] = pd.NA


        current_emb_dict = emb_topic # Use topic embeddings for projection
        title_suffix = f" ({selected_topic})"
    
    if current_df.empty:
        return go.Figure().update_layout(title="Keine Daten für diese Auswahl"), "", "", "Keine Teilnehmerdaten für diese Auswahl."
    
    x_coords, y_coords, x_label, y_label = get_plot_axes_data(
        current_df, current_emb_dict, selected_topic, sel_x_axis, sel_y_axis, 'interview_id'
    )
    current_df['plot_x'] = x_coords
    current_df['plot_y'] = y_coords
    current_df.dropna(subset=['plot_x', 'plot_y', 'interview_id'], inplace=True) # Ensure ID is also not NaN
    if current_df.empty:
         return go.Figure().update_layout(title="Keine plotbaren Daten nach Projektion"), "", "", "Keine plotbaren Teilnehmerdaten."

    fig = create_participant_plot(
        current_df, 'plot_x', 'plot_y',
        color_by_col=coloring_var,
        custom_data_cols=['interview_id', coloring_var if coloring_var in current_df.columns else 'interview_id'],
        title="Teilnehmer-Visualisierung" + title_suffix,
        selected_participant_id=selected_participant_id,
        hover_name_col='interview_id',
        x_axis_label=x_label, y_axis_label=y_label
    )
    
    x_desc = generate_semantic_axis_description_html(sel_x_axis, ALL_DATA.get('semantic_axes'), 'X') if sel_x_axis and sel_y_axis else ""
    y_desc = generate_semantic_axis_description_html(sel_y_axis, ALL_DATA.get('semantic_axes'), 'Y') if sel_x_axis and sel_y_axis else ""

    details_panel_content = no_update
    if not triggered_id or triggered_id in ['participant-topic-dropdown', 'participant-coloring-dropdown', 'participant-x-axis-dropdown', 'participant-y-axis-dropdown']:
        details_panel_content = "Wählen Sie einen Teilnehmer für Details."

    return fig, x_desc, y_desc, details_panel_content


@app.callback(
    Output('participant-details-panel', 'children', allow_duplicate=True),
    Output('selected-participant-store', 'data', allow_duplicate=True),
    Output('participant-scatter-plot', 'figure', allow_duplicate=True),
    Input('participant-scatter-plot', 'clickData'),
    State('participant-topic-dropdown', 'value'),
    State('selected-participant-store', 'data'),
    State('participant-scatter-plot', 'figure'),
    prevent_initial_call=True
)
def display_participant_click_data(clickData, selected_topic, current_selected_id, current_fig_state):
    if clickData is None or not clickData['points']:
        return no_update, no_update, no_update

    # Assuming 'interview_id' is the first element in customdata
    clicked_participant_id = str(clickData['points'][0]['customdata'][0]) 
    
    if clicked_participant_id == current_selected_id:
        new_selected_id = None
        details_html = "Wählen Sie einen Teilnehmer für Details."
    else:
        new_selected_id = clicked_participant_id
        participant_df_overall = ALL_DATA.get('participant_overall')
        participant_df_topic = ALL_DATA.get('participant_topic')
        
        participant_info = pd.DataFrame() # Default
        if participant_df_overall is not None:
            participant_info = participant_df_overall[participant_df_overall['interview_id'] == new_selected_id].copy()
        
        if not participant_info.empty and selected_topic != "Gesamtes Interview" and participant_df_topic is not None:
            topic_summary_series = participant_df_topic[
                (participant_df_topic['interview_id'] == new_selected_id) &
                (participant_df_topic['topic'] == selected_topic)
            ]['topic_summary']
            if not topic_summary_series.empty:
                 participant_info.loc[:, 'topic_summary'] = topic_summary_series.iloc[0]
            else:
                 participant_info.loc[:, 'topic_summary'] = "Keine themenspezifische Zusammenfassung."
        elif not participant_info.empty: # Ensure column exists for generate_participant_details_html
            if 'overall_summary' in participant_info.columns and 'topic_summary' not in participant_info.columns:
                 participant_info.loc[:, 'topic_summary'] = participant_info['overall_summary'] # Fallback for function call consistency
            elif 'topic_summary' not in participant_info.columns :
                 participant_info.loc[:, 'topic_summary'] = pd.NA


        details_html = generate_participant_details_html(participant_info, selected_topic)

    updated_fig_dict = no_update
    if current_fig_state:
        updated_fig = go.Figure(current_fig_state)
        for trace in updated_fig.data:
            if hasattr(trace, 'customdata') and trace.customdata is not None:
                opacities = []
                for cd_point_tuple in trace.customdata: # customdata is often a list of tuples or lists
                    point_id = str(cd_point_tuple[0]) 
                    if new_selected_id:
                        opacities.append(1.0 if point_id == new_selected_id else 0.2)
                    else:
                        opacities.append(0.8) 
                if hasattr(trace, 'marker') and hasattr(trace.marker, 'opacity'):
                    trace.marker.opacity = opacities
        updated_fig_dict = updated_fig.to_dict()
    
    return details_html, new_selected_id, updated_fig_dict

@app.callback(
    Output('selected-participant-store', 'data', allow_duplicate=True),
    Output('participant-details-panel', 'children', allow_duplicate=True),
    Output('participant-scatter-plot', 'figure', allow_duplicate=True),
    Input('participant-clear-selection-btn', 'n_clicks'),
    State('participant-scatter-plot', 'figure'),
    prevent_initial_call=True
)
def clear_participant_selection(n_clicks, current_fig_state):
    if n_clicks is None:
        return no_update, no_update, no_update
    
    details_html = "Wählen Sie einen Teilnehmer für Details."
    updated_fig_dict = no_update
    if current_fig_state:
        updated_fig = go.Figure(current_fig_state)
        for trace in updated_fig.data:
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'opacity'):
                if isinstance(trace.marker.opacity, (list, np.ndarray)):
                    if trace.x is not None : # Check if trace.x has data
                         trace.marker.opacity = [0.8] * len(trace.x) 
                    else:
                         trace.marker.opacity = 0.8 # Fallback if trace.x is None
                else:
                    trace.marker.opacity = 0.8
        updated_fig_dict = updated_fig.to_dict()
            
    return None, details_html, updated_fig_dict


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True) # NEW