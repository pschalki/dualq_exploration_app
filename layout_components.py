# layout_components.py
from dash import dcc, html
import dash_bootstrap_components as dbc
from plotting_utils import create_wahlabsicht_bar_chart
import pandas as pd # <<< ADD THIS IMPORT

def generate_cluster_details_html(cluster_data, selected_topic):
    # ... (rest of the function is the same)
    if cluster_data is None or cluster_data.empty:
        return html.P("Klicken Sie auf einen Cluster, um Details anzuzeigen.")

    row = cluster_data.iloc[0]
    summary_col = 'topic_summary' if selected_topic != "Gesamtes Interview" else 'overall_summary'
    # Ensure 'label' column exists, fallback to cluster_id if not
    cluster_display_name = row.get('label', f"Cluster {row.get('cluster_id')}")
    summary = row.get(summary_col, "Keine Zusammenfassung verfügbar.")


    details = [
        html.H4(cluster_display_name), # Use cluster_display_name
        dbc.Row([
            dbc.Col(html.Strong("Teilnehmer:"), width=6),
            dbc.Col(f"{row.get('anzahl_teilnehmer', 'N/A')}", width=6)
        ]),
        dbc.Row([
            dbc.Col(html.Strong("Frauenanteil:"), width=6),
            dbc.Col(f"{row.get('frauenanteil', 0)*100:.0f}%", width=6)
        ]),
        dbc.Row([
            dbc.Col(html.Strong("Ø Alter:"), width=6),
            dbc.Col(f"{row.get('durchschnittsalter', 0):.1f}", width=6)
        ]),
        dbc.Row([
            dbc.Col(html.Strong("Ø Einkommen:"), width=6),
            dbc.Col(f"{row.get('durchschnittseinkommen', 0):.0f}€", width=6) 
        ]),
        html.Hr(),
        html.Strong("Zusammenfassung:" if selected_topic == "Gesamtes Interview" else f"Zusammenfassung ({selected_topic}):"),
        html.P(summary, style={'maxHeight': '200px', 'overflowY': 'auto', 'fontSize': 'small'}),
    ]

    # Check for 'wahlabsicht_verteilung_parsed' before accessing
    if 'wahlabsicht_verteilung_parsed' in row and pd.notna(row['wahlabsicht_verteilung_parsed']) and row['wahlabsicht_verteilung_parsed']:
        details.append(html.Hr())
        details.append(dcc.Graph(
            figure=create_wahlabsicht_bar_chart(row['wahlabsicht_verteilung_parsed']),
            config={'displayModeBar': False}
        ))
    return html.Div(details)


def generate_participant_details_html(participant_data, selected_topic):
    if participant_data is None or participant_data.empty:
        return html.P("Klicken Sie auf einen Teilnehmer, um Details anzuzeigen.")

    row = participant_data.iloc[0]
    # Determine which summary to show
    summary_text = ""
    if selected_topic == "Gesamtes Interview":
        summary_text = row.get('overall_summary', "Keine Zusammenfassung verfügbar.")
    else:
        # The 'topic_summary' column should have been prepared in the app.py callback
        summary_text = row.get('topic_summary', "Keine themenspezifische Zusammenfassung verfügbar.")


    demographics = [
        ("ID", row.get('interview_id')),
        ("Alter", row.get('alter')),
        ("Geschlecht", row.get('geschlecht')),
        ("Einkommen", row.get('einkommen')),
        ("Beruf", row.get('beruf')),
        ("Wahlabsicht", row.get('wahlabsicht')),
        ("Cluster ID", row.get('cluster_id')),
    ]

    details_content = [html.H4(f"Details für {row.get('interview_id')}")]
    for label, value in demographics:
        if pd.notnull(value): # This is where 'pd' was needed
            details_content.append(dbc.Row([
                dbc.Col(html.Strong(f"{label}:"), width=4),
                dbc.Col(str(value), width=8)
            ]))
    
    details_content.extend([
        html.Hr(),
        html.Strong("Zusammenfassung:" if selected_topic == "Gesamtes Interview" else f"Zusammenfassung ({selected_topic}):"),
        html.P(summary_text, style={'maxHeight': '300px', 'overflowY': 'auto', 'fontSize': 'small'})
    ])
    return html.Div(details_content)

def generate_semantic_axis_description_html(axis_key, semantic_axes_dict, x_or_y):
    # ... (rest of the function is the same)
    if not axis_key or not semantic_axes_dict:
        return ""
    axis_info = semantic_axes_dict.get(axis_key)
    if not axis_info:
        return ""
    
    return dbc.Card(
        dbc.CardBody([
            html.H6(f"{x_or_y}-Achse: {axis_key}", className="card-title"),
            html.P([html.Strong(f"{axis_info.get('X_Name')}: "), axis_info.get('Description_X')], className="card-text small"),
            html.P([html.Strong(f"{axis_info.get('Y_Name')}: "), axis_info.get('Description_Y')], className="card-text small"),
        ]),
        className="mt-2",
        style={'fontSize': '0.8rem'}
    )