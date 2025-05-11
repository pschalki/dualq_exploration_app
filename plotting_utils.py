# plotting_utils.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd # <<< ADDED IMPORT
from data_loader import PARTY_COLORS, DEFAULT_COLOR

def project_embedding(embedding, axis_embedding_x, axis_embedding_y):
    if embedding is None or axis_embedding_x is None or axis_embedding_y is None:
        return 0
    axis_vector = axis_embedding_y - axis_embedding_x
    return np.dot(embedding, axis_vector)

def create_cluster_plot(df, x_col, y_col, size_col, color_col, text_col, custom_data_cols,
                        title, selected_cluster_id=None, hover_name_col=None,
                        x_axis_label="X", y_axis_label="Y"):
    fig = go.Figure()

    base_opacity = 0.8
    selected_opacity = 1.0
    dimmed_opacity = 0.2

    unique_clusters = df[color_col].unique()
    colors = px.colors.qualitative.Plotly

    min_bubble_diameter_pixels = 20 
    max_bubble_diameter_pixels = 70 
    
    if not df[size_col].empty and df[size_col].max() > 0:
        # Adjusted sizeref logic: smaller sizeref = larger bubbles
        # We want the largest bubble (max_size_col_value) to be max_bubble_diameter_pixels
        # We want the smallest bubble (min_size_col_value) to be min_bubble_diameter_pixels
        # This is an approximation and might need tuning.
        # A common way is sizeref = max_data_value / (max_pixel_size**2) but this is for area.
        # For diameter, it's more direct. If sizeref = 1, data value = pixel diameter.
        # To scale, if max_data = 10, max_pixel = 50, then effective_size = data * (50/10)
        # So, marker.size = data_column. We scale by adjusting sizeref.
        # Plotly's sizeref behavior can be a bit unintuitive for diameter.
        # Let's try: sizeref = max(df[size_col]) / max_bubble_diameter_pixels * 2.0 (the 2.0 is empirical)
        # Or a simpler, more direct way: make sizes proportional before passing to Plotly
        
        # Let's try scaling the 'size' column directly and use a small sizeref
        # This makes the 'size' values more directly map to visual size.
        min_data_size = df[size_col].min()
        max_data_size = df[size_col].max()
        
        if max_data_size == min_data_size : # all same size
             scaled_sizes = pd.Series([ (min_bubble_diameter_pixels + max_bubble_diameter_pixels) / 2 ] * len(df), index=df.index)
        else:
            # Linear scaling: output = output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start)
            scaled_sizes = min_bubble_diameter_pixels + \
                           ((max_bubble_diameter_pixels - min_bubble_diameter_pixels) / (max_data_size - min_data_size)) * \
                           (df[size_col] - min_data_size)
        
        # Handle cases where min_data_size is 0 or all values are the same
        scaled_sizes = scaled_sizes.fillna(min_bubble_diameter_pixels) # if some original sizes were NaN
        scaled_sizes = scaled_sizes.clip(lower=min_bubble_diameter_pixels) # ensure min size
        
        current_sizeref = 1 # Since we pre-scaled sizes to pixels
    else:
        scaled_sizes = pd.Series([min_bubble_diameter_pixels] * len(df), index=df.index) # Default if no sizes
        current_sizeref = 1

    for i, cluster_val in enumerate(unique_clusters):
        cluster_df_mask = (df[color_col] == cluster_val)
        cluster_df_slice = df[cluster_df_mask] # Get a slice for this cluster
        
        if cluster_df_slice.empty:
            continue

        current_scaled_sizes_slice = scaled_sizes[cluster_df_mask]

        opacity = base_opacity
        if selected_cluster_id:
            opacity = selected_opacity if str(cluster_val) == str(selected_cluster_id) else dimmed_opacity
        
        customdata_slice = cluster_df_slice[custom_data_cols]
        
        current_text_col_values = cluster_df_slice.get(text_col, pd.Series([str(cluster_val)] * len(cluster_df_slice), index=cluster_df_slice.index))
        current_hover_name_values = cluster_df_slice.get(hover_name_col, current_text_col_values)

        fig.add_trace(go.Scatter(
            x=cluster_df_slice[x_col],
            y=cluster_df_slice[y_col],
            mode='markers+text',
            marker=dict(
                size=current_scaled_sizes_slice, # USE PRE-SCALED SIZES
                sizemode='diameter', 
                sizeref=current_sizeref, # Should be 1 if sizes are pre-scaled to pixels
                color=colors[i % len(colors)],
                opacity=opacity,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=current_text_col_values,
            textposition="middle center", 
            textfont=dict(size=10, color="DarkSlateGrey"),
            customdata=customdata_slice,
            name=str(cluster_val),
            hovertext=current_hover_name_values,
            hovertemplate=
                f"<b>Cluster</b>: %{{hovertext}}<br>" +
                f"{x_axis_label}: %{{x:.2f}}<br>" +
                f"{y_axis_label}: %{{y:.2f}}<br>" +
                # Hovertemplate uses original size, not scaled
                f"Participants: %{{customdata[2]:.0f}}" + # Assuming anzahl_teilnehmer is 3rd in custom_data_cols
                "<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        showlegend=True,
        clickmode='event+select',
        dragmode='pan',
        plot_bgcolor='rgba(240,240,240,0.95)',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=50, b=40),
        transition_duration=300,
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=True, automargin=True), 
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=True, automargin=True),
        # Ensure text doesn't get cut off by expanding plot area if needed
        # modebar_add = ['v1hovermode', 'toggleSpikelines'] # Example: add more modebar buttons
    )
    # Try to prevent text overflow by adjusting margins or using automargin
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def create_participant_plot(df, x_col, y_col, color_by_col, custom_data_cols,
                            title, selected_participant_id=None, hover_name_col=None,
                            x_axis_label="X", y_axis_label="Y"):
    
    color_discrete_map = None
    if color_by_col == 'wahlabsicht' and color_by_col in df.columns:
        unique_parties = df[color_by_col].dropna().unique()
        color_discrete_map = {party: PARTY_COLORS.get(party, DEFAULT_COLOR) for party in unique_parties}
    elif color_by_col == 'cluster_id' and color_by_col in df.columns:
        unique_clusters = sorted(df[color_by_col].dropna().unique())
        plotly_colors = px.colors.qualitative.Plotly
        color_discrete_map = {
            cluster: plotly_colors[i % len(plotly_colors)] for i, cluster in enumerate(unique_clusters)
        }

    # Create the figure WITHOUT the opacity argument first
    fig = px.scatter(
        df, 
        x=x_col,
        y=y_col,
        color=color_by_col if color_by_col in df.columns else None,
        color_discrete_map=color_discrete_map,
        custom_data=custom_data_cols,
        title=title,
        hover_name=hover_name_col if hover_name_col in df.columns else None,
    )
    
    # --- MODIFICATION START: Apply opacity by iterating through traces ---
    # Prepare the list of opacities for all points in the original df order
    base_opacities_for_all_points = [0.8] * len(df)
    if selected_participant_id and 'interview_id' in df.columns:
        str_selected_id = str(selected_participant_id)
        base_opacities_for_all_points = [
            1.0 if str(pid) == str_selected_id else 0.2 
            for pid in df['interview_id']
        ]

    # px.scatter might create multiple traces if 'color' is used.
    # Each trace in fig.data corresponds to one category of the 'color' variable.
    # The points within each trace are a subset of the original df.
    # We need to map the original opacities to the points within each trace.
    
    for trace in fig.data:
        if hasattr(trace, 'customdata') and trace.customdata is not None:
            # Get the original indices of the points in this trace
            # This assumes customdata[0] is 'interview_id' which allows us to find original index
            # A more robust way is if px.scatter preserves original indices, or if we pass original indices in customdata
            # For now, let's assume 'interview_id' is unique and in customdata[0]
            
            trace_opacities = []
            # Get the interview_ids for the points in the current trace
            # trace.customdata is an array of arrays/tuples, each sub-array for a point
            trace_interview_ids = [str(cd[0]) for cd in trace.customdata]

            for pid_in_trace in trace_interview_ids:
                # Find this pid in the original df to get its pre-calculated opacity
                try:
                    # Find the index of this participant in the original df order
                    # This assumes df has a unique 'interview_id' that matches cd[0]
                    original_df_idx = df[df['interview_id'] == pid_in_trace].index[0]
                    trace_opacities.append(base_opacities_for_all_points[original_df_idx])
                except (IndexError, KeyError):
                    # Fallback if ID not found (should not happen with clean data)
                    trace_opacities.append(0.8) 
            
            if hasattr(trace, 'marker'):
                trace.marker.opacity = trace_opacities
            else: # Should not happen for scatter traces
                trace.marker = go.scatter.Marker(opacity=trace_opacities)
        else: # If no customdata (e.g., single trace, no coloring)
            if hasattr(trace, 'marker'):
                trace.marker.opacity = base_opacities_for_all_points # This assumes the trace contains all points
            else:
                trace.marker = go.scatter.Marker(opacity=base_opacities_for_all_points)

    # --- MODIFICATION END ---
    
    fig.update_traces(
        marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')), # Default marker style
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        showlegend=True if color_by_col and color_by_col in df.columns else False,
        clickmode='event+select',
        dragmode='pan',
        plot_bgcolor='rgba(240,240,240,0.95)',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20),
        transition_duration=300,
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=True),
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=True)
    )
    
    hover_id_col_index = 0 
    hover_color_col_index = 1 
    
    valid_id_for_hover = len(custom_data_cols) > hover_id_col_index
    valid_color_for_hover = len(custom_data_cols) > hover_color_col_index and \
                            custom_data_cols[hover_color_col_index] in df.columns # Check col exists in df

    hovertemplate_parts = []
    # Use hover_name_col if provided and exists, else default to 'ID'
    hover_name_display = df[hover_name_col].name if hover_name_col and hover_name_col in df.columns else 'ID'

    if valid_id_for_hover:
        hovertemplate_parts.append(f"<b>{hover_name_display}</b>: %{{customdata[{hover_id_col_index}]}}")
    
    if x_axis_label: hovertemplate_parts.append(f"{x_axis_label}: %{{x:.2f}}")
    if y_axis_label: hovertemplate_parts.append(f"{y_axis_label}: %{{y:.2f}}")
    
    if color_by_col and valid_color_for_hover:
         hovertemplate_parts.append(f"{str(df[custom_data_cols[hover_color_col_index]].name).title()}: %{{customdata[{hover_color_col_index}]}}")
    elif color_by_col and not valid_color_for_hover and color_by_col in df.columns: # If coloring var not in customdata but in df
        hovertemplate_parts.append(f"{str(color_by_col).title()}: %{{{color_by_col}}}")


    fig.update_traces(hovertemplate="<br>".join(hovertemplate_parts) + "<extra></extra>")

    return fig

def create_wahlabsicht_bar_chart(verteilung_dict):
    if not verteilung_dict or not isinstance(verteilung_dict, dict) or not any(verteilung_dict.values()): # Added check for empty values
        return go.Figure().update_layout(
            title="Keine Wahlabsicht Daten", 
            height=100, 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False) # Hide axes for empty
        )

    sorted_verteilung = sorted(verteilung_dict.items(), key=lambda item: item[1], reverse=True)
    
    parties = [item[0] for item in sorted_verteilung]
    percentages = [item[1] * 100 for item in sorted_verteilung] 

    colors = [PARTY_COLORS.get(p, DEFAULT_COLOR) for p in parties]

    fig = go.Figure(go.Bar(
        y=parties,
        x=percentages,
        orientation='h',
        marker_color=colors,
        text=[f'{p:.0f}%' for p in percentages],
        textposition='outside'
    ))
    fig.update_layout(
        title_text="Wahlabsicht Verteilung",
        xaxis_title="Prozent",
        yaxis_title="",
        yaxis=dict(autorange="reversed"), 
        height=max(150, len(parties) * 30 + 50), 
        margin=dict(l=100, r=30, t=40, b=30), # Ensure enough right margin for text
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        xaxis=dict(showgrid=False), # Cleaner look
    )
    return fig