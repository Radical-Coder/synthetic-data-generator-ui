"""
Streamlit Web Application for Auto-Detecting Synthetic Data Generator

This application provides a user-friendly interface for generating synthetic data
without requiring any coding knowledge.

Author: Robel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import io
import base64
from datetime import datetime

# Import the synthetic data generator
# Make sure AUTO-DETECTING_SYNTHETIC_DATA_GENERATOR_ENHANCED.py is in the same directory
try:
    from AUTO_DETECTING_SYNTHETIC_DATA_GENERATOR_ENHANCED import (
        AutoDetectingSyntheticGenerator,
        print_comparison_stats
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from AUTO_DETECTING_SYNTHETIC_DATA_GENERATOR_ENHANCED import (
        AutoDetectingSyntheticGenerator,
        print_comparison_stats
    )


# Page configuration
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'synthetic_df' not in st.session_state:
    st.session_state.synthetic_df = None
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'column_info' not in st.session_state:
    st.session_state.column_info = None
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'reference_tables' not in st.session_state:
    st.session_state.reference_tables = {}
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = None


def create_comparison_chart(real_data, synth_data, column_name, chart_type='distribution'):
    """
    Create interactive comparison charts for a single column.
    
    Args:
        real_data: Real data series
        synth_data: Synthetic data series
        column_name: Name of the column
        chart_type: Type of chart ('distribution', 'box', 'bar')
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if chart_type == 'distribution':
        # Try KDE, fall back to histogram if data has no variance
        try:
            kde_real = stats.gaussian_kde(real_data)
            kde_synth = stats.gaussian_kde(synth_data)
            x_range = np.linspace(
                min(real_data.min(), synth_data.min()),
                max(real_data.max(), synth_data.max()),
                200
            )
            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_real(x_range),
                mode='lines',
                name='Real Data',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_synth(x_range),
                mode='lines',
                name='Synthetic Data',
                line=dict(color='red', width=2, dash='dash')
            ))
            y_max = max(kde_real(x_range).max(), kde_synth(x_range).max())
            fig.add_trace(go.Scatter(
                x=[real_data.mean(), real_data.mean()],
                y=[0, y_max],
                mode='lines',
                name='Real Mean',
                line=dict(color='blue', width=1, dash='dot'),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=[synth_data.mean(), synth_data.mean()],
                y=[0, y_max],
                mode='lines',
                name='Synthetic Mean',
                line=dict(color='red', width=1, dash='dot'),
                showlegend=True
            ))
        except Exception:
            # Fall back to histogram when KDE fails (e.g. constant columns like Year)
            fig.add_trace(go.Histogram(
                x=real_data, name='Real Data',
                marker_color='blue', opacity=0.6, nbinsx=30
            ))
            fig.add_trace(go.Histogram(
                x=synth_data, name='Synthetic Data',
                marker_color='red', opacity=0.6, nbinsx=30
            ))
            fig.update_layout(barmode='overlay')

        fig.update_layout(
            title=f'{column_name} - Distribution Comparison',
            xaxis_title=column_name,
            yaxis_title='Density',
            hovermode='x unified',
            height=400
        )
        
    elif chart_type == 'box':
        # Box plot
        fig.add_trace(go.Box(
            y=real_data,
            name='Real Data',
            marker_color='blue',
            boxmean='sd'
        ))
        
        fig.add_trace(go.Box(
            y=synth_data,
            name='Synthetic Data',
            marker_color='red',
            boxmean='sd'
        ))
        
        fig.update_layout(
            title=f'{column_name} - Box Plot Comparison',
            yaxis_title=column_name,
            height=400
        )
        
    elif chart_type == 'bar':
        # Categorical bar chart - always show side by side grouped bars
        real_counts = real_data.value_counts().head(20)
        synth_counts = synth_data.value_counts().head(20)

        # Use union of both so no category is missing from either side
        all_categories = sorted(list(set(real_counts.index) | set(synth_counts.index)))

        real_freq = [real_counts.get(cat, 0) for cat in all_categories]
        synth_freq = [synth_counts.get(cat, 0) for cat in all_categories]

        fig.add_trace(go.Bar(
            x=all_categories,
            y=real_freq,
            name='Real Data',
            marker_color='blue',
            opacity=0.85,
            offsetgroup=0
        ))

        fig.add_trace(go.Bar(
            x=all_categories,
            y=synth_freq,
            name='Synthetic Data',
            marker_color='red',
            opacity=0.85,
            offsetgroup=1
        ))

        # Adjust chart width based on number of categories
        chart_height = 400
        chart_width = max(600, len(all_categories) * 50)

        fig.update_layout(
            title=f'{column_name} - Frequency Comparison',
            xaxis_title=column_name,
            yaxis_title='Count',
            barmode='group',
            bargap=0.15,
            bargroupgap=0.05,
            height=chart_height,
            width=chart_width,
            xaxis=dict(tickangle=-45)
        )
    
    return fig


def get_download_link(df, filename):
    """
    Generate a download link for dataframe.
    
    Args:
        df: Pandas dataframe
        filename: Name for download file
    
    Returns:
        Download button
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown('<div class="main-header">Auto-Detecting Synthetic Data Generator</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>Welcome!</b> This tool automatically generates high-quality synthetic data that preserves 
the statistical properties of your original dataset. No coding required!
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool uses advanced machine learning to:
    - Automatically detect data types
    - Preserve statistical distributions
    - Generate realistic synthetic data
    - Validate quality automatically
    """)
    
    st.header("Supported Data Types")
    st.markdown("""
    - **Numeric**: Continuous & Discrete
    - **Categorical**: Text categories
    - **Boolean**: True/False values
    - **DateTime**: Dates & timestamps
    - **Email**: Fake email generation
    - **Phone**: Fake phone number generation
    """)
    
    st.header("Current Status")
    if st.session_state.original_df is not None:
        st.success(f"Data Loaded: {st.session_state.original_df.shape[0]} rows")
    else:
        st.info("No data loaded")
    
    if st.session_state.generation_complete:
        st.success(f"Synthetic Data Generated: {st.session_state.synthetic_df.shape[0]} rows")
    else:
        st.info("Not generated yet")


# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upload & Generate",
    "Quality Dashboard", 
    "Statistical Comparison",
    "Download Results",
    "Performance Benchmarks"
])


# ============================================================================
# TAB 1: UPLOAD & GENERATE
# ============================================================================
with tab1:
    st.markdown('<div class="sub-header">Step 1: Upload Your Dataset</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or JSON file",
        type=['csv', 'json'],
        help="Upload a CSV or flat JSON file. The tool will automatically detect column types."
    )
    
    if uploaded_file is not None:
        try:
            # Load the data based on file type
            file_ext = uploaded_file.name.split('.')[-1].lower()

            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)

            elif file_ext == 'json':
                import json
                raw = json.load(uploaded_file)

                # Handle array of objects: [{...}, {...}]
                if isinstance(raw, list):
                    df = pd.DataFrame(raw)

                # Handle records dict: {"col": {"0": val, "1": val}}
                elif isinstance(raw, dict):
                    # Check for nested structures
                    sample_val = next(iter(raw.values()))
                    if isinstance(sample_val, (dict, list)):
                        try:
                            df = pd.DataFrame(raw)
                        except Exception:
                            st.error("Nested JSON structures are not supported. Please provide a flat JSON file — either an array of objects or a simple key-value records format.")
                            st.stop()
                    else:
                        df = pd.DataFrame([raw])
                else:
                    st.error("Unsupported JSON format. Please provide a flat JSON file — either an array of objects or a simple key-value records format.")
                    st.stop()

                # Flatten any remaining nested columns
                nested_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, (dict, list))).any()]
                if nested_cols:
                    st.warning(f"The following columns contain nested data and were dropped: {nested_cols}. Only flat structures are supported.")
                    df = df.drop(columns=nested_cols)

            st.session_state.original_df = df

            st.markdown('<div class="success-box">File uploaded successfully!</div>', unsafe_allow_html=True)
            
            # Show dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Preview data
            st.markdown('<div class="sub-header">Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()
    
    # Configuration section (only show if data is loaded)
    if st.session_state.original_df is not None:
        st.markdown('<div class="sub-header">Step 2: Configure Generation Parameters</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_rows = st.number_input(
                "Number of rows to generate",
                min_value=100,
                max_value=1000000,
                value=min(1000, len(st.session_state.original_df)),
                step=100,
                help="How many synthetic rows to generate"
            )
        
        with col2:
            random_seed = st.number_input(
                "Random Seed (for reproducibility)",
                min_value=0,
                max_value=9999,
                value=42,
                help="Use same seed to get identical results"
            )
        
        # Advanced options (collapsible)
        with st.expander("⚙️ Advanced Options"):
            max_components = st.slider(
                "Maximum GMM Components",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum Gaussian components for continuous data modeling"
            )
            
            discrete_threshold = st.slider(
                "Discrete Detection Threshold",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="Unique value ratio below which numeric columns are treated as discrete"
            )
            
            use_reference_tables = st.checkbox(
                "Use Reference Tables",
                value=False,
                help="Provide custom lists of valid values for categorical columns"
            )
            
            if use_reference_tables:
                if st.session_state.original_df is not None:
                    cat_cols = [
                        col for col in st.session_state.original_df.columns
                        if st.session_state.original_df[col].dtype == object
                        or str(st.session_state.original_df[col].dtype) == 'string'
                    ]
                    st.markdown("**Upload Reference Tables for Categorical Columns**")
                    st.caption("Each reference table should be a CSV file with a single column of valid values.")
                    selected_ref_cols = st.multiselect(
                        "Select columns to add reference tables for:",
                        options=cat_cols,
                        help="Only categorical columns are shown"
                    )
                    for col in selected_ref_cols:
                        ref_file = st.file_uploader(
                            f"Upload reference CSV for: {col}",
                            type=["csv"],
                            key=f"ref_{col}"
                        )
                        if ref_file is not None:
                            try:
                                ref_df = pd.read_csv(ref_file)
                                ref_values = ref_df.iloc[:, 0].dropna().astype(str).unique().tolist()
                                # Store with a stable key so it survives reruns
                                st.session_state.reference_tables[col] = ref_values
                                st.session_state[f"ref_loaded_{col}"] = True
                            except Exception as e:
                                st.error(f"Could not read reference file for {col}: {e}")
                        # Show loaded status even after rerun
                        if st.session_state.get(f"ref_loaded_{col}") and col in st.session_state.reference_tables:
                            st.success(f"{col} — {len(st.session_state.reference_tables[col]):,} values loaded from reference table")

                    # Show summary of all active reference tables
                    if st.session_state.reference_tables:
                        st.markdown("**Active Reference Tables:**")
                        for col, vals in st.session_state.reference_tables.items():
                            st.info(f"{col}: {len(vals):,} reference values loaded")
                else:
                    st.warning("Please upload a dataset first to see available columns.")
        
        # Generate button
        st.markdown('<div class="sub-header">Step 3: Generate Synthetic Data</div>', unsafe_allow_html=True)
        
        # Show active reference tables reminder before generate button
        if st.session_state.reference_tables:
            st.success(f"Reference tables active for: {', '.join(st.session_state.reference_tables.keys())}")
            if st.button("Clear All Reference Tables"):
                st.session_state.reference_tables = {}
                for key in list(st.session_state.keys()):
                    if key.startswith("ref_loaded_"):
                        del st.session_state[key]
                st.rerun()

        if st.button("Generate Synthetic Data", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing dataset and generating synthetic data..."):
                try:
                    # Initialize generator
                    ref_tables = st.session_state.get('reference_tables', {})
                    generator = AutoDetectingSyntheticGenerator(
                        max_gmm_components=max_components,
                        random_state=random_seed,
                        discrete_threshold=discrete_threshold,
                        categorical_libraries=ref_tables if ref_tables else None
                    )
                    
                    # Fit the generator
                    progress_text = st.empty()
                    progress_text.info("Step 1/3: Analyzing column types...")
                    generator.fit(st.session_state.original_df)
                    
                    # Store column info
                    st.session_state.column_info = generator.column_info
                    
                    # Generate synthetic data
                    progress_text.info("Step 2/3: Generating synthetic data...")
                    sample_size = min(num_rows, len(st.session_state.original_df))
                    seed_sample = st.session_state.original_df.sample(sample_size, random_state=random_seed)
                    synthetic_df = generator.generate(num_rows=num_rows, seed_df=seed_sample)
                    
                    # Store results
                    st.session_state.synthetic_df = synthetic_df
                    st.session_state.generator = generator
                    st.session_state.generation_complete = True
                    
                    progress_text.info("Step 3/3: Validating quality...")
                    
                    # Clear progress
                    progress_text.empty()
                    
                    # Success message
                    st.markdown(f"""
                    <div class="success-box">
                    <b>Generation Complete!</b><br>
                    Successfully generated {synthetic_df.shape[0]:,} rows with {synthetic_df.shape[1]} columns.<br>
                    Navigate to other tabs to view quality dashboard and download results.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show column detection summary
                    st.markdown('<div class="sub-header">Column Detection Summary</div>', unsafe_allow_html=True)
                    
                    detection_data = []
                    active_ref_tables = st.session_state.get('reference_tables', {})
                    for col, info in st.session_state.column_info.items():
                        method = info['method']
                        if col in active_ref_tables:
                            method = f"reference_table_sampling ({len(active_ref_tables[col]):,} values)"
                        # Friendly display names
                        type_display = {
                            'email': 'email',
                            'phone': 'phone',
                            'categorical': 'categorical',
                            'boolean': 'boolean',
                            'discrete_numeric': 'discrete_numeric',
                            'continuous_numeric': 'continuous_numeric',
                            'datetime': 'datetime'
                        }.get(info['type'], info['type'])
                        detection_data.append({
                            'Column': col,
                            'Detected Type': type_display,
                            'Generation Method': method
                        })
                    
                    detection_df = pd.DataFrame(detection_data)
                    st.dataframe(detection_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.markdown("""
        <div class="warning-box">
        Please upload a CSV or JSON file to begin.
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# TAB 2: QUALITY DASHBOARD
# ============================================================================
with tab2:
    st.markdown('<div class="sub-header">Quality Validation Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.generation_complete:
        real_df = st.session_state.original_df
        synth_df = st.session_state.synthetic_df
        
        # Summary metrics
        st.markdown("### Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Rows", f"{len(real_df):,}")
        with col2:
            st.metric("Synthetic Rows", f"{len(synth_df):,}")
        with col3:
            numeric_cols = real_df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        with col4:
            categorical_cols = real_df.select_dtypes(include=['object', 'category', 'bool']).columns
            st.metric("Categorical Columns", len(categorical_cols))
        
        # Column selector
        st.markdown("### Select Columns to Visualize")
        
        all_columns = list(real_df.columns)
        selected_columns = st.multiselect(
            "Choose columns for detailed comparison",
            options=all_columns,
            default=all_columns[:3] if len(all_columns) >= 3 else all_columns,
            help="Select which columns you want to see detailed visualizations for"
        )
        
        if not selected_columns:
            st.warning("Please select at least one column to visualize.")
        else:
            # Generate charts for selected columns
            for col in selected_columns:
                st.markdown(f"### {col}")
                
                col_info = st.session_state.column_info.get(col, {})
                col_type = col_info.get('type', 'unknown')
                
                # Display column type badge
                type_colors = {
                    'continuous_numeric': '🔵',
                    'discrete_numeric': '🟢',
                    'categorical': '🟡',
                    'boolean': '🟣',
                    'datetime': '🟠'
                }
                type_emoji = type_colors.get(col_type, '⚪')
                st.markdown(f"**Type:** {type_emoji} {col_type}")
                
                # Create appropriate visualizations based on type
                if col_type in ['continuous_numeric', 'discrete_numeric']:
                    real_data = real_df[col].dropna()
                    synth_data = synth_df[col].dropna()
                    
                    # Show statistics
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.markdown("**Real Data Stats**")
                        st.write(f"Mean: {real_data.mean():.2f}")
                        st.write(f"Std: {real_data.std():.2f}")
                        st.write(f"Min: {real_data.min():.2f}")
                        st.write(f"Max: {real_data.max():.2f}")
                    
                    with stat_col2:
                        st.markdown("**Synthetic Data Stats**")
                        st.write(f"Mean: {synth_data.mean():.2f}")
                        st.write(f"Std: {synth_data.std():.2f}")
                        st.write(f"Min: {synth_data.min():.2f}")
                        st.write(f"Max: {synth_data.max():.2f}")
                    
                    # Distribution plot
                    fig_dist = create_comparison_chart(real_data, synth_data, col, 'distribution')
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Box plot
                    fig_box = create_comparison_chart(real_data, synth_data, col, 'box')
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                elif col_type in ['categorical', 'boolean']:
                    real_data = real_df[col].dropna()
                    synth_data = synth_df[col].dropna()
                    
                    # Show top categories
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.markdown("**Real Data Distribution**")
                        real_dist = real_data.value_counts(normalize=True).head(5)
                        for cat, freq in real_dist.items():
                            st.write(f"{cat}: {freq:.1%}")
                    
                    with stat_col2:
                        st.markdown("**Synthetic Data Distribution**")
                        synth_dist = synth_data.value_counts(normalize=True).head(5)
                        for cat, freq in synth_dist.items():
                            st.write(f"{cat}: {freq:.1%}")
                    
                    # Bar chart
                    fig_bar = create_comparison_chart(real_data, synth_data, col, 'bar')
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                elif col_type == 'datetime':
                    st.info(f"DateTime column: {col}. Range preserved from {real_df[col].min()} to {real_df[col].max()}")
                
                st.markdown("---")
    
    else:
        st.markdown("""
        <div class="warning-box">
        Please generate synthetic data first (Tab 1: Upload & Generate).
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# TAB 3: STATISTICAL COMPARISON
# ============================================================================
with tab3:
    st.markdown('<div class="sub-header">Detailed Statistical Comparison</div>', unsafe_allow_html=True)
    
    if st.session_state.generation_complete:
        real_df = st.session_state.original_df
        synth_df = st.session_state.synthetic_df
        
        # Numeric columns comparison
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            st.markdown("### 🔢 Numeric Columns")
            
            numeric_comparison = []
            for col in numeric_cols:
                if col in synth_df.columns:
                    real_mean = real_df[col].mean()
                    synth_mean = synth_df[col].mean()
                    real_std = real_df[col].std()
                    synth_std = synth_df[col].std()
                    
                    mean_diff = abs(synth_mean - real_mean)
                    mean_diff_pct = (mean_diff / real_mean * 100) if real_mean != 0 else 0
                    
                    numeric_comparison.append({
                        'Column': col,
                        'Real Mean': f"{real_mean:.2f}",
                        'Synthetic Mean': f"{synth_mean:.2f}",
                        'Mean Diff %': f"{mean_diff_pct:.1f}%",
                        'Real Std': f"{real_std:.2f}",
                        'Synthetic Std': f"{synth_std:.2f}"
                    })
            
            comparison_df = pd.DataFrame(numeric_comparison)
            st.dataframe(comparison_df, use_container_width=True)
        
        # Categorical columns comparison
        categorical_cols = real_df.select_dtypes(include=['object', 'category', 'bool']).columns
        
        if len(categorical_cols) > 0:
            st.markdown("### 📝 Categorical Columns")
            
            for col in categorical_cols[:10]:
                if col in synth_df.columns:
                    st.markdown(f"**{col}**")

                    # Get top categories sorted alphabetically so both sides always match
                    real_dist = real_df[col].value_counts(normalize=True)
                    synth_dist = synth_df[col].value_counts(normalize=True)
                    top_cats = sorted(real_dist.index[:10].tolist())

                    # Single merged table - real and synthetic on same row
                    merged_table = pd.DataFrame({
                        'Category': top_cats,
                        'Real Frequency': [f"{real_dist.get(c, 0):.1%}" for c in top_cats],
                        'Synthetic Frequency': [f"{synth_dist.get(c, 0):.1%}" for c in top_cats]
                    })
                    st.dataframe(merged_table, use_container_width=True, hide_index=True)
                    st.markdown("---")
        
        # DateTime columns
        datetime_cols = real_df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) > 0:
            st.markdown("### 📅 DateTime Columns")
            
            datetime_comparison = []
            for col in datetime_cols:
                if col in synth_df.columns:
                    datetime_comparison.append({
                        'Column': col,
                        'Real Min': str(real_df[col].min()),
                        'Real Max': str(real_df[col].max()),
                        'Synthetic Min': str(synth_df[col].min()),
                        'Synthetic Max': str(synth_df[col].max())
                    })
            
            if datetime_comparison:
                datetime_df = pd.DataFrame(datetime_comparison)
                st.dataframe(datetime_df, use_container_width=True)
    
    else:
        st.markdown("""
        <div class="warning-box">
        Please generate synthetic data first (Tab 1: Upload & Generate).
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# TAB 4: DOWNLOAD RESULTS
# ============================================================================
with tab4:
    st.markdown('<div class="sub-header">Download Your Results</div>', unsafe_allow_html=True)
    
    if st.session_state.generation_complete:
        synth_df = st.session_state.synthetic_df
        
        st.markdown("### Download Options")
        
        # Preview synthetic data
        st.markdown("**Synthetic Data Preview:**")
        st.dataframe(synth_df.head(20), use_container_width=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as CSV
            csv = synth_df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Synthetic Data (CSV)",
                data=csv,
                file_name=f"synthetic_data_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Download column info as JSON
            import json
            column_info_json = json.dumps(st.session_state.column_info, indent=2, default=str)
            st.download_button(
                label="📋 Download Column Info (JSON)",
                data=column_info_json,
                file_name=f"column_info_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # File information
        st.markdown("### File Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Rows", f"{len(synth_df):,}")
        with info_col2:
            st.metric("Columns", synth_df.shape[1])
        with info_col3:
            file_size_mb = synth_df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Estimated Size", f"{file_size_mb:.2f} MB")
        
        st.markdown("""
        <div class="info-box">
        <b>Tip:</b> The synthetic data has the same structure as your original data 
        and can be used directly in your applications, testing environments, or shared 
        with partners without privacy concerns.
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="warning-box">
        Please generate synthetic data first (Tab 1: Upload & Generate).
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# TAB 5: PERFORMANCE BENCHMARKS
# ============================================================================
with tab5:
    st.markdown('<div class="sub-header">Performance Benchmarks</div>', unsafe_allow_html=True)
    st.markdown("Test how fast the tool generates synthetic data at different row counts.")

    if st.session_state.original_df is None:
        st.markdown("""
        <div class="warning-box">
        Please upload a dataset first (Tab 1: Upload & Generate) before running benchmarks.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### Select Row Sizes to Benchmark")
        
        col1, col2 = st.columns(2)
        with col1:
            run_1k   = st.checkbox("1,000 rows",     value=True)
            run_10k  = st.checkbox("10,000 rows",    value=True)
            run_50k  = st.checkbox("50,000 rows",    value=True)
        with col2:
            run_100k = st.checkbox("100,000 rows",   value=False)
            run_500k = st.checkbox("500,000 rows",   value=False)
            run_1m   = st.checkbox("1,000,000 rows", value=False)

        sizes_to_run = []
        if run_1k:   sizes_to_run.append(1000)
        if run_10k:  sizes_to_run.append(10000)
        if run_50k:  sizes_to_run.append(50000)
        if run_100k: sizes_to_run.append(100000)
        if run_500k: sizes_to_run.append(500000)
        if run_1m:   sizes_to_run.append(1000000)

        if st.button("Run Benchmark", type="primary", use_container_width=True):
            if not sizes_to_run:
                st.warning("Please select at least one row size to benchmark.")
            else:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, size in enumerate(sizes_to_run):
                    status_text.info(f"Benchmarking {size:,} rows... ({i+1}/{len(sizes_to_run)})")
                    
                    try:
                        import time
                        # Initialize generator
                        gen = AutoDetectingSyntheticGenerator(
                            max_gmm_components=3,
                            random_state=42,
                            discrete_threshold=0.05
                        )
                        gen.fit(st.session_state.original_df)

                        # Time the generation
                        start = time.time()
                        seed_sample = st.session_state.original_df.sample(
                            min(size, len(st.session_state.original_df)),
                            random_state=42,
                            replace=True
                        )
                        gen.generate(num_rows=size, seed_df=seed_sample)
                        elapsed = time.time() - start

                        rows_per_sec = int(size / elapsed)
                        results.append({
                            "Rows Generated": f"{size:,}",
                            "Time (seconds)": round(elapsed, 2),
                            "Rows per Second": f"{rows_per_sec:,}"
                        })

                    except Exception as e:
                        results.append({
                            "Rows Generated": f"{size:,}",
                            "Time (seconds)": "Error",
                            "Rows per Second": str(e)[:50]
                        })

                    progress_bar.progress((i + 1) / len(sizes_to_run))

                status_text.success("Benchmark complete!")
                st.session_state.benchmark_results = results

        # Show results if available
        if st.session_state.get('benchmark_results'):
            results = st.session_state.benchmark_results

            st.markdown("### Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Line chart - only for successful runs
            try:
                chart_data = [
                    r for r in results
                    if isinstance(r["Time (seconds)"], (int, float))
                ]
                if len(chart_data) >= 2:
                    import plotly.graph_objects as go
                    x_vals = [r["Rows Generated"] for r in chart_data]
                    y_vals = [r["Time (seconds)"] for r in chart_data]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers',
                        name='Generation Time',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8)
                    ))
                    fig.update_layout(
                        title="Generation Time vs Row Count",
                        xaxis_title="Rows Generated",
                        yaxis_title="Time (seconds)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Summary
                    fastest = min(chart_data, key=lambda r: r["Time (seconds)"])
                    slowest = max(chart_data, key=lambda r: r["Time (seconds)"])
                    avg_rps = sum(
                        int(r["Rows per Second"].replace(",", ""))
                        for r in chart_data
                    ) // len(chart_data)

                    st.markdown("### Summary")
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.metric("Fastest Run", f"{fastest['Time (seconds)']}s", fastest['Rows Generated'])
                    with s2:
                        st.metric("Slowest Run", f"{slowest['Time (seconds)']}s", slowest['Rows Generated'])
                    with s3:
                        st.metric("Avg Rows/Second", f"{avg_rps:,}")

            except Exception:
                pass


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <b>Auto-Detecting Synthetic Data Generator</b><br>
    Developed by Robel <br>
    © 2026 | Version 1.1
</div>
""", unsafe_allow_html=True)
