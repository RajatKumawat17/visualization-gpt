#!/usr/bin/env python3
"""
Configuration file for the Automated Visualization Engine
Customize settings and behavior here
"""

import os
from pathlib import Path

# =============================================================================
# BASIC SETTINGS
# =============================================================================

# Output directories
OUTPUT_DIR = "outputs"
CHARTS_DIR = "charts" 
REPORTS_DIR = "reports"

# Chart settings
CHART_DPI = 300
CHART_STYLE = 'seaborn-v0_8'  # matplotlib style
FIGURE_SIZE = (10, 6)  # default figure size
COLOR_PALETTE = "husl"  # seaborn palette

# =============================================================================
# DATA PROCESSING SETTINGS
# =============================================================================

# File handling
MAX_FILE_SIZE_MB = 100  # Maximum file size to process
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']
ENCODING_ATTEMPTS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

# Data analysis limits
MAX_CATEGORICAL_UNIQUE = 50  # Max unique values to show in categorical plots
MAX_CORRELATION_PAIRS = 10   # Max scatter plots for correlations
MIN_CORRELATION_THRESHOLD = 0.5  # Minimum correlation to show scatter plot

# Missing data thresholds
MISSING_DATA_THRESHOLD = 0.1  # Warn if >10% missing data
HIGH_CARDINALITY_THRESHOLD = 100  # Warn if categorical column has >100 unique values

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Chart generation limits
MAX_DISTRIBUTION_PLOTS = 6    # Max individual distribution plots
MAX_CATEGORICAL_PLOTS = 5     # Max categorical analysis plots
MAX_SCATTER_PLOTS = 4         # Max scatter plots to generate
HISTOGRAM_BINS = 'auto'       # Number of bins for histograms

# Interactive chart settings
INTERACTIVE_CHARTS_ENABLED = True
PLOTLY_THEME = 'plotly_white'  # Plotly theme

# Chart types to generate
CHART_TYPES_ENABLED = {
    'distributions': True,
    'categorical': True,
    'correlations': True,
    'interactive': True,
    'time_series': True,  # If datetime columns detected
    'outlier_analysis': True
}

# =============================================================================
# LLM INTEGRATION SETTINGS (Future Enhancement)
# =============================================================================

# OpenAI API settings (uncomment when ready to use)
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# OPENAI_MODEL = 'gpt-4'
# LLM_ENABLED = False

# LLM prompts for insights
INSIGHT_PROMPT_TEMPLATE = """
Analyze this dataset and provide 3-5 key insights:

Dataset Info:
- Shape: {shape}
- Columns: {columns}
- Data Types: {dtypes}
- Missing Data: {missing_data}

Summary Statistics:
{summary_stats}

Focus on:
1. Data quality issues
2. Interesting patterns or trends
3. Potential relationships between variables
4. Recommendations for further analysis
"""

# =============================================================================
# REPORT SETTINGS
# =============================================================================

# HTML report customization
REPORT_TITLE = "Automated Data Analysis Report"
REPORT_THEME = {
    'primary_color': '#2E86AB',
    'secondary_color': '#A23B72',
    'background_color': '#F18F01',
    'text_color': '#333333',
    'font_family': 'Arial, sans-serif'
}

# Report sections to include
REPORT_SECTIONS = {
    'executive_summary': True,
    'data_overview': True,
    'key_insights': True,
    'visualizations': True,
    'recommendations': True,
    'technical_details': False  # Hide technical details by default
}

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Memory management
CHUNK_SIZE = 10000  # For processing large files
MAX_MEMORY_MB = 1000  # Max memory usage before chunking

# Parallel processing
ENABLE_MULTIPROCESSING = False  # Set to True for parallel chart generation
MAX_WORKERS = 4  # Number of parallel workers

# Caching
ENABLE_CACHING = True
CACHE_DIR = ".cache"
CACHE_EXPIRY_DAYS = 7

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "viz_engine.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Debug options
DEBUG_MODE = False
SAVE_INTERMEDIATE_FILES = False  # Save data processing steps
VERBOSE_OUTPUT = True  # Print detailed progress

# Testing
GENERATE_SAMPLE_DATA = True  # Generate sample data if no file provided
SAMPLE_DATA_SIZE = 1000  # Number of rows for sample data

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_output_path(subdir=""):
    """Get the full output path"""
    path = Path(OUTPUT_DIR)
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if MAX_FILE_SIZE_MB <= 0:
        errors.append("MAX_FILE_SIZE_MB must be positive")
    
    if CHART_DPI < 72:
        errors.append("CHART_DPI should be at least 72")
    
    if MIN_CORRELATION_THRESHOLD < 0 or MIN_CORRELATION_THRESHOLD > 1:
        errors.append("MIN_CORRELATION_THRESHOLD must be between 0 and 1")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True

# Validate config on import
validate_config()

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

# Make key settings available for import
__all__ = [
    'OUTPUT_DIR', 'CHARTS_DIR', 'REPORTS_DIR',
    'CHART_DPI', 'FIGURE_SIZE', 'COLOR_PALETTE',
    'MAX_FILE_SIZE_MB', 'SUPPORTED_FORMATS',
    'CHART_TYPES_ENABLED', 'REPORT_SECTIONS',
    'get_output_path', 'validate_config'
]