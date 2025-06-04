#!/usr/bin/env python3
"""
GPT-Powered Automated Visualization Engine
A complete implementation for automated data analysis and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AutoVizEngine:
    """Main class for automated data visualization"""
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.insights = []
        self.generated_charts = []
        
    def load_data(self, file_path):
        """Load CSV or Excel data with robust error handling"""
        try:
            file_path = Path(file_path)
            if file_path.suffix.lower() == '.csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        self.df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV file")
                    
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
                
            print(f"✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def analyze_data_structure(self):
        """Analyze data structure and generate insights"""
        if self.df is None:
            return {}
            
        analysis = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'datetime_columns': [],
            'summary_stats': {}
        }
        
        # Detect datetime columns
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col].dropna().head(100))
                    analysis['datetime_columns'].append(col)
                except:
                    pass
        
        # Generate summary statistics for numeric columns
        if analysis['numeric_columns']:
            analysis['summary_stats'] = self.df[analysis['numeric_columns']].describe().to_dict()
        
        return analysis
    
    def generate_automatic_insights(self):
        """Generate automatic insights based on data patterns"""
        analysis = self.analyze_data_structure()
        insights = []
        
        # Basic data overview
        insights.append(f"Dataset contains {analysis['shape'][0]:,} rows and {analysis['shape'][1]} columns")
        
        # Missing data insights
        missing_cols = [col for col, missing in analysis['missing_values'].items() if missing > 0]
        if missing_cols:
            insights.append(f"Missing data found in {len(missing_cols)} columns: {', '.join(missing_cols[:3])}")
        
        # Data type insights
        insights.append(f"Data types: {len(analysis['numeric_columns'])} numeric, {len(analysis['categorical_columns'])} categorical")
        
        # Numeric data insights
        for col in analysis['numeric_columns'][:3]:  # Top 3 numeric columns
            stats = analysis['summary_stats'].get(col, {})
            if stats:
                insights.append(f"{col}: mean={stats.get('mean', 0):.2f}, std={stats.get('std', 0):.2f}")
        
        # Categorical data insights
        for col in analysis['categorical_columns'][:3]:  # Top 3 categorical columns
            unique_count = self.df[col].nunique()
            insights.append(f"{col}: {unique_count} unique values")
        
        self.insights = insights
        return insights
    
    def create_distribution_plots(self):
        """Create distribution plots for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return []
        
        charts_created = []
        
        # Single distribution plots
        for col in numeric_cols[:4]:  # Limit to first 4 columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            self.df[col].hist(bins=30, ax=ax1, alpha=0.7, color='skyblue')
            ax1.set_title(f'Distribution of {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Frequency')
            
            # Box plot
            self.df.boxplot(column=col, ax=ax2)
            ax2.set_title(f'Box Plot of {col}')
            
            plt.tight_layout()
            chart_path = self.charts_dir / f"distribution_{col.replace(' ', '_')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            charts_created.append({
                'type': 'distribution',
                'column': col,
                'path': str(chart_path),
                'description': f'Distribution analysis of {col}'
            })
        
        # Multi-column histogram
        if len(numeric_cols) > 1:
            n_cols = min(len(numeric_cols), 4)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:4]):
                self.df[col].hist(bins=20, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            chart_path = self.charts_dir / "multi_distribution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            charts_created.append({
                'type': 'multi_distribution',
                'columns': list(numeric_cols[:4]),
                'path': str(chart_path),
                'description': 'Distribution comparison of numeric columns'
            })
        
        return charts_created
    
    def create_categorical_plots(self):
        """Create plots for categorical data"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        charts_created = []
        
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            value_counts = self.df[col].value_counts().head(10)  # Top 10 categories
            
            if len(value_counts) > 1:
                # Bar plot
                plt.figure(figsize=(10, 6))
                value_counts.plot(kind='bar', color='lightcoral')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                chart_path = self.charts_dir / f"categorical_{col.replace(' ', '_')}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts_created.append({
                    'type': 'categorical',
                    'column': col,
                    'path': str(chart_path),
                    'description': f'Category distribution of {col}'
                })
        
        return charts_created
    
    def create_correlation_plots(self):
        """Create correlation analysis plots"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return []
        
        charts_created = []
        
        # Correlation heatmap
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        chart_path = self.charts_dir / "correlation_heatmap.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        charts_created.append({
            'type': 'correlation',
            'columns': list(numeric_cols),
            'path': str(chart_path),
            'description': 'Correlation analysis between numeric variables'
        })
        
        # Scatter plots for highly correlated pairs
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.5:  # High correlation threshold
                    high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
        
        # Create scatter plots for top 2 correlated pairs
        for col1, col2, corr_val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:2]:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.df[col1], self.df[col2], alpha=0.6, color='green')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(f'Scatter Plot: {col1} vs {col2} (r={corr_val:.2f})')
            
            # Add trend line
            z = np.polyfit(self.df[col1].fillna(0), self.df[col2].fillna(0), 1)
            p = np.poly1d(z)
            plt.plot(self.df[col1], p(self.df[col1]), "r--", alpha=0.8)
            
            plt.tight_layout()
            chart_path = self.charts_dir / f"scatter_{col1}_{col2}.png".replace(' ', '_')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            charts_created.append({
                'type': 'scatter',
                'columns': [col1, col2],
                'path': str(chart_path),
                'description': f'Scatter plot showing relationship between {col1} and {col2}'
            })
        
        return charts_created
    
    def create_interactive_plots(self):
        """Create interactive plots using Plotly"""
        charts_created = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Interactive scatter plot
            fig = px.scatter(self.df, x=numeric_cols[0], y=numeric_cols[1], 
                           title=f'Interactive Scatter: {numeric_cols[0]} vs {numeric_cols[1]}',
                           hover_data=numeric_cols[:4].tolist())
            
            chart_path = self.charts_dir / "interactive_scatter.html"
            fig.write_html(chart_path)
            
            charts_created.append({
                'type': 'interactive_scatter',
                'columns': [numeric_cols[0], numeric_cols[1]],
                'path': str(chart_path),
                'description': 'Interactive scatter plot with hover details'
            })
        
        # Interactive distribution plot
        if len(numeric_cols) > 0:
            fig = px.histogram(self.df, x=numeric_cols[0], nbins=30,
                             title=f'Interactive Distribution: {numeric_cols[0]}')
            
            chart_path = self.charts_dir / "interactive_histogram.html"
            fig.write_html(chart_path)
            
            charts_created.append({
                'type': 'interactive_histogram',
                'column': numeric_cols[0],
                'path': str(chart_path),
                'description': f'Interactive histogram of {numeric_cols[0]}'
            })
        
        return charts_created
    
    def generate_all_visualizations(self):
        """Generate all types of visualizations"""
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("🔍 Generating automatic insights...")
        insights = self.generate_automatic_insights()
        
        print("📊 Creating visualizations...")
        all_charts = []
        
        # Generate different types of charts
        all_charts.extend(self.create_distribution_plots())
        all_charts.extend(self.create_categorical_plots())
        all_charts.extend(self.create_correlation_plots())
        all_charts.extend(self.create_interactive_plots())
        
        self.generated_charts = all_charts
        
        # Generate summary report
        self.generate_summary_report(insights, all_charts)
        
        print(f"✅ Generated {len(all_charts)} visualizations")
        return all_charts
    
    def generate_summary_report(self, insights, charts):
        """Generate a comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Automated Data Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .insight {{ background: #e8f5e9; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .chart-section {{ margin: 30px 0; }}
                .chart-item {{ border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 5px; }}
                .chart-item img {{ max-width: 100%; height: auto; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 Automated Data Analysis Report</h1>
                <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📋 Key Insights</h2>
        """
        
        for insight in insights:
            html_content += f'<div class="insight">• {insight}</div>\n'
        
        html_content += '<h2>📊 Generated Visualizations</h2>\n'
        
        for chart in charts:
            if chart['path'].endswith('.png'):
                # Convert absolute path to relative for HTML
                rel_path = os.path.relpath(chart['path'], self.reports_dir)
                html_content += f"""
                <div class="chart-item">
                    <h3>{chart['description']}</h3>
                    <p><strong>Type:</strong> {chart['type']} | <strong>Columns:</strong> {chart.get('columns', chart.get('column', 'N/A'))}</p>
                    <img src="{rel_path}" alt="{chart['description']}">
                </div>
                """
            elif chart['path'].endswith('.html'):
                html_content += f"""
                <div class="chart-item">
                    <h3>{chart['description']}</h3>
                    <p><strong>Type:</strong> {chart['type']} | <strong>Columns:</strong> {chart.get('columns', chart.get('column', 'N/A'))}</p>
                    <p><a href="{os.path.relpath(chart['path'], self.reports_dir)}" target="_blank">View Interactive Chart</a></p>
                </div>
                """
        
        html_content += """
            <div class="header" style="margin-top: 40px;">
                <h3>🎯 Summary</h3>
                <p>This automated analysis identified key patterns in your data and generated relevant visualizations. 
                Review the charts above to gain insights into your dataset's structure and relationships.</p>
            </div>
        </body>
        </html>
        """
        
        report_path = self.reports_dir / "analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Comprehensive report saved: {report_path}")
        return str(report_path)

def main():
    """Main function to run the visualization engine"""
    print("🚀 GPT-Powered Automated Visualization Engine")
    print("=" * 50)
    
    # Initialize the engine
    engine = AutoVizEngine()
    
    # Get file path from command line or user input
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to your CSV/Excel file: ").strip()
    
    # Load and analyze data
    if engine.load_data(file_path):
        charts = engine.generate_all_visualizations()
        
        print("\n" + "=" * 50)
        print("🎉 Analysis Complete!")
        print(f"📁 Output directory: {engine.output_dir}")
        print(f"📊 Charts directory: {engine.charts_dir}")
        print(f"📄 Reports directory: {engine.reports_dir}")
        print("\n💡 Key Insights:")
        for insight in engine.insights:
            print(f"   • {insight}")
        
        print(f"\n📈 Generated {len(charts)} visualizations:")
        for chart in charts:
            print(f"   • {chart['type']}: {chart['description']}")
    
    else:
        print("❌ Failed to load data. Please check your file path and format.")

if __name__ == "__main__":
    main()