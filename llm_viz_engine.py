#!/usr/bin/env python3
"""
Enhanced GPT-Powered Automated Visualization Engine with Ollama Integration
Generates visualizations based on natural language prompts using local LLM
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
import requests
from datetime import datetime
from pathlib import Path
import warnings
import re
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore")

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class OllamaLLMInterface:
    """Interface for communicating with Ollama local LLM"""

    def __init__(self, base_url="http://localhost:11434", model="llama3.2"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_response(self, prompt: str) -> str:
        """Generate response from Ollama LLM"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "top_p": 0.9, "max_tokens": 1000},
            }

            response = self.session.post(
                f"{self.base_url}/api/generate", json=payload, timeout=30
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code}"

        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"


class PromptBasedVizEngine:
    """Enhanced visualization engine with prompt-based generation"""

    def __init__(self, output_dir="outputs", ollama_model="llama3.2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.df = None
        self.data_summary = {}
        self.generated_charts = []

        # Initialize Ollama LLM
        self.llm = OllamaLLMInterface(model=ollama_model)
        self.llm_available = self.llm.is_available()

        if not self.llm_available:
            print("⚠️  Ollama not available. Falling back to rule-based approach.")
        else:
            print(f"✅ Connected to Ollama with model: {ollama_model}")

    def load_data(self, file_path):
        """Load CSV or Excel data with robust error handling"""
        try:
            file_path = Path(file_path)
            if file_path.suffix.lower() == ".csv":
                # Try different encodings and separators
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        self.df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV file")

            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")

            # Generate data summary for LLM context
            self._generate_data_summary()

            print(
                f"✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns"
            )
            return True

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

    def _generate_data_summary(self):
        """Generate comprehensive data summary for LLM context"""
        if self.df is None:
            return

        numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(self.df.select_dtypes(include=["object"]).columns)
        datetime_cols = []

        # Detect datetime columns
        for col in categorical_cols:
            try:
                pd.to_datetime(self.df[col].dropna().head(100))
                datetime_cols.append(col)
            except:
                pass

        self.data_summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "sample_data": self.df.head(3).to_dict("records"),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_stats": (
                self.df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_stats": {
                col: self.df[col].value_counts().head(5).to_dict()
                for col in categorical_cols[:3]
            },
        }

    def parse_visualization_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """Use LLM to parse user prompt and generate visualization instructions"""
        if not self.llm_available:
            return self._fallback_prompt_parsing(user_prompt)

        context_prompt = f"""
You are a data visualization expert. Based on the user's request and the dataset information, provide a JSON response with visualization instructions.

Dataset Information:
- Shape: {self.data_summary['shape']}
- Numeric columns: {self.data_summary['numeric_columns']}
- Categorical columns: {self.data_summary['categorical_columns']}
- DateTime columns: {self.data_summary['datetime_columns']}
- Sample data: {self.data_summary['sample_data']}

User Request: "{user_prompt}"

Respond with ONLY a JSON object in this exact format:
{{
    "chart_type": "one of: histogram, scatter, bar, line, heatmap, box, violin, pie, correlation_matrix, time_series",
    "x_column": "column name or null",
    "y_column": "column name or null", 
    "color_column": "column name or null",
    "title": "descriptive chart title",
    "description": "brief description of what the chart shows",
    "additional_params": {{
        "bins": 30,
        "figsize": [10, 6],
        "groupby": "column name or null"
    }}
}}
"""

        response = self.llm.generate_response(context_prompt)

        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                viz_instructions = json.loads(json_match.group())
                return viz_instructions
            else:
                raise ValueError("No valid JSON found in response")

        except Exception as e:
            print(f"⚠️  Error parsing LLM response: {e}")
            return self._fallback_prompt_parsing(user_prompt)

    def _fallback_prompt_parsing(self, user_prompt: str) -> Dict[str, Any]:
        """Fallback rule-based prompt parsing when LLM is not available"""
        prompt_lower = user_prompt.lower()

        # Default chart configuration
        viz_config = {
            "chart_type": "histogram",
            "x_column": None,
            "y_column": None,
            "color_column": None,
            "title": "Data Visualization",
            "description": "Automated visualization based on prompt",
            "additional_params": {"bins": 30, "figsize": [10, 6], "groupby": None},
        }

        # Simple keyword-based parsing
        if any(
            word in prompt_lower for word in ["scatter", "relationship", "correlation"]
        ):
            viz_config["chart_type"] = "scatter"
            if len(self.data_summary["numeric_columns"]) >= 2:
                viz_config["x_column"] = self.data_summary["numeric_columns"][0]
                viz_config["y_column"] = self.data_summary["numeric_columns"][1]

        elif any(
            word in prompt_lower
            for word in ["bar", "count", "category", "distribution"]
        ):
            viz_config["chart_type"] = "bar"
            if self.data_summary["categorical_columns"]:
                viz_config["x_column"] = self.data_summary["categorical_columns"][0]

        elif any(
            word in prompt_lower for word in ["line", "trend", "time", "over time"]
        ):
            viz_config["chart_type"] = "line"
            if self.data_summary["datetime_columns"]:
                viz_config["x_column"] = self.data_summary["datetime_columns"][0]
                if self.data_summary["numeric_columns"]:
                    viz_config["y_column"] = self.data_summary["numeric_columns"][0]

        elif any(word in prompt_lower for word in ["heatmap", "correlation matrix"]):
            viz_config["chart_type"] = "correlation_matrix"

        elif any(word in prompt_lower for word in ["histogram", "distribution"]):
            viz_config["chart_type"] = "histogram"
            if self.data_summary["numeric_columns"]:
                viz_config["x_column"] = self.data_summary["numeric_columns"][0]

        # Extract column names mentioned in prompt
        for col in self.data_summary["columns"]:
            if col.lower() in prompt_lower:
                if viz_config["x_column"] is None:
                    viz_config["x_column"] = col
                elif viz_config["y_column"] is None and col != viz_config["x_column"]:
                    viz_config["y_column"] = col

        return viz_config

    def create_chart_from_config(
        self, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a chart based on the configuration from LLM"""
        try:
            chart_type = config.get("chart_type", "histogram")
            x_col = config.get("x_column")
            y_col = config.get("y_column")
            color_col = config.get("color_column")
            title = config.get("title", "Generated Visualization")
            additional_params = config.get("additional_params", {})

            figsize = additional_params.get("figsize", [10, 6])

            # Generate filename
            chart_filename = (
                f"{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            chart_path = self.charts_dir / chart_filename

            plt.figure(figsize=figsize)

            if chart_type == "histogram":
                if x_col and x_col in self.df.columns:
                    bins = additional_params.get("bins", 30)
                    self.df[x_col].hist(bins=bins, alpha=0.7, edgecolor="black")
                    plt.xlabel(x_col)
                    plt.ylabel("Frequency")
                else:
                    return None

            elif chart_type == "scatter":
                if (
                    x_col
                    and y_col
                    and x_col in self.df.columns
                    and y_col in self.df.columns
                ):
                    if color_col and color_col in self.df.columns:
                        scatter = plt.scatter(
                            self.df[x_col],
                            self.df[y_col],
                            c=pd.Categorical(self.df[color_col]).codes,
                            alpha=0.6,
                            cmap="viridis",
                        )
                        plt.colorbar(scatter, label=color_col)
                    else:
                        plt.scatter(self.df[x_col], self.df[y_col], alpha=0.6)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                else:
                    return None

            elif chart_type == "bar":
                if x_col and x_col in self.df.columns:
                    if self.df[x_col].dtype == "object":
                        value_counts = self.df[x_col].value_counts().head(10)
                        value_counts.plot(kind="bar", color="lightblue")
                        plt.xticks(rotation=45)
                    else:
                        return None
                else:
                    return None

            elif chart_type == "line":
                if (
                    x_col
                    and y_col
                    and x_col in self.df.columns
                    and y_col in self.df.columns
                ):
                    # Sort by x column if it's datetime
                    if x_col in self.data_summary["datetime_columns"]:
                        df_sorted = self.df.copy()
                        df_sorted[x_col] = pd.to_datetime(df_sorted[x_col])
                        df_sorted = df_sorted.sort_values(x_col)
                        plt.plot(df_sorted[x_col], df_sorted[y_col])
                    else:
                        plt.plot(self.df[x_col], self.df[y_col])
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                else:
                    return None

            elif chart_type == "correlation_matrix":
                numeric_cols = self.data_summary["numeric_columns"]
                if len(numeric_cols) >= 2:
                    corr_matrix = self.df[numeric_cols].corr()
                    sns.heatmap(
                        corr_matrix,
                        annot=True,
                        cmap="coolwarm",
                        center=0,
                        square=True,
                        fmt=".2f",
                    )
                else:
                    return None

            elif chart_type == "box":
                if x_col and x_col in self.df.columns:
                    if y_col and y_col in self.df.columns:
                        sns.boxplot(data=self.df, x=x_col, y=y_col)
                    else:
                        self.df.boxplot(column=x_col)
                else:
                    return None

            elif chart_type == "pie":
                if (
                    x_col
                    and x_col in self.df.columns
                    and self.df[x_col].dtype == "object"
                ):
                    value_counts = self.df[x_col].value_counts().head(8)
                    plt.pie(
                        value_counts.values,
                        labels=value_counts.index,
                        autopct="%1.1f%%",
                    )
                else:
                    return None

            else:
                print(f"⚠️  Unsupported chart type: {chart_type}")
                return None

            plt.title(title)
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            chart_info = {
                "type": chart_type,
                "path": str(chart_path),
                "title": title,
                "description": config.get("description", "Generated visualization"),
                "x_column": x_col,
                "y_column": y_col,
                "color_column": color_col,
                "config": config,
            }

            return chart_info

        except Exception as e:
            print(f"❌ Error creating chart: {str(e)}")
            plt.close()
            return None

    def create_interactive_chart(
        self, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create interactive charts using Plotly"""
        try:
            chart_type = config.get("chart_type", "histogram")
            x_col = config.get("x_column")
            y_col = config.get("y_column")
            color_col = config.get("color_column")
            title = config.get("title", "Interactive Visualization")

            chart_filename = f"interactive_{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            chart_path = self.charts_dir / chart_filename

            fig = None

            if chart_type == "scatter" and x_col and y_col:
                fig = px.scatter(
                    self.df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=title,
                    hover_data=self.data_summary["numeric_columns"][:3],
                )

            elif chart_type == "histogram" and x_col:
                fig = px.histogram(self.df, x=x_col, title=title, nbins=30)

            elif chart_type == "bar" and x_col:
                if self.df[x_col].dtype == "object":
                    value_counts = self.df[x_col].value_counts().head(10)
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=title,
                        labels={"x": x_col, "y": "Count"},
                    )

            elif chart_type == "line" and x_col and y_col:
                fig = px.line(self.df, x=x_col, y=y_col, title=title)

            if fig:
                fig.write_html(chart_path)
                return {
                    "type": f"interactive_{chart_type}",
                    "path": str(chart_path),
                    "title": title,
                    "description": f"Interactive {config.get('description', 'visualization')}",
                    "x_column": x_col,
                    "y_column": y_col,
                    "color_column": color_col,
                }

            return None

        except Exception as e:
            print(f"❌ Error creating interactive chart: {str(e)}")
            return None

    def generate_visualization_from_prompt(
        self, user_prompt: str, create_interactive: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate visualization based on user prompt"""
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return []

        print(f"🔍 Processing prompt: '{user_prompt}'")

        # Parse prompt using LLM or fallback
        config = self.parse_visualization_prompt(user_prompt)

        print(f"📊 Generating {config.get('chart_type', 'unknown')} chart...")

        generated_charts = []

        # Create static chart
        static_chart = self.create_chart_from_config(config)
        if static_chart:
            generated_charts.append(static_chart)
            print(f"✅ Created static chart: {static_chart['title']}")

        # Create interactive chart if requested
        if create_interactive and config.get("chart_type") in [
            "scatter",
            "histogram",
            "bar",
            "line",
        ]:
            interactive_chart = self.create_interactive_chart(config)
            if interactive_chart:
                generated_charts.append(interactive_chart)
                print(f"✅ Created interactive chart: {interactive_chart['title']}")

        self.generated_charts.extend(generated_charts)
        return generated_charts

    def generate_summary_report(
        self, charts: List[Dict[str, Any]], user_prompt: str = ""
    ):
        """Generate HTML report for prompt-based visualizations"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prompt-Based Visualization Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; background-color: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .prompt-section {{ background: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #2196f3; }}
                .chart-section {{ margin: 30px 0; }}
                .chart-item {{ background: white; border: 1px solid #e0e0e0; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .chart-item img {{ max-width: 100%; height: auto; border-radius: 5px; }}
                .chart-meta {{ background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
                .interactive-link {{ display: inline-block; background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 10px 0; }}
                .interactive-link:hover {{ background: #45a049; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 AI-Powered Visualization Report</h1>
                <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Visualizations created using {"Ollama LLM" if self.llm_available else "Rule-based parsing"}</p>
            </div>
        """

        if user_prompt:
            html_content += f"""
            <div class="prompt-section">
                <h3>💬 Your Request</h3>
                <p><strong>"{user_prompt}"</strong></p>
            </div>
            """

        html_content += "<h2>📊 Generated Visualizations</h2>\n"

        for i, chart in enumerate(charts, 1):
            html_content += f"""
            <div class="chart-item">
                <h3>Chart {i}: {chart['title']}</h3>
                <div class="chart-meta">
                    <p><strong>Type:</strong> {chart['type']}</p>
                    <p><strong>Description:</strong> {chart['description']}</p>
                    <p><strong>Columns:</strong> X={chart.get('x_column', 'N/A')}, Y={chart.get('y_column', 'N/A')}</p>
                </div>
            """

            if chart["path"].endswith(".png"):
                rel_path = os.path.relpath(chart["path"], self.reports_dir)
                html_content += f'<img src="{rel_path}" alt="{chart['title']}">'
            elif chart["path"].endswith(".html"):
                rel_path = os.path.relpath(chart["path"], self.reports_dir)
                html_content += f'<a href="{rel_path}" target="_blank" class="interactive-link">📈 View Interactive Chart</a>'

            html_content += "</div>\n"

        html_content += """
            <div class="header" style="margin-top: 40px;">
                <h3>🎯 Summary</h3>
                <p>These visualizations were generated based on your natural language request. 
                The AI analyzed your prompt and dataset to create the most appropriate charts.</p>
            </div>
        </body>
        </html>
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"prompt_analysis_report_{timestamp}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📄 Report saved: {report_path}")
        return str(report_path)


def main():
    """Main function to run the enhanced visualization engine"""
    print("🚀 Enhanced AI-Powered Visualization Engine with Ollama")
    print("=" * 60)

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python enhanced_viz_engine.py <data_file> [ollama_model]")
        print("Example: python enhanced_viz_engine.py data.csv llama3.2")
        return

    file_path = sys.argv[1]
    ollama_model = sys.argv[2] if len(sys.argv) > 2 else "llama3.2"

    # Initialize the engine
    engine = PromptBasedVizEngine(ollama_model=ollama_model)

    # Load data
    if not engine.load_data(file_path):
        return

    print("\n" + "=" * 60)
    print("🎯 Interactive Visualization Generation")
    print("Enter your visualization requests (type 'quit' to exit)")
    print("Examples:")
    print("  - 'Show me a scatter plot of price vs quantity'")
    print("  - 'Create a histogram of sales by region'")
    print("  - 'Display correlation between all numeric columns'")
    print("  - 'Show trends over time for revenue'")
    print("=" * 60)

    all_generated_charts = []

    while True:
        try:
            user_prompt = input("\n💬 Your request: ").strip()

            if user_prompt.lower() in ["quit", "exit", "q"]:
                break

            if not user_prompt:
                continue

            # Generate visualization based on prompt
            charts = engine.generate_visualization_from_prompt(user_prompt)

            if charts:
                all_generated_charts.extend(charts)
                print(f"✅ Generated {len(charts)} visualization(s)")

                # Generate individual report for this prompt
                engine.generate_summary_report(charts, user_prompt)
            else:
                print(
                    "❌ Could not generate visualization. Please try rephrasing your request."
                )

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    # Generate final comprehensive report
    if all_generated_charts:
        print(f"\n📊 Generated {len(all_generated_charts)} total visualizations")
        final_report = engine.generate_summary_report(
            all_generated_charts, "Multiple requests"
        )
        print(f"📄 Final report: {final_report}")

    print(f"\n📁 All outputs saved to: {engine.output_dir}")


if __name__ == "__main__":
    main()
