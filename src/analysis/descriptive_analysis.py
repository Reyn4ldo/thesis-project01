"""
Descriptive Analysis for AMR Surveillance Data.
Reproduces key findings from surveillance data including resistance patterns,
prevalence rates, and correlations.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


class DescriptiveAnalyzer:
    """Performs comprehensive descriptive analysis of AMR surveillance data."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with cleaned AMR data.

        Args:
            data: Cleaned AMR surveillance dataframe
        """
        self.data = data.copy()
        self.antibiotic_cols = self._get_antibiotic_columns()
        self.results = {}

    def _get_antibiotic_columns(self) -> List[str]:
        """Identify antibiotic result columns."""
        antibiotic_suffixes = ["_int", "_encoded"]
        cols = []
        for col in self.data.columns:
            if any(col.endswith(suffix) for suffix in antibiotic_suffixes):
                if col.endswith("_encoded"):
                    cols.append(col)
        return sorted(cols)

    def analyze_all(self) -> Dict:
        """
        Run all descriptive analyses.

        Returns:
            Dictionary containing all analysis results
        """
        print("Running comprehensive descriptive analysis...")

        # Basic statistics
        self.results["basic_stats"] = self.get_basic_statistics()

        # Resistance prevalence
        self.results["resistance_prevalence"] = self.calculate_resistance_prevalence()

        # MDR analysis
        self.results["mdr_analysis"] = self.analyze_mdr_patterns()

        # Resistance by categorical variables
        self.results["resistance_by_source"] = self.analyze_by_category("sample_source")
        self.results["resistance_by_species"] = self.analyze_by_category("bacterial_species")
        self.results["resistance_by_region"] = self.analyze_by_category(
            "administrative_region"
        )

        # Correlation analysis
        self.results["correlations"] = self.calculate_resistance_correlations()

        # MAR index analysis
        self.results["mar_analysis"] = self.analyze_mar_index()

        print("✓ Descriptive analysis complete")
        return self.results

    def get_basic_statistics(self) -> Dict:
        """Calculate basic dataset statistics."""
        stats = {
            "total_isolates": len(self.data),
            "total_features": len(self.data.columns),
            "antibiotic_tests": len(self.antibiotic_cols),
        }

        # Count by categorical variables
        for col in ["bacterial_species", "sample_source", "administrative_region"]:
            if col in self.data.columns:
                stats[f"{col}_counts"] = self.data[col].value_counts().to_dict()

        # MDR statistics
        if "is_mdr" in self.data.columns:
            stats["mdr_count"] = int(self.data["is_mdr"].sum())
            stats["mdr_prevalence"] = float(self.data["is_mdr"].mean())
            stats["non_mdr_count"] = int((self.data["is_mdr"] == 0).sum())

        # MAR index statistics
        if "mar_index_calculated" in self.data.columns:
            stats["mar_index_mean"] = float(self.data["mar_index_calculated"].mean())
            stats["mar_index_std"] = float(self.data["mar_index_calculated"].std())
            stats["mar_index_min"] = float(self.data["mar_index_calculated"].min())
            stats["mar_index_max"] = float(self.data["mar_index_calculated"].max())
            stats["mar_index_median"] = float(
                self.data["mar_index_calculated"].median()
            )

        return stats

    def calculate_resistance_prevalence(self) -> pd.DataFrame:
        """
        Calculate resistance prevalence for each antibiotic.

        Returns:
            DataFrame with resistance statistics per antibiotic
        """
        prevalence_data = []

        for col in self.antibiotic_cols:
            # Extract antibiotic name
            antibiotic_name = col.replace("_encoded", "")

            # Count resistance levels (0=S, 1=I, 2=R)
            total = self.data[col].notna().sum()
            if total == 0:
                continue

            susceptible = (self.data[col] == 0).sum()
            intermediate = (self.data[col] == 1).sum()
            resistant = (self.data[col] == 2).sum()

            prevalence_data.append(
                {
                    "antibiotic": antibiotic_name,
                    "total_tested": total,
                    "susceptible_count": susceptible,
                    "intermediate_count": intermediate,
                    "resistant_count": resistant,
                    "susceptible_pct": (susceptible / total) * 100,
                    "intermediate_pct": (intermediate / total) * 100,
                    "resistant_pct": (resistant / total) * 100,
                    "resistance_rate": (resistant / total) * 100,
                }
            )

        df = pd.DataFrame(prevalence_data)
        df = df.sort_values("resistance_rate", ascending=False)
        return df

    def analyze_mdr_patterns(self) -> Dict:
        """Analyze MDR patterns in detail."""
        if "is_mdr" not in self.data.columns:
            return {"error": "MDR classification not found"}

        analysis = {}

        # Overall MDR statistics
        mdr_isolates = self.data[self.data["is_mdr"] == 1]
        non_mdr_isolates = self.data[self.data["is_mdr"] == 0]

        analysis["mdr_count"] = len(mdr_isolates)
        analysis["non_mdr_count"] = len(non_mdr_isolates)
        analysis["mdr_prevalence_pct"] = (len(mdr_isolates) / len(self.data)) * 100

        # MDR by species
        if "bacterial_species" in self.data.columns:
            mdr_by_species = (
                self.data.groupby("bacterial_species")["is_mdr"]
                .agg(["sum", "count", "mean"])
                .reset_index()
            )
            mdr_by_species.columns = [
                "species",
                "mdr_count",
                "total_count",
                "mdr_rate",
            ]
            mdr_by_species["mdr_rate"] *= 100
            analysis["mdr_by_species"] = mdr_by_species.to_dict("records")

        # MDR by source
        if "sample_source" in self.data.columns:
            mdr_by_source = (
                self.data.groupby("sample_source")["is_mdr"]
                .agg(["sum", "count", "mean"])
                .reset_index()
            )
            mdr_by_source.columns = [
                "source",
                "mdr_count",
                "total_count",
                "mdr_rate",
            ]
            mdr_by_source["mdr_rate"] *= 100
            analysis["mdr_by_source"] = mdr_by_source.to_dict("records")

        # MDR by region
        if "administrative_region" in self.data.columns:
            mdr_by_region = (
                self.data.groupby("administrative_region")["is_mdr"]
                .agg(["sum", "count", "mean"])
                .reset_index()
            )
            mdr_by_region.columns = [
                "region",
                "mdr_count",
                "total_count",
                "mdr_rate",
            ]
            mdr_by_region["mdr_rate"] *= 100
            analysis["mdr_by_region"] = mdr_by_region.to_dict("records")

        return analysis

    def analyze_by_category(self, category_col: str) -> Dict:
        """
        Analyze resistance patterns by categorical variable.

        Args:
            category_col: Name of categorical column to analyze

        Returns:
            Dictionary with analysis results
        """
        if category_col not in self.data.columns:
            return {"error": f"Column {category_col} not found"}

        analysis = {}
        analysis["category"] = category_col

        # Count by category
        analysis["counts"] = self.data[category_col].value_counts().to_dict()

        # Resistance rates by category for each antibiotic
        resistance_by_category = []
        for col in self.antibiotic_cols[:5]:  # Top 5 antibiotics
            antibiotic_name = col.replace("_encoded", "")
            for category_val in self.data[category_col].unique():
                if pd.isna(category_val):
                    continue

                subset = self.data[self.data[category_col] == category_val]
                total = subset[col].notna().sum()
                if total > 0:
                    resistant = (subset[col] == 2).sum()
                    resistance_by_category.append(
                        {
                            "category_value": category_val,
                            "antibiotic": antibiotic_name,
                            "total_tested": total,
                            "resistant_count": resistant,
                            "resistance_rate": (resistant / total) * 100,
                        }
                    )

        analysis["resistance_rates"] = resistance_by_category

        return analysis

    def calculate_resistance_correlations(self) -> pd.DataFrame:
        """
        Calculate correlations between antibiotic resistance.

        Returns:
            Correlation matrix for antibiotic resistances
        """
        if not self.antibiotic_cols:
            return pd.DataFrame()

        # Select antibiotic columns
        antibiotic_data = self.data[self.antibiotic_cols].copy()

        # Calculate correlation matrix
        corr_matrix = antibiotic_data.corr()

        # Clean column names for readability
        clean_names = [col.replace("_encoded", "") for col in corr_matrix.columns]
        corr_matrix.columns = clean_names
        corr_matrix.index = clean_names

        return corr_matrix

    def analyze_mar_index(self) -> Dict:
        """Analyze MAR index distribution and patterns."""
        if "mar_index_calculated" not in self.data.columns:
            return {"error": "MAR index not found"}

        analysis = {}

        # Basic statistics
        mar_values = self.data["mar_index_calculated"].dropna()
        analysis["mean"] = float(mar_values.mean())
        analysis["median"] = float(mar_values.median())
        analysis["std"] = float(mar_values.std())
        analysis["min"] = float(mar_values.min())
        analysis["max"] = float(mar_values.max())

        # Quartiles
        analysis["q1"] = float(mar_values.quantile(0.25))
        analysis["q3"] = float(mar_values.quantile(0.75))

        # MAR categories
        analysis["mar_0.0"] = int((mar_values == 0).sum())
        analysis["mar_0.0_0.2"] = int(
            ((mar_values > 0) & (mar_values <= 0.2)).sum()
        )
        analysis["mar_0.2_0.4"] = int(
            ((mar_values > 0.2) & (mar_values <= 0.4)).sum()
        )
        analysis["mar_above_0.4"] = int((mar_values > 0.4).sum())

        # MAR by source
        if "sample_source" in self.data.columns:
            mar_by_source = (
                self.data.groupby("sample_source")["mar_index_calculated"]
                .agg(["mean", "median", "std", "count"])
                .reset_index()
            )
            analysis["mar_by_source"] = mar_by_source.to_dict("records")

        # MAR vs MDR relationship
        if "is_mdr" in self.data.columns:
            mdr_mar = (
                self.data.groupby("is_mdr")["mar_index_calculated"]
                .agg(["mean", "median", "std", "count"])
                .reset_index()
            )
            mdr_mar["is_mdr"] = mdr_mar["is_mdr"].map({0: "Non-MDR", 1: "MDR"})
            analysis["mar_by_mdr"] = mdr_mar.to_dict("records")

        return analysis

    def generate_visualizations(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Generate visualization plots for the analysis.

        Args:
            output_dir: Directory to save plots (if None, returns figures without saving)

        Returns:
            Dictionary mapping plot names to file paths
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        plot_paths = {}

        # 1. Resistance prevalence plot
        if "resistance_prevalence" in self.results:
            fig, ax = plt.subplots(figsize=(12, 6))
            prevalence_df = self.results["resistance_prevalence"]
            top_antibiotics = prevalence_df.head(15)

            x = range(len(top_antibiotics))
            width = 0.25

            ax.bar(
                [i - width for i in x],
                top_antibiotics["susceptible_pct"],
                width,
                label="Susceptible",
                color="green",
                alpha=0.8,
            )
            ax.bar(
                x,
                top_antibiotics["intermediate_pct"],
                width,
                label="Intermediate",
                color="orange",
                alpha=0.8,
            )
            ax.bar(
                [i + width for i in x],
                top_antibiotics["resistant_pct"],
                width,
                label="Resistant",
                color="red",
                alpha=0.8,
            )

            ax.set_xlabel("Antibiotic")
            ax.set_ylabel("Percentage (%)")
            ax.set_title("Antibiotic Susceptibility Profile (Top 15)")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [a.replace("_", " ").title() for a in top_antibiotics["antibiotic"]],
                rotation=45,
                ha="right",
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            if output_dir:
                path = output_dir / "resistance_prevalence.png"
                plt.savefig(path, dpi=300, bbox_inches="tight")
                plot_paths["resistance_prevalence"] = path
            plt.close()

        # 2. MDR prevalence by source
        if "mdr_analysis" in self.results and "mdr_by_source" in self.results["mdr_analysis"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            mdr_data = pd.DataFrame(self.results["mdr_analysis"]["mdr_by_source"])

            ax.bar(mdr_data["source"], mdr_data["mdr_rate"], color="coral", alpha=0.8)
            ax.set_xlabel("Sample Source")
            ax.set_ylabel("MDR Prevalence (%)")
            ax.set_title("Multi-Drug Resistance Prevalence by Sample Source")
            ax.set_xticklabels(
                [s.replace("_", " ").title() for s in mdr_data["source"]],
                rotation=45,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            if output_dir:
                path = output_dir / "mdr_by_source.png"
                plt.savefig(path, dpi=300, bbox_inches="tight")
                plot_paths["mdr_by_source"] = path
            plt.close()

        # 3. Correlation heatmap
        if "correlations" in self.results:
            corr_matrix = self.results["correlations"]
            if not corr_matrix.empty:
                fig, ax = plt.subplots(figsize=(14, 12))
                sns.heatmap(
                    corr_matrix,
                    annot=False,
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    ax=ax,
                )
                ax.set_title(
                    "Correlation Matrix of Antibiotic Resistance", fontsize=14, pad=20
                )
                plt.tight_layout()

                if output_dir:
                    path = output_dir / "resistance_correlation.png"
                    plt.savefig(path, dpi=300, bbox_inches="tight")
                    plot_paths["resistance_correlation"] = path
                plt.close()

        # 4. MAR index distribution
        if "mar_analysis" in self.results and "mar_index_calculated" in self.data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            mar_values = self.data["mar_index_calculated"].dropna()

            ax.hist(mar_values, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axvline(
                mar_values.mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mar_values.mean():.3f}",
            )
            ax.axvline(
                mar_values.median(),
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {mar_values.median():.3f}",
            )
            ax.set_xlabel("MAR Index")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Multiple Antibiotic Resistance (MAR) Index")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            if output_dir:
                path = output_dir / "mar_distribution.png"
                plt.savefig(path, dpi=300, bbox_inches="tight")
                plot_paths["mar_distribution"] = path
            plt.close()

        return plot_paths

    def generate_summary_report(self) -> str:
        """
        Generate a text summary report of the analysis.

        Returns:
            Formatted string with analysis summary
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AMR SURVEILLANCE DATA - DESCRIPTIVE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Basic statistics
        if "basic_stats" in self.results:
            stats = self.results["basic_stats"]
            report_lines.append("DATASET OVERVIEW")
            report_lines.append("-" * 80)
            report_lines.append(f"Total Isolates: {stats['total_isolates']}")
            report_lines.append(f"Antibiotic Tests: {stats['antibiotic_tests']}")
            if "mdr_count" in stats:
                report_lines.append(
                    f"MDR Isolates: {stats['mdr_count']} ({stats['mdr_prevalence']*100:.1f}%)"
                )
                report_lines.append(f"Non-MDR Isolates: {stats['non_mdr_count']}")
            if "mar_index_mean" in stats:
                report_lines.append(f"MAR Index (mean ± std): {stats['mar_index_mean']:.3f} ± {stats['mar_index_std']:.3f}")
                report_lines.append(f"MAR Index Range: [{stats['mar_index_min']:.3f}, {stats['mar_index_max']:.3f}]")
            report_lines.append("")

        # Resistance prevalence
        if "resistance_prevalence" in self.results:
            prevalence_df = self.results["resistance_prevalence"]
            report_lines.append("TOP 10 MOST RESISTANT ANTIBIOTICS")
            report_lines.append("-" * 80)
            top_10 = prevalence_df.head(10)
            for idx, row in top_10.iterrows():
                antibiotic = row["antibiotic"].replace("_", " ").title()
                resistance_rate = row["resistance_rate"]
                report_lines.append(f"{antibiotic:40s}: {resistance_rate:6.1f}%")
            report_lines.append("")

        # MDR analysis
        if "mdr_analysis" in self.results:
            mdr = self.results["mdr_analysis"]
            report_lines.append("MULTI-DRUG RESISTANCE (MDR) ANALYSIS")
            report_lines.append("-" * 80)
            report_lines.append(f"MDR Prevalence: {mdr['mdr_prevalence_pct']:.1f}%")

            if "mdr_by_source" in mdr:
                report_lines.append("\nMDR by Sample Source:")
                for source_data in sorted(
                    mdr["mdr_by_source"], key=lambda x: x["mdr_rate"], reverse=True
                ):
                    source = source_data["source"].replace("_", " ").title()
                    rate = source_data["mdr_rate"]
                    count = source_data["mdr_count"]
                    total = source_data["total_count"]
                    report_lines.append(
                        f"  {source:40s}: {rate:5.1f}% ({count}/{total})"
                    )
            report_lines.append("")

        # MAR index
        if "mar_analysis" in self.results:
            mar = self.results["mar_analysis"]
            if "mean" in mar:
                report_lines.append("MAR INDEX ANALYSIS")
                report_lines.append("-" * 80)
                report_lines.append(f"Mean MAR Index: {mar['mean']:.3f}")
                report_lines.append(f"Median MAR Index: {mar['median']:.3f}")
                report_lines.append(f"Standard Deviation: {mar['std']:.3f}")
                report_lines.append(f"Range: [{mar['min']:.3f}, {mar['max']:.3f}]")

                if "mar_by_mdr" in mar:
                    report_lines.append("\nMAR Index by MDR Status:")
                    for mdr_data in mar["mar_by_mdr"]:
                        status = mdr_data["is_mdr"]
                        mean_mar = mdr_data["mean"]
                        count = mdr_data["count"]
                        report_lines.append(
                            f"  {status:10s}: {mean_mar:.3f} (n={count})"
                        )
                report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def run_descriptive_analysis(
    input_file: Path, output_dir: Optional[Path] = None
) -> DescriptiveAnalyzer:
    """
    Run complete descriptive analysis on AMR data.

    Args:
        input_file: Path to cleaned CSV data
        output_dir: Optional directory to save results

    Returns:
        DescriptiveAnalyzer instance with results
    """
    print(f"Loading data from {input_file}")
    data = pd.read_csv(input_file)

    print(f"Loaded {len(data)} isolates")

    # Initialize analyzer
    analyzer = DescriptiveAnalyzer(data)

    # Run all analyses
    analyzer.analyze_all()

    # Generate visualizations
    if output_dir:
        output_dir = Path(output_dir)
        print(f"Generating visualizations in {output_dir}")
        plot_paths = analyzer.generate_visualizations(output_dir)
        print(f"✓ Generated {len(plot_paths)} plots")

        # Save summary report
        report_path = output_dir / "descriptive_analysis_report.txt"
        report_text = analyzer.generate_summary_report()
        report_path.write_text(report_text)
        print(f"✓ Saved report to {report_path}")

    # Print summary
    print("\n" + analyzer.generate_summary_report())

    return analyzer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python descriptive_analysis.py <input_csv> [output_dir]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    analyzer = run_descriptive_analysis(input_path, output_path)
