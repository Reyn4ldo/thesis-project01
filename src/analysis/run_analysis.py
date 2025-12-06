"""
Main runner for comprehensive AMR surveillance analysis.
Combines descriptive analysis and unsupervised learning.
"""

import argparse
from pathlib import Path

from descriptive_analysis import run_descriptive_analysis
from unsupervised_learning import run_unsupervised_analysis


def run_full_analysis(
    input_file: Path,
    output_dir: Path,
    n_components: int = 2,
    n_clusters_range: tuple = (2, 10)
):
    """
    Run complete analysis pipeline.

    Args:
        input_file: Path to cleaned AMR data CSV
        output_dir: Directory to save all results
        n_components: Number of components for dimensionality reduction
        n_clusters_range: Range of clusters to try
    """
    print("=" * 80)
    print("COMPREHENSIVE AMR SURVEILLANCE ANALYSIS")
    print("=" * 80)
    print()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    descriptive_dir = output_dir / "descriptive"
    unsupervised_dir = output_dir / "unsupervised"
    descriptive_dir.mkdir(exist_ok=True)
    unsupervised_dir.mkdir(exist_ok=True)

    print("PHASE 1: DESCRIPTIVE ANALYSIS")
    print("=" * 80)
    desc_analyzer = run_descriptive_analysis(input_file, descriptive_dir)
    print()

    print("PHASE 2: UNSUPERVISED LEARNING")
    print("=" * 80)
    unsup_analyzer = run_unsupervised_analysis(
        input_file,
        unsupervised_dir,
        n_components=n_components,
        n_clusters_range=n_clusters_range
    )
    print()

    # Generate combined summary
    print("=" * 80)
    print("GENERATING COMBINED SUMMARY")
    print("=" * 80)

    combined_report = []
    combined_report.append("=" * 80)
    combined_report.append("COMPREHENSIVE AMR SURVEILLANCE ANALYSIS REPORT")
    combined_report.append("=" * 80)
    combined_report.append("")
    combined_report.append("This report combines descriptive statistics and unsupervised learning")
    combined_report.append("to provide comprehensive insights into AMR patterns.")
    combined_report.append("")
    combined_report.append("=" * 80)
    combined_report.append("")

    # Add descriptive analysis section
    combined_report.append(desc_analyzer.generate_summary_report())
    combined_report.append("")
    combined_report.append("=" * 80)
    combined_report.append("")

    # Add unsupervised learning section
    combined_report.append(unsup_analyzer.generate_summary_report())

    # Save combined report
    combined_report_path = output_dir / "comprehensive_analysis_report.txt"
    combined_report_path.write_text("\n".join(combined_report))
    print(f"✓ Saved comprehensive report to {combined_report_path}")

    # Create summary file
    summary_path = output_dir / "analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write("AMR SURVEILLANCE ANALYSIS - SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        # Descriptive stats
        if desc_analyzer.results.get("basic_stats"):
            stats = desc_analyzer.results["basic_stats"]
            f.write("DATASET OVERVIEW:\n")
            f.write(f"  Total isolates: {stats['total_isolates']}\n")
            if "mdr_count" in stats:
                f.write(f"  MDR isolates: {stats['mdr_count']} ({stats['mdr_prevalence']*100:.1f}%)\n")
            if "mar_index_mean" in stats:
                f.write(f"  Mean MAR index: {stats['mar_index_mean']:.3f}\n")
            f.write("\n")

        # Clustering summary
        f.write("PATTERN DISCOVERY (UNSUPERVISED LEARNING):\n")
        if "kmeans" in unsup_analyzer.clustering_results:
            kmeans = unsup_analyzer.clustering_results["kmeans"]
            f.write(f"  K-Means best K: {kmeans['best_k']}\n")
            f.write(f"  K-Means silhouette: {kmeans['best_silhouette']:.3f}\n")
        if "hierarchical" in unsup_analyzer.clustering_results:
            hier = unsup_analyzer.clustering_results["hierarchical"]
            f.write(f"  Hierarchical best K: {hier['best_k']}\n")
            f.write(f"  Hierarchical silhouette: {hier['best_silhouette']:.3f}\n")
        if "dbscan" in unsup_analyzer.clustering_results:
            dbscan = unsup_analyzer.clustering_results["dbscan"]
            f.write(f"  DBSCAN clusters: {dbscan['n_clusters']}\n")
            f.write(f"  DBSCAN noise points: {dbscan['n_noise']}\n")
        f.write("\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"  Descriptive analysis: {descriptive_dir}/\n")
        f.write(f"  Unsupervised learning: {unsupervised_dir}/\n")
        f.write(f"  Comprehensive report: {combined_report_path}\n")

    print(f"✓ Saved summary to {summary_path}")
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print()

    return desc_analyzer, unsup_analyzer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive AMR surveillance analysis"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to cleaned AMR data CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of components for dimensionality reduction (default: 2)"
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Minimum number of clusters to try (default: 2)"
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=10,
        help="Maximum number of clusters to try (default: 10)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    output_path = Path(args.output)

    run_full_analysis(
        input_file=input_path,
        output_dir=output_path,
        n_components=args.n_components,
        n_clusters_range=(args.min_clusters, args.max_clusters)
    )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
