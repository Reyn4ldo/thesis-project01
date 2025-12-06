"""
Unsupervised Learning for AMR Pattern Discovery.
Includes dimensionality reduction (PCA, t-SNE, UMAP) and clustering (K-Means, Hierarchical, DBSCAN).
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

warnings.filterwarnings("ignore")


class UnsupervisedAnalyzer:
    """Performs unsupervised learning for AMR pattern discovery."""

    def __init__(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """
        Initialize the analyzer with AMR data.

        Args:
            data: AMR surveillance dataframe
            feature_columns: List of feature columns to use (if None, auto-detect)
        """
        self.data = data.copy()
        
        if feature_columns is None:
            # Auto-detect antibiotic encoded columns
            self.feature_columns = [
                col for col in data.columns if col.endswith("_encoded")
            ]
        else:
            self.feature_columns = feature_columns
        
        # Prepare feature matrix
        self.X = self.data[self.feature_columns].fillna(0).values
        self.X_scaled = None
        self.scaler = None
        
        # Store results
        self.results = {}
        self.dim_reduction_results = {}
        self.clustering_results = {}

    def scale_features(self):
        """Standardize features for dimensionality reduction and clustering."""
        if self.X_scaled is None:
            print("Scaling features...")
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X)
            print(f"✓ Scaled {self.X_scaled.shape[1]} features")

    def run_all(self, n_components: int = 2, n_clusters_range: Tuple[int, int] = (2, 10)):
        """
        Run all unsupervised learning analyses.

        Args:
            n_components: Number of components for dimensionality reduction
            n_clusters_range: Range of clusters to try (min, max)

        Returns:
            Dictionary with all results
        """
        print("Running comprehensive unsupervised learning analysis...")
        
        # Scale features first
        self.scale_features()
        
        # Dimensionality reduction
        print("\n=== Dimensionality Reduction ===")
        self.run_pca(n_components=n_components)
        self.run_tsne(n_components=n_components)
        if UMAP_AVAILABLE:
            self.run_umap(n_components=n_components)
        
        # Clustering
        print("\n=== Clustering Analysis ===")
        self.run_kmeans_analysis(n_clusters_range=n_clusters_range)
        self.run_hierarchical_clustering(n_clusters_range=n_clusters_range)
        self.run_dbscan()
        
        # Analyze discovered patterns
        print("\n=== Pattern Analysis ===")
        self.analyze_patterns()
        
        self.results["dim_reduction"] = self.dim_reduction_results
        self.results["clustering"] = self.clustering_results
        
        print("\n✓ Unsupervised learning analysis complete")
        return self.results

    def run_pca(self, n_components: int = 2):
        """Run Principal Component Analysis."""
        print(f"Running PCA (n_components={n_components})...")
        
        pca = PCA(n_components=min(n_components, self.X_scaled.shape[1]))
        X_pca = pca.fit_transform(self.X_scaled)
        
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        self.dim_reduction_results["pca"] = {
            "components": X_pca,
            "explained_variance": explained_var,
            "cumulative_variance": cumulative_var,
            "n_components": n_components,
            "feature_importance": pca.components_,
            "model": pca
        }
        
        print(f"✓ PCA complete. Explained variance: {cumulative_var[-1]:.2%}")
        return X_pca

    def run_tsne(self, n_components: int = 2, perplexity: int = 30):
        """Run t-SNE (t-Distributed Stochastic Neighbor Embedding)."""
        print(f"Running t-SNE (n_components={n_components}, perplexity={perplexity})...")
        
        # Adjust perplexity if dataset is small
        actual_perplexity = min(perplexity, (len(self.X_scaled) - 1) // 3)
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=actual_perplexity,
            random_state=42,
            max_iter=1000
        )
        X_tsne = tsne.fit_transform(self.X_scaled)
        
        self.dim_reduction_results["tsne"] = {
            "components": X_tsne,
            "n_components": n_components,
            "perplexity": actual_perplexity,
            "kl_divergence": tsne.kl_divergence_
        }
        
        print(f"✓ t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}")
        return X_tsne

    def run_umap(self, n_components: int = 2, n_neighbors: int = 15):
        """Run UMAP (Uniform Manifold Approximation and Projection)."""
        if not UMAP_AVAILABLE:
            print("⚠ UMAP not available, skipping...")
            return None
        
        print(f"Running UMAP (n_components={n_components}, n_neighbors={n_neighbors})...")
        
        # Adjust n_neighbors if dataset is small
        actual_neighbors = min(n_neighbors, len(self.X_scaled) - 1)
        
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=actual_neighbors,
            random_state=42
        )
        X_umap = umap_model.fit_transform(self.X_scaled)
        
        self.dim_reduction_results["umap"] = {
            "components": X_umap,
            "n_components": n_components,
            "n_neighbors": actual_neighbors,
            "model": umap_model
        }
        
        print(f"✓ UMAP complete")
        return X_umap

    def run_kmeans_analysis(self, n_clusters_range: Tuple[int, int] = (2, 10)):
        """
        Run K-Means clustering with multiple K values.

        Args:
            n_clusters_range: Range of K values to try (min, max)
        """
        print(f"Running K-Means analysis (K={n_clusters_range[0]} to {n_clusters_range[1]})...")
        
        results = []
        max_k = min(n_clusters_range[1], len(self.X_scaled) - 1)
        
        for k in range(n_clusters_range[0], max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            silhouette = silhouette_score(self.X_scaled, labels)
            davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(self.X_scaled, labels)
            
            results.append({
                "n_clusters": k,
                "inertia": inertia,
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
                "calinski_harabasz_score": calinski_harabasz,
                "labels": labels,
                "model": kmeans
            })
        
        # Find optimal K using silhouette score
        best_idx = max(range(len(results)), key=lambda i: results[i]["silhouette_score"])
        best_k = results[best_idx]["n_clusters"]
        
        self.clustering_results["kmeans"] = {
            "all_results": results,
            "best_k": best_k,
            "best_labels": results[best_idx]["labels"],
            "best_model": results[best_idx]["model"],
            "best_silhouette": results[best_idx]["silhouette_score"]
        }
        
        print(f"✓ K-Means complete. Best K={best_k} (silhouette={results[best_idx]['silhouette_score']:.3f})")
        return results[best_idx]["labels"]

    def run_hierarchical_clustering(self, n_clusters_range: Tuple[int, int] = (2, 10)):
        """Run Hierarchical Agglomerative Clustering."""
        print(f"Running Hierarchical Clustering...")
        
        results = []
        max_k = min(n_clusters_range[1], len(self.X_scaled) - 1)
        
        for k in range(n_clusters_range[0], max_k + 1):
            hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = hierarchical.fit_predict(self.X_scaled)
            
            # Calculate metrics
            silhouette = silhouette_score(self.X_scaled, labels)
            davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(self.X_scaled, labels)
            
            results.append({
                "n_clusters": k,
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
                "calinski_harabasz_score": calinski_harabasz,
                "labels": labels,
                "model": hierarchical
            })
        
        # Find optimal K
        best_idx = max(range(len(results)), key=lambda i: results[i]["silhouette_score"])
        best_k = results[best_idx]["n_clusters"]
        
        self.clustering_results["hierarchical"] = {
            "all_results": results,
            "best_k": best_k,
            "best_labels": results[best_idx]["labels"],
            "best_silhouette": results[best_idx]["silhouette_score"]
        }
        
        print(f"✓ Hierarchical complete. Best K={best_k} (silhouette={results[best_idx]['silhouette_score']:.3f})")
        return results[best_idx]["labels"]

    def run_dbscan(self, eps: float = 0.5, min_samples: int = 5):
        """Run DBSCAN (Density-Based Spatial Clustering)."""
        print(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        # Adjust min_samples if dataset is small
        actual_min_samples = min(min_samples, len(self.X_scaled) // 20)
        if actual_min_samples < 2:
            actual_min_samples = 2
        
        dbscan = DBSCAN(eps=eps, min_samples=actual_min_samples)
        labels = dbscan.fit_predict(self.X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate metrics only if we have valid clusters
        if n_clusters > 1:
            # Exclude noise points for silhouette score
            mask = labels != -1
            if mask.sum() > n_clusters:
                silhouette = silhouette_score(self.X_scaled[mask], labels[mask])
            else:
                silhouette = -1
        else:
            silhouette = -1
        
        self.clustering_results["dbscan"] = {
            "labels": labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette_score": silhouette,
            "eps": eps,
            "min_samples": actual_min_samples,
            "model": dbscan
        }
        
        print(f"✓ DBSCAN complete. Clusters={n_clusters}, Noise points={n_noise}")
        return labels

    def analyze_patterns(self):
        """Analyze discovered patterns from clustering."""
        print("Analyzing discovered AMR patterns...")
        
        pattern_analysis = {}
        
        # Analyze K-Means clusters
        if "kmeans" in self.clustering_results:
            kmeans_labels = self.clustering_results["kmeans"]["best_labels"]
            pattern_analysis["kmeans_patterns"] = self._analyze_cluster_characteristics(
                kmeans_labels, "K-Means"
            )
        
        # Analyze Hierarchical clusters
        if "hierarchical" in self.clustering_results:
            hier_labels = self.clustering_results["hierarchical"]["best_labels"]
            pattern_analysis["hierarchical_patterns"] = self._analyze_cluster_characteristics(
                hier_labels, "Hierarchical"
            )
        
        # Analyze DBSCAN clusters
        if "dbscan" in self.clustering_results:
            dbscan_labels = self.clustering_results["dbscan"]["labels"]
            pattern_analysis["dbscan_patterns"] = self._analyze_cluster_characteristics(
                dbscan_labels, "DBSCAN"
            )
        
        self.results["pattern_analysis"] = pattern_analysis
        print("✓ Pattern analysis complete")
        return pattern_analysis

    def _analyze_cluster_characteristics(self, labels: np.ndarray, method_name: str) -> Dict:
        """Analyze characteristics of each cluster."""
        unique_labels = np.unique(labels)
        cluster_chars = []
        
        for label in unique_labels:
            if label == -1:  # Noise in DBSCAN
                continue
            
            mask = labels == label
            cluster_data = self.data[mask]
            
            char = {
                "cluster_id": int(label),
                "size": int(mask.sum()),
                "percentage": float((mask.sum() / len(labels)) * 100)
            }
            
            # MDR prevalence in cluster
            if "is_mdr" in cluster_data.columns:
                char["mdr_count"] = int(cluster_data["is_mdr"].sum())
                char["mdr_rate"] = float(cluster_data["is_mdr"].mean())
            
            # MAR index in cluster
            if "mar_index_calculated" in cluster_data.columns:
                char["mean_mar_index"] = float(cluster_data["mar_index_calculated"].mean())
            
            # Dominant species
            if "bacterial_species" in cluster_data.columns:
                species_counts = cluster_data["bacterial_species"].value_counts()
                if len(species_counts) > 0:
                    char["dominant_species"] = species_counts.index[0]
                    char["dominant_species_pct"] = float(
                        (species_counts.iloc[0] / len(cluster_data)) * 100
                    )
            
            # Dominant source
            if "sample_source" in cluster_data.columns:
                source_counts = cluster_data["sample_source"].value_counts()
                if len(source_counts) > 0:
                    char["dominant_source"] = source_counts.index[0]
                    char["dominant_source_pct"] = float(
                        (source_counts.iloc[0] / len(cluster_data)) * 100
                    )
            
            cluster_chars.append(char)
        
        return {
            "method": method_name,
            "n_clusters": len(cluster_chars),
            "clusters": cluster_chars
        }

    def generate_visualizations(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Generate visualizations for unsupervised learning results.

        Args:
            output_dir: Directory to save plots

        Returns:
            Dictionary mapping plot names to file paths
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        # 1. PCA explained variance
        if "pca" in self.dim_reduction_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            pca_data = self.dim_reduction_results["pca"]
            n_comp = len(pca_data["explained_variance"])
            
            # Individual variance
            ax1.bar(range(1, n_comp + 1), pca_data["explained_variance"])
            ax1.set_xlabel("Principal Component")
            ax1.set_ylabel("Explained Variance Ratio")
            ax1.set_title("PCA: Explained Variance by Component")
            ax1.grid(axis="y", alpha=0.3)
            
            # Cumulative variance
            ax2.plot(range(1, n_comp + 1), pca_data["cumulative_variance"], marker='o')
            ax2.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
            ax2.axhline(y=0.9, color='g', linestyle='--', label='90% threshold')
            ax2.set_xlabel("Number of Components")
            ax2.set_ylabel("Cumulative Explained Variance")
            ax2.set_title("PCA: Cumulative Explained Variance")
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            if output_dir:
                path = output_dir / "pca_variance.png"
                plt.savefig(path, dpi=300, bbox_inches="tight")
                plot_paths["pca_variance"] = path
            plt.close()
        
        # 2. Dimensionality reduction scatter plots with MDR coloring
        if "is_mdr" in self.data.columns:
            mdr_labels = self.data["is_mdr"].values
            mdr_colors = ["blue" if mdr == 0 else "red" for mdr in mdr_labels]
            
            n_methods = sum([
                "pca" in self.dim_reduction_results,
                "tsne" in self.dim_reduction_results,
                "umap" in self.dim_reduction_results
            ])
            
            if n_methods > 0:
                fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
                if n_methods == 1:
                    axes = [axes]
                
                ax_idx = 0
                
                # PCA plot
                if "pca" in self.dim_reduction_results:
                    X_pca = self.dim_reduction_results["pca"]["components"]
                    axes[ax_idx].scatter(X_pca[:, 0], X_pca[:, 1], c=mdr_colors, alpha=0.6, s=50)
                    axes[ax_idx].set_xlabel("PC1")
                    axes[ax_idx].set_ylabel("PC2")
                    axes[ax_idx].set_title("PCA Projection (MDR colored)")
                    axes[ax_idx].legend(
                        handles=[
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Non-MDR'),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='MDR')
                        ]
                    )
                    ax_idx += 1
                
                # t-SNE plot
                if "tsne" in self.dim_reduction_results:
                    X_tsne = self.dim_reduction_results["tsne"]["components"]
                    axes[ax_idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=mdr_colors, alpha=0.6, s=50)
                    axes[ax_idx].set_xlabel("t-SNE 1")
                    axes[ax_idx].set_ylabel("t-SNE 2")
                    axes[ax_idx].set_title("t-SNE Projection (MDR colored)")
                    axes[ax_idx].legend(
                        handles=[
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Non-MDR'),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='MDR')
                        ]
                    )
                    ax_idx += 1
                
                # UMAP plot
                if "umap" in self.dim_reduction_results:
                    X_umap = self.dim_reduction_results["umap"]["components"]
                    axes[ax_idx].scatter(X_umap[:, 0], X_umap[:, 1], c=mdr_colors, alpha=0.6, s=50)
                    axes[ax_idx].set_xlabel("UMAP 1")
                    axes[ax_idx].set_ylabel("UMAP 2")
                    axes[ax_idx].set_title("UMAP Projection (MDR colored)")
                    axes[ax_idx].legend(
                        handles=[
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Non-MDR'),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='MDR')
                        ]
                    )
                
                plt.tight_layout()
                if output_dir:
                    path = output_dir / "dimensionality_reduction_mdr.png"
                    plt.savefig(path, dpi=300, bbox_inches="tight")
                    plot_paths["dim_reduction_mdr"] = path
                plt.close()
        
        # 3. K-Means elbow plot
        if "kmeans" in self.clustering_results:
            kmeans_results = self.clustering_results["kmeans"]["all_results"]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            k_values = [r["n_clusters"] for r in kmeans_results]
            inertias = [r["inertia"] for r in kmeans_results]
            silhouettes = [r["silhouette_score"] for r in kmeans_results]
            
            # Elbow plot
            ax1.plot(k_values, inertias, marker='o')
            ax1.set_xlabel("Number of Clusters (K)")
            ax1.set_ylabel("Inertia")
            ax1.set_title("K-Means Elbow Plot")
            ax1.grid(alpha=0.3)
            
            # Silhouette plot
            ax2.plot(k_values, silhouettes, marker='o', color='green')
            best_k = self.clustering_results["kmeans"]["best_k"]
            ax2.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
            ax2.set_xlabel("Number of Clusters (K)")
            ax2.set_ylabel("Silhouette Score")
            ax2.set_title("K-Means Silhouette Score")
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            if output_dir:
                path = output_dir / "kmeans_elbow.png"
                plt.savefig(path, dpi=300, bbox_inches="tight")
                plot_paths["kmeans_elbow"] = path
            plt.close()
        
        # 4. Clustering results on PCA
        if "pca" in self.dim_reduction_results and "kmeans" in self.clustering_results:
            X_pca = self.dim_reduction_results["pca"]["components"]
            kmeans_labels = self.clustering_results["kmeans"]["best_labels"]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6, s=50)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"K-Means Clustering on PCA (K={self.clustering_results['kmeans']['best_k']})")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            
            if output_dir:
                path = output_dir / "clustering_pca.png"
                plt.savefig(path, dpi=300, bbox_inches="tight")
                plot_paths["clustering_pca"] = path
            plt.close()
        
        return plot_paths

    def generate_summary_report(self) -> str:
        """Generate a text summary of unsupervised learning results."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AMR SURVEILLANCE - UNSUPERVISED LEARNING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Dimensionality reduction
        report_lines.append("DIMENSIONALITY REDUCTION")
        report_lines.append("-" * 80)
        
        if "pca" in self.dim_reduction_results:
            pca_data = self.dim_reduction_results["pca"]
            report_lines.append(f"PCA: {pca_data['n_components']} components")
            report_lines.append(f"  Cumulative variance explained: {pca_data['cumulative_variance'][-1]:.2%}")
        
        if "tsne" in self.dim_reduction_results:
            tsne_data = self.dim_reduction_results["tsne"]
            report_lines.append(f"t-SNE: {tsne_data['n_components']} components")
            report_lines.append(f"  KL divergence: {tsne_data['kl_divergence']:.4f}")
        
        if "umap" in self.dim_reduction_results:
            umap_data = self.dim_reduction_results["umap"]
            report_lines.append(f"UMAP: {umap_data['n_components']} components")
        
        report_lines.append("")
        
        # Clustering
        report_lines.append("CLUSTERING ANALYSIS")
        report_lines.append("-" * 80)
        
        if "kmeans" in self.clustering_results:
            kmeans_data = self.clustering_results["kmeans"]
            report_lines.append(f"K-Means: Best K={kmeans_data['best_k']}")
            report_lines.append(f"  Silhouette score: {kmeans_data['best_silhouette']:.3f}")
        
        if "hierarchical" in self.clustering_results:
            hier_data = self.clustering_results["hierarchical"]
            report_lines.append(f"Hierarchical: Best K={hier_data['best_k']}")
            report_lines.append(f"  Silhouette score: {hier_data['best_silhouette']:.3f}")
        
        if "dbscan" in self.clustering_results:
            dbscan_data = self.clustering_results["dbscan"]
            report_lines.append(f"DBSCAN: {dbscan_data['n_clusters']} clusters, {dbscan_data['n_noise']} noise points")
            if dbscan_data['silhouette_score'] > 0:
                report_lines.append(f"  Silhouette score: {dbscan_data['silhouette_score']:.3f}")
        
        report_lines.append("")
        
        # Pattern analysis
        if "pattern_analysis" in self.results:
            report_lines.append("DISCOVERED AMR PATTERNS")
            report_lines.append("-" * 80)
            
            for method_key, pattern_data in self.results["pattern_analysis"].items():
                if pattern_data and "clusters" in pattern_data:
                    report_lines.append(f"\n{pattern_data['method']} Patterns:")
                    for cluster in pattern_data["clusters"]:
                        report_lines.append(f"\n  Cluster {cluster['cluster_id']}:")
                        report_lines.append(f"    Size: {cluster['size']} ({cluster['percentage']:.1f}%)")
                        if "mdr_rate" in cluster:
                            report_lines.append(f"    MDR rate: {cluster['mdr_rate']*100:.1f}%")
                        if "mean_mar_index" in cluster:
                            report_lines.append(f"    Mean MAR index: {cluster['mean_mar_index']:.3f}")
                        if "dominant_species" in cluster:
                            report_lines.append(f"    Dominant species: {cluster['dominant_species']} ({cluster['dominant_species_pct']:.1f}%)")
                        if "dominant_source" in cluster:
                            report_lines.append(f"    Dominant source: {cluster['dominant_source']} ({cluster['dominant_source_pct']:.1f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def run_unsupervised_analysis(
    input_file: Path,
    output_dir: Optional[Path] = None,
    n_components: int = 2,
    n_clusters_range: Tuple[int, int] = (2, 10)
) -> UnsupervisedAnalyzer:
    """
    Run complete unsupervised learning analysis.

    Args:
        input_file: Path to cleaned CSV data
        output_dir: Optional directory to save results
        n_components: Number of components for dimensionality reduction
        n_clusters_range: Range of K values for clustering

    Returns:
        UnsupervisedAnalyzer instance with results
    """
    print(f"Loading data from {input_file}")
    data = pd.read_csv(input_file)
    print(f"Loaded {len(data)} isolates")
    
    # Initialize analyzer
    analyzer = UnsupervisedAnalyzer(data)
    print(f"Using {len(analyzer.feature_columns)} features for analysis")
    
    # Run all analyses
    analyzer.run_all(n_components=n_components, n_clusters_range=n_clusters_range)
    
    # Generate visualizations
    if output_dir:
        output_dir = Path(output_dir)
        print(f"\nGenerating visualizations in {output_dir}")
        plot_paths = analyzer.generate_visualizations(output_dir)
        print(f"✓ Generated {len(plot_paths)} plots")
        
        # Save summary report
        report_path = output_dir / "unsupervised_analysis_report.txt"
        report_text = analyzer.generate_summary_report()
        report_path.write_text(report_text)
        print(f"✓ Saved report to {report_path}")
    
    # Print summary
    print("\n" + analyzer.generate_summary_report())
    
    return analyzer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python unsupervised_learning.py <input_csv> [output_dir]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    analyzer = run_unsupervised_analysis(input_path, output_path)
