"""Hito 4 — Clustering exploratorio (K-Means) sobre las 90 features acusticas del Hito 3.

Ejecuta un agrupamiento no supervisado (K-Means, k=16) sobre las features estandarizadas,
compara la particion resultante contra las especies reales (ARI, Silhouette) y explora su
relacion con el ruido de fondo medido en Hito 2/3 (snr_db, inband_energy_ratio).

No entrena ni evalua un clasificador: es un diagnostico exploratorio de separabilidad.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import print_header  # noqa: E402

FEATURES_CSV = PROJECT_ROOT / "src" / "data" / "features" / "features.csv"
HITO3_SUMMARY = PROJECT_ROOT / "experiments" / "results" / "hito3" / "hito3_summary.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "hito4_5" / "clustering"
SEED = 42
N_CLUSTERS = 16
SILHOUETTE_SAMPLE_SIZE = 17893


def load_feature_columns() -> list[str]:
    with open(HITO3_SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)
    return list(summary["feature_columns"])


def run_hito4_clustering(seed: int = SEED, n_clusters: int = N_CLUSTERS) -> dict:
    print_header("Hito 4 — Clustering exploratorio (K-Means) sobre features acusticas")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = load_feature_columns()
    df = pd.read_csv(FEATURES_CSV)
    df = df[df["partition"] != "test"].reset_index(drop=True)
    print(f"Clips usados (train+val, sin test): {len(df):,}")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y_species = df["primary_label"].to_numpy()

    train_mask = (df["partition"] == "train").to_numpy()
    scaler = StandardScaler().fit(X[train_mask])
    X_scaled = scaler.transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    ari = float(adjusted_rand_score(y_species, cluster_labels))
    sil = float(
        silhouette_score(
            X_scaled, cluster_labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=seed
        )
    )
    print(f"Adjusted Rand Index (cluster vs especie real): {ari:.4f}")
    print(f"Silhouette Score (muestra de {SILHOUETTE_SAMPLE_SIZE}): {sil:.4f}")

    # Proyeccion PCA 3D para visualizacion (ajustada sobre train, igual criterio que Hito 3)
    pca = PCA(n_components=3, random_state=seed).fit(X_scaled[train_mask])
    coords = pca.transform(X_scaled)
    df_out = df[["clip_id", "recording_id", "primary_label", "author", "partition"]].copy()
    df_out["cluster_id"] = cluster_labels
    df_out["pca_1"] = coords[:, 0]
    df_out["pca_2"] = coords[:, 1]
    df_out["pca_3"] = coords[:, 2]
    df_out.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

    # Distancia de cada clip a su centroide asignado (en espacio escalado de 90 dims)
    centroids = kmeans.cluster_centers_
    dist_to_centroid = np.linalg.norm(X_scaled - centroids[cluster_labels], axis=1)

    # Especie mayoritaria de cada cluster -> clip "alineado" si su especie coincide con la mayoria del cluster
    cluster_majority = (
        pd.Series(y_species).groupby(cluster_labels).agg(lambda s: s.value_counts().idxmax())
    )
    majority_per_clip = cluster_majority.reindex(cluster_labels).to_numpy()
    aligned = majority_per_clip == y_species

    snr = df["snr_db"].to_numpy(dtype=np.float64)
    inband = df["inband_energy_ratio"].to_numpy(dtype=np.float64)
    rho_snr, p_snr = spearmanr(dist_to_centroid, snr)
    rho_inband, p_inband = spearmanr(dist_to_centroid, inband)
    snr_aligned_mean = float(np.mean(snr[aligned]))
    snr_misaligned_mean = float(np.mean(snr[~aligned]))

    print(f"Clips alineados con especie mayoritaria de su cluster: {aligned.sum():,}/{len(aligned):,}")
    print(f"SNR medio (alineados): {snr_aligned_mean:.2f} dB | SNR medio (no alineados): {snr_misaligned_mean:.2f} dB")
    print(f"Spearman(distancia a centroide, snr_db) = {rho_snr:.4f} (p={p_snr:.2e})")
    print(f"Spearman(distancia a centroide, inband_energy_ratio) = {rho_inband:.4f} (p={p_inband:.2e})")

    metrics = {
        "status": "FULL_EXECUTION",
        "seed": seed,
        "n_clusters": n_clusters,
        "n_clips_used": int(len(df)),
        "feature_columns_used": len(feature_cols),
        "adjusted_rand_index": ari,
        "silhouette_score": sil,
        "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "n_clips_aligned_with_cluster_majority_species": int(aligned.sum()),
        "n_clips_total": int(len(aligned)),
        "snr_db_mean_aligned": snr_aligned_mean,
        "snr_db_mean_misaligned": snr_misaligned_mean,
        "spearman_dist_vs_snr_db": {"rho": float(rho_snr), "p_value": float(p_snr)},
        "spearman_dist_vs_inband_energy_ratio": {"rho": float(rho_inband), "p_value": float(p_inband)},
    }
    with open(OUT_DIR / "clustering_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _make_figures(coords, cluster_labels, y_species, dist_to_centroid, snr, aligned, figures_dir)
    print(f"\nResultados guardados en: {OUT_DIR}")
    return metrics


def _make_figures(coords, cluster_labels, y_species, dist_to_centroid, snr, aligned, figures_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    species_codes, species_idx = np.unique(y_species, return_inverse=True)

    # --- clusters_2d.png: cluster vs especie real, lado a lado ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap="tab20", s=6, alpha=0.6)
    axes[0].set_title("Coloreado por cluster (K-Means, k=16)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    sc1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=species_idx, cmap="tab20", s=6, alpha=0.6)
    axes[1].set_title("Coloreado por especie real (primary_label)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.suptitle("Hito 4 — Proyeccion PCA 2D: cluster K-Means vs. especie real")
    fig.tight_layout()
    fig.savefig(figures_dir / "clusters_2d.png", dpi=130)
    plt.close(fig)

    # --- clusters_3d.png ---
    fig = plt.figure(figsize=(14, 6))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax0.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=cluster_labels, cmap="tab20", s=5, alpha=0.5)
    ax0.set_title("Cluster K-Means")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=species_idx, cmap="tab20", s=5, alpha=0.5)
    ax1.set_title("Especie real")
    fig.suptitle("Hito 4 — Proyeccion PCA 3D: cluster K-Means vs. especie real")
    fig.tight_layout()
    fig.savefig(figures_dir / "clusters_3d.png", dpi=130)
    plt.close(fig)

    # --- clusters_vs_noise.png ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Alineado\ncon cluster", "No alineado\ncon cluster"]
    try:
        axes[0].boxplot([snr[aligned], snr[~aligned]], tick_labels=labels)
    except TypeError:
        axes[0].boxplot([snr[aligned], snr[~aligned]], labels=labels)
    axes[0].set_ylabel("SNR (dB)")
    axes[0].set_title("SNR segun alineacion cluster-especie")
    axes[1].scatter(dist_to_centroid, snr, s=4, alpha=0.25)
    axes[1].set_xlabel("Distancia al centroide asignado (espacio escalado)")
    axes[1].set_ylabel("SNR (dB)")
    axes[1].set_title("Distancia al centroide vs. SNR")
    fig.suptitle("Hito 4 — Relacion entre agrupamiento y ruido de fondo")
    fig.tight_layout()
    fig.savefig(figures_dir / "clusters_vs_noise.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run_hito4_clustering()
