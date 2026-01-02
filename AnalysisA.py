import os
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from stl import mesh
from scipy.spatial import cKDTree
from scipy.stats import norm

# ==========================================
# GEOMETRY UTILITIES (From Notebook)
# ==========================================

def load_mesh(path):
    m = mesh.Mesh.from_file(path)
    V = m.vectors.reshape(-1, 3).astype(np.float64)
    F = np.arange(len(V)).reshape(-1, 3)
    return V, F, m

def deduplicate_mesh(V, F, tol=1e-6):
    key = np.round(V / tol).astype(np.int64)
    _, keep_idx, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Vd = V[keep_idx]
    Fd = inverse[F]
    mask = np.all([Fd[:, 0] != Fd[:, 1], Fd[:, 1] != Fd[:, 2], Fd[:, 2] != Fd[:, 0]], axis=0)
    return Vd, Fd[mask]

def scale_center(V, Vref):
    vmin, vmax = V.min(0), V.max(0)
    rmin, rmax = Vref.min(0), Vref.max(0)
    s = np.linalg.norm(rmax - rmin) / np.linalg.norm(vmax - vmin)
    cV, cR = (vmin + vmax)/2, (rmin + rmax)/2
    V1 = (V - cV) * s + cR
    return V1, s, cV, cR

def pca_axes(P):
    P0 = P - P.mean(0)
    _, V = np.linalg.eigh(np.cov(P0.T))
    if np.linalg.det(V) < 0: V[:, 0] *= -1
    return V

def best_pca_rotation(A, B, sample=30000, seed=0):
    VA, VB = pca_axes(A), pca_axes(B)
    # 24 proper rotations
    mats = []
    for perm in itertools.permutations(range(3)):
        P = np.eye(3)[:, perm]
        for signs in itertools.product([-1, 1], repeat=3):
            S = np.diag(signs)
            R = P @ S
            if np.linalg.det(R) > 0: mats.append(R)
    
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(A), size=min(sample, len(A)), replace=False)
    As, kd = A[idx], cKDTree(B)
    best_rms, best_R = np.inf, None
    for M in mats:
        R = VB @ M @ VA.T
        X = (R @ As.T).T + (B.mean(0) - R @ A.mean(0))
        d, _ = kd.query(X, k=1)
        rms = np.sqrt((d**2).mean())
        if rms < best_rms:
            best_rms, best_R = rms, R
    return best_R, B.mean(0) - best_R @ A.mean(0)

def icp_point_to_point(src, ref, iters=50, tol=1e-5):
    kd = cKDTree(ref)
    R, t, prev = np.eye(3), np.zeros(3), np.inf
    for _ in range(iters):
        X = (R @ src.T).T + t
        d, idx = kd.query(X, k=1)
        Y = ref[idx]
        muX, muY = X.mean(0), Y.mean(0)
        U, _, Vt = np.linalg.svd((X - muX).T @ (Y - muY))
        R_ = Vt.T @ U.T
        if np.linalg.det(R_) < 0: Vt[-1, :] *= -1; R_ = Vt.T @ U.T
        t_ = muY - R_ @ muX
        R, t, err = R_ @ R, R_ @ t + t_, float(np.mean(d**2))
        if abs(prev - err) < tol: break
        prev = err
    return (R @ src.T).T + t, R, t

# ==========================================
# MAIN ANALYSIS LOGIC
# ==========================================

def analyze_shapes(src_path, ref_path, output_dir, model_id):
    # 1. Load and Preprocess
    V_src, F_src, _ = load_mesh(src_path)
    V_ref, F_ref, _ = load_mesh(ref_path)
    V_src, F_src = deduplicate_mesh(V_src, F_src)
    V_ref, _ = deduplicate_mesh(V_ref, np.array([]))

    # 2. Registration (PCA then ICP)
    V_s1, s, cV, cR = scale_center(V_src, V_ref)
    R0, t0 = best_pca_rotation(V_s1, V_ref)
    V_pca = (R0 @ V_s1.T).T + t0
    V_aligned, _, _ = icp_point_to_point(V_pca, V_ref)

    # 3. Deviation Calculation
    kd = cKDTree(V_ref)
    dist, _ = kd.query(V_aligned, k=1)
    
    stats = {
        "model_id": model_id,
        "mean_dev": np.mean(dist),
        "std_dev": np.std(dist),
        "max_dev": np.max(dist),
        "rmse": np.sqrt(np.mean(dist**2))
    }

    # 4. Save Plotly Interactive HTML
    fig = go.Figure(data=[go.Mesh3d(
        x=V_aligned[:,0], y=V_aligned[:,1], z=V_aligned[:,2],
        intensity=dist, colorscale='Viridis', showscale=True
    )])
    fig.update_layout(title=f"Deviation: {model_id}")
    fig.write_html(os.path.join(output_dir, f"{model_id}_deviation.html"))

    # 5. Save Histogram
    plt.figure()
    plt.hist(dist, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Deviation Distribution - {model_id}")
    plt.savefig(os.path.join(output_dir, f"{model_id}_histogram.png"))
    plt.close()

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV with model pairs")
    parser.add_argument("--out", default="output_results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    all_results = []

    for idx, row in df.iterrows():
        mid = row.get('id', f"model_{idx}")
        print(f"Processing {mid}...")
        try:
            res = analyze_shapes(row['model_path'], row['reference_path'], args.out, mid)
            all_results.append(res)
        except Exception as e:
            print(f"Failed {mid}: {e}")

    pd.DataFrame(all_results).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    main()