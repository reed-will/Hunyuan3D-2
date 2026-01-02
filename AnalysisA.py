import os
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import trimesh
from scipy.spatial import cKDTree

# ==========================================
# GEOMETRY UTILITIES (from Notebook)
# ==========================================

def load_mesh(path):
    """Loads STL or GLB files and returns vertices/faces."""
    full_path = os.path.expanduser(path.strip())
    mesh_data = trimesh.load(full_path)
    
    if isinstance(mesh_data, trimesh.Scene):
        # mesh_obj = mesh_data.dump(concatenate=True)
        mesh_obj = mesh_data.to_geometry() # other version deprecated
    else:
        mesh_obj = mesh_data
        
    V = np.array(mesh_obj.vertices, dtype=np.float64)
    F = np.array(mesh_obj.faces)
    return V, F

def deduplicate_mesh(V, F, tol=1e-6):
    """Removes duplicate vertices and degenerate faces."""
    key = np.round(V / tol).astype(np.int64)
    _, keep_idx, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Vd = V[keep_idx]
    Fd = inverse[F]
    mask = np.all([Fd[:, 0] != Fd[:, 1], Fd[:, 1] != Fd[:, 2], Fd[:, 2] != Fd[:, 0]], axis=0)
    return Vd, Fd[mask]

def scale_center(V, Vref):
    """Scale by bbox diagonal and center on reference centroid."""
    vmin, vmax = V.min(0), V.max(0)
    rmin, rmax = Vref.min(0), Vref.max(0)
    s = np.linalg.norm(rmax - rmin) / np.linalg.norm(vmax - vmin)
    cV, cR = (vmin + vmax)/2, (rmin + rmax)/2
    V1 = (V - cV) * s + cR
    return V1, s, cV, cR

def pca_axes(P):
    """Extracts principal axes using Eigendecomposition."""
    P0 = P - P.mean(0)
    _, V = np.linalg.eigh(np.cov(P0.T))
    if np.linalg.det(V) < 0: V[:, 0] *= -1
    return V

def best_pca_rotation(A, B, sample=30000, seed=0):
    """Notebook Logic: Picks the axis mapping that minimizes subsampled NN RMS."""
    VA, VB = pca_axes(A), pca_axes(B)
    mats = []
    # Generate 24 proper rotations (proper axis maps)
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
    t = B.mean(0) - best_R @ A.mean(0)
    return best_R, t

# ==========================================
# ICP LOGIC OPTIONS
# ==========================================

def icp_point_to_point_trimmed(src, ref, iters=75, tol=1e-5, trims=(95, 97, 98)):
    """RESTORED FROM NOTEBOOK: Coarse-to-fine ICP with percentile trimming."""
    kd = cKDTree(ref)
    R, t = np.eye(3), np.zeros(3)
    prev = np.inf
    for trim in trims:
        steps = max(1, iters // len(trims))
        for _ in range(steps):
            X = (R @ src.T).T + t
            d, idx = kd.query(X, k=1)
            Y = ref[idx]
            thr = np.percentile(d, trim) # Robustness: Ignore outliers above threshold
            keep = d <= thr
            Xk, Yk = X[keep], Y[keep]
            muX, muY = Xk.mean(0), Yk.mean(0)
            U, _, Vt = np.linalg.svd((Xk - muX).T @ (Yk - muY))
            R_ = Vt.T @ U.T
            if np.linalg.det(R_) < 0: Vt[-1, :] *= -1; R_ = Vt.T @ U.T
            t_ = muY - R_ @ muX
            R, t, err = R_ @ R, R_ @ t + t_, float(np.mean((Xk - Yk)**2))
            if abs(prev - err) < tol * max(1.0, prev): break
            prev = err
    return (R @ src.T).T + t, R, t

def icp_simple(src, ref, iters=50, tol=1e-5):
    """Script Version: Standard Point-to-Point ICP without trimming."""
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
# MAIN EXECUTION
# ==========================================

def analyze_shapes(src_path, ref_path, output_dir, model_id, use_trimmed=True):
    V_src, F_src = load_mesh(src_path)
    V_ref, F_ref = load_mesh(ref_path)
    V_src, F_src = deduplicate_mesh(V_src, F_src)
    V_ref, _ = deduplicate_mesh(V_ref, np.array([[0,0,0]], dtype=int))

    # Initial Alignment
    V_s1, s, cV, cR = scale_center(V_src, V_ref)
    R0, t0 = best_pca_rotation(V_s1, V_ref)
    V_pca = (R0 @ V_s1.T).T + t0
    
    # Registration Choice
    if use_trimmed:
        V_aligned, _, _ = icp_point_to_point_trimmed(V_pca, V_ref)
    else:
        V_aligned, _, _ = icp_simple(V_pca, V_ref)

    # Deviation Calculation
    kd = cKDTree(V_ref)
    dist, _ = kd.query(V_aligned, k=1)
    
    # Stats & Visualization (Saving logic remains same)
    stats = {"id": model_id, "mean": np.mean(dist), "rmse": np.sqrt(np.mean(dist**2)), "max": np.max(dist)}
    
    fig = go.Figure(data=[go.Mesh3d(x=V_aligned[:,0], y=V_aligned[:,1], z=V_aligned[:,2],
                    intensity=dist, colorscale='Viridis', i=F_src[:,0], j=F_src[:,1], k=F_src[:,2])])
    fig.write_html(os.path.join(output_dir, f"{model_id}.html"))
    return stats

def main():
    # --- CONFIGURATION FLAG ---
    USE_TRIMMED_ICP = True # Set to False to use the simple ICP version
    # --------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    
    results = []
    for idx, row in df.iterrows():
        m_path, r_path = row['model_path'], row['reference_path']
        name = os.path.basename(m_path).split('.')[0]
        try:
            res = analyze_shapes(m_path, r_path, args.out, name, use_trimmed=USE_TRIMMED_ICP)
            results.append(res)
        except Exception as e:
            print(f"Error on {name}: {e}")

    pd.DataFrame(results).to_csv(os.path.join(args.out, "summary.csv"), index=False)

if __name__ == "__main__":
    main()