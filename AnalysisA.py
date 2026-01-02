import os
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import trimesh # Switched to trimesh for .glb and .stl support
from scipy.spatial import cKDTree

# ==========================================
# GEOMETRY UTILITIES
# ==========================================

def load_mesh(path):
    """Loads STL or GLB files and returns vertices/faces."""
    # Expand '~' to full home directory path
    full_path = os.path.expanduser(path)
    
    mesh_data = trimesh.load(full_path)
    
    # GLB files often load as 'Scenes'. We merge them into a single mesh.
    if isinstance(mesh_data, trimesh.Scene):
        # mesh_obj = mesh_data.dump(concatenate=True)
        mesh_obj = mesh_data.to_geometry() # other version deprecated
    else:
        mesh_obj = mesh_data
        
    V = np.array(mesh_obj.vertices, dtype=np.float64)
    F = np.array(mesh_obj.faces)
    return V, F

def deduplicate_mesh(V, F, tol=1e-6):
    key = np.round(V / tol).astype(np.int64)
    _, keep_idx, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Vd = V[keep_idx]
    Fd = inverse[F]
    # Remove degenerate faces
    mask = np.all([Fd[:, 0] != Fd[:, 1], Fd[:, 1] != Fd[:, 2], Fd[:, 2] != Fd[:, 0]], axis=0)
    return Vd, Fd[mask]

# (Include scale_center, pca_axes, best_pca_rotation, icp_point_to_point from previous script)
# ... [Keeping the registration logic from the notebook] ...

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
# MAIN EXECUTION
# ==========================================

def analyze_shapes(src_path, ref_path, output_dir, model_id):
    # 1. Load and Preprocess (with format handling)
    V_src, F_src = load_mesh(src_path)
    V_ref, F_ref = load_mesh(ref_path)
    V_src, F_src = deduplicate_mesh(V_src, F_src)
    V_ref, _ = deduplicate_mesh(V_ref, np.array([[0,0,0]], dtype=int))

    # 2. Registration Logic
    V_s1, s, cV, cR = scale_center(V_src, V_ref)
    R0, t0 = best_pca_rotation(V_s1, V_ref)
    V_pca = (R0 @ V_s1.T).T + t0
    V_aligned, _, _ = icp_point_to_point(V_pca, V_ref)

    # 3. Deviation Stats
    kd = cKDTree(V_ref)
    dist, _ = kd.query(V_aligned, k=1)
    
    stats = {
        "model_id": model_id,
        "mean_deviation": np.mean(dist),
        "rmse": np.sqrt(np.mean(dist**2)),
        "max_deviation": np.max(dist)
    }

    # 4. Save Visuals
    fig = go.Figure(data=[go.Mesh3d(
        x=V_aligned[:,0], y=V_aligned[:,1], z=V_aligned[:,2],
        intensity=dist, colorscale='Viridis', showscale=True,
        i=F_src[:,0], j=F_src[:,1], k=F_src[:,2]
    )])
    fig.write_html(os.path.join(output_dir, f"{model_id}_analysis.html"))
    
    plt.figure()
    plt.hist(dist, bins=50)
    plt.title(f"Deviations: {model_id}")
    plt.savefig(os.path.join(output_dir, f"{model_id}_hist.png"))
    plt.close()

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    
    # Ensure CSV is cleaned of extra spaces in headers
    df.columns = [c.strip() for c in df.columns]
    
    results = []
    for idx, row in df.iterrows():
        # Using your exact CSV column names
        m_path = row['model_path'].strip()
        r_path = row['reference_path'].strip()
        
        # Create a name for the folder based on the file name
        name = os.path.basename(m_path).split('.')[0]
        print(f"[{idx+1}/{len(df)}] Analyzing {name}...")
        
        try:
            res = analyze_shapes(m_path, r_path, args.out, name)
            results.append(res)
        except Exception as e:
            print(f"Error on {name}: {e}")

    pd.DataFrame(results).to_csv(os.path.join(args.out, "all_stats.csv"), index=False)

if __name__ == "__main__":
    main()

