import trimesh
import open3d as o3d
import numpy as np
import json
import os
import csv
import argparse
import traceback

class ProfessionalAnalyzer:
    def __init__(self, target_path, test_path, voxel_size=None, sample_threshold=10000):
        self.target_path = os.path.expanduser(target_path)
        self.test_path = os.path.expanduser(test_path)
        self.test_name = os.path.basename(self.test_path)
        self.sample_threshold = sample_threshold
        
        # Load and deeply copy to avoid read-only memory map errors
        self.target_mesh = self._load_mesh(self.target_path)
        self.test_mesh = self._load_mesh(self.test_path)
        
        # Robust Voxel Calculation
        extents = self.target_mesh.bounding_box.extents
        self.voxel_size = voxel_size if voxel_size else np.max(extents) / 60
            
    def _load_mesh(self, path):
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            if hasattr(mesh, 'concatenate'):
                mesh = mesh.concatenate()
            else:
                mesh = trimesh.util.concatenate(mesh.dump())

        # Force writeable arrays
        v = np.array(mesh.vertices, copy=True, dtype=np.float64)
        f = np.array(mesh.faces, copy=True, dtype=np.int64)
        new_mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
        new_mesh.fix_normals()
        return new_mesh

    def _pca_alignment(self, pcd):
        """Calculates the center and principal axes of a point cloud."""
        pts = np.asarray(pcd.points)
        center = np.mean(pts, axis=0)
        pts_centered = pts - center
        # Covariance matrix for PCA
        cov = np.dot(pts_centered.T, pts_centered) / pts.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # Sort eigenvectors by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        return center, eigenvectors[:, idx]

    def _get_dense_pcd(self, mesh, count=250000):
        #if len(mesh.vertices) < self.sample_threshold:
        points, face_indices = trimesh.sample.sample_surface(mesh, count)
        normals = mesh.face_normals[face_indices]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, copy=True))
        pcd.normals = o3d.utility.Vector3dVector(np.array(normals, copy=True))
        return pcd

    def align_and_measure(self):
        # 1. PRE-PROCESSING: DENSE SAMPLING
        target_pcd = self._get_dense_pcd(self.target_mesh)
        test_pcd = self._get_dense_pcd(self.test_mesh)

        # 2. PCA SCALING & CENTERING
        # We use the standard deviation of distances to the center for robust scaling
        t_pts = np.asarray(target_pcd.points)
        s_pts = np.asarray(test_pcd.points)
        
        t_center, t_axes = self._pca_alignment(target_pcd)
        s_center, s_axes = self._pca_alignment(test_pcd)
        
        # Calculate robust scale factor
        t_scale = np.std(np.linalg.norm(t_pts - t_center, axis=1))
        s_scale = np.std(np.linalg.norm(s_pts - s_center, axis=1))
        scale_factor = t_scale / s_scale
        
        # Apply initial transform (Center and Scale)
        test_pcd.scale(scale_factor, center=s_center)
        test_pcd.translate(t_center - s_center)

        # 3. GLOBAL REGISTRATION (FAST GLOBAL REGISTRATION)
        # Extract features
        t_down = target_pcd.voxel_down_sample(self.voxel_size)
        s_down = test_pcd.voxel_down_sample(self.voxel_size)
        t_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=30))
        s_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=30))
        
        t_fpfh = o3d.pipelines.registration.compute_fpfh_feature(t_down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*5, max_nn=100))
        s_fpfh = o3d.pipelines.registration.compute_fpfh_feature(s_down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*5, max_nn=100))

        # Fast Global Registration is better than RANSAC for mechanical parts
        result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            s_down, t_down, s_fpfh, t_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=self.voxel_size * 1.5,
                iteration_number=64)
        )

        # 4. MULTI-SCALE REFINEMENT (POINT-TO-PLANE ICP)
        current_trans = result_fgr.transformation
        target_pcd.estimate_normals() # Ensure target has normals for Point-to-Plane
        
        for scale in [2.0, 1.0, 0.5]:
            reg_p2l = o3d.pipelines.registration.registration_icp(
                test_pcd, target_pcd, self.voxel_size * scale, current_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            current_trans = reg_p2l.transformation

        test_pcd.transform(current_trans)

        # 5. FINAL DISTANCE MEASUREMENTS
        d_test_to_target = np.asarray(test_pcd.compute_point_cloud_distance(target_pcd))
        d_target_to_test = np.asarray(target_pcd.compute_point_cloud_distance(test_pcd))

        self.results = {
            "test_model": self.test_name,
            "chamfer_dist": float(np.mean(d_test_to_target**2) + np.mean(d_target_to_test**2)),
            "hausdorff_dist": float(max(np.max(d_test_to_target), np.max(d_target_to_test))),
            "mean_deviation": float(np.mean(d_test_to_target)),
            "std_deviation": float(np.std(d_test_to_target))
        }
        
        self.final_pcd = test_pcd
        return d_test_to_target

    def save_output(self, distances, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        v_max = np.percentile(distances, 98)
        norm_dist = np.clip(distances / (v_max if v_max > 0 else 1), 0, 1)
        
        # Color mapping (Jet)
        c = np.zeros((len(norm_dist), 3))
        v4 = norm_dist * 4
        c[:, 0] = np.clip(np.minimum(v4 - 1.5, -v4 + 4.5), 0, 1)
        c[:, 1] = np.clip(np.minimum(v4 - 0.5, -v4 + 3.5), 0, 1)
        c[:, 2] = np.clip(np.minimum(v4 + 0.5, -v4 + 2.5), 0, 1)
        
        self.final_pcd.colors = o3d.utility.Vector3dVector(c)
        base_name = os.path.splitext(self.test_name)[0]
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{base_name}_heatmap.ply"), self.final_pcd)
        with open(os.path.join(output_dir, f"{base_name}_metrics.json"), "w") as f:
            json.dump(self.results, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    csv_path = os.path.expanduser(args.csv_path)
    all_results = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            try:
                analyzer = ProfessionalAnalyzer(row[0].strip(), row[1].strip())
                dists = analyzer.align_and_measure()
                analyzer.save_output(dists, args.out)
                all_results.append(analyzer.results)
                print(f"COMPLETED: {analyzer.test_name} | Chamfer: {analyzer.results['chamfer_dist']:.6f}")
            except Exception:
                traceback.print_exc()

    if all_results:
        summary_path = os.path.join(args.out, "batch_summary.csv")
        with open(summary_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(all_results)

if __name__ == "__main__":
    main()