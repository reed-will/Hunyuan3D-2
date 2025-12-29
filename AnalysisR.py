import trimesh
import open3d as o3d
import numpy as np
import json
import os
import csv
import argparse
import traceback

class ModelAnalyzer:
    def __init__(self, target_path, test_path, voxel_size=None, sample_threshold=25000, debug=False):
        self.target_path = os.path.expanduser(target_path)
        self.test_path = os.path.expanduser(test_path)
        self.test_name = os.path.basename(self.test_path)
        self.sample_threshold = sample_threshold
        self.debug = debug
        
        self.target_mesh = self._load_mesh(self.target_path)
        self.test_mesh = self._load_mesh(self.test_path)
        
        # Initial voxel size for registration
        extents = self.target_mesh.bounding_box.extents
        diag = np.sqrt(np.sum(extents**2))
        self.voxel_size = voxel_size if voxel_size else diag / 80
            
        self.results = {}
        self.sampled_target_pcd = None # Store for debug saving

    def _load_mesh(self, path):
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        
        # Break read-only memory maps
        vertices = np.array(mesh.vertices, copy=True, dtype=np.float64)
        faces = np.array(mesh.faces, copy=True, dtype=np.int64)
        
        new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        new_mesh.fix_normals()
        return new_mesh

    def _ensure_dense_pcd(self, mesh, label):
        v_count = len(mesh.vertices)
        is_sampled = False
        
        if v_count < self.sample_threshold:
            print(f"[{label}] Sparse ({v_count} vertices). Sampling {count} points from faces...")
            points, face_indices = trimesh.sample.sample_surface(mesh, count)
            normals = mesh.face_normals[face_indices]
            is_sampled = True
        else:
            print(f"[{label}] Dense ({v_count} vertices).")
            points = mesh.vertices
            normals = mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else mesh.face_normals[0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, copy=True))
        pcd.normals = o3d.utility.Vector3dVector(np.array(normals, copy=True))
        
        return pcd, is_sampled

    def align_and_measure(self, output_dir):
        # STEP 1: PRE-ALIGNMENT
        t_center = self.target_mesh.centroid.copy()
        self.target_mesh.vertices -= t_center
        self.test_mesh.vertices -= self.test_mesh.centroid.copy()

        t_ext = self.target_mesh.bounding_box.extents
        s_ext = self.test_mesh.bounding_box.extents
        scale_factor = np.mean(t_ext / s_ext)
        self.test_mesh.apply_scale(scale_factor)

        # STEP 2: DENSE SAMPLING
        target_pcd, target_was_sampled = self._ensure_dense_pcd(self.target_mesh, "Target")
        test_pcd, _ = self._ensure_dense_pcd(self.test_mesh, "Test", count=150000)

        # Debug Save: Save the sampled target if requested
        if self.debug and target_was_sampled:
            debug_path = os.path.join(output_dir, f"DEBUG_sampled_target_{self.test_name}.ply")
            o3d.io.write_point_cloud(debug_path, target_pcd)
            print(f"[DEBUG] Saved sampled target cloud to: {debug_path}")

        # STEP 3: GLOBAL REGISTRATION (RANSAC)
        t_down, t_fpfh = self._get_features(target_pcd, self.voxel_size)
        s_down, s_fpfh = self._get_features(test_pcd, self.voxel_size)

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            s_down, t_down, s_fpfh, t_fpfh, True, self.voxel_size * 2,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 2)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        # STEP 4: MULTI-SCALE ICP
        current_transformation = result_ransac.transformation
        for scale in [3, 1, 0.5]:
            iter_voxel_size = self.voxel_size * scale
            result_icp = o3d.pipelines.registration.registration_icp(
                test_pcd, target_pcd, iter_voxel_size, current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            current_transformation = result_icp.transformation

        test_pcd.transform(current_transformation)

        # STEP 5: METRICS
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

    def _get_features(self, pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
        return pcd_down, fpfh

    def save_output(self, distances, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        v_max = np.percentile(distances, 98)
        norm_dist = np.clip(distances / (v_max if v_max > 0 else 1), 0, 1)
        
        # Manual Jet Colormap
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
    parser = argparse.ArgumentParser(description="Professional Batch 3D Analysis")
    parser.add_argument("csv_path", help="Path to input CSV")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Save sampled target point clouds for inspection")
    args = parser.parse_args()

    csv_path = os.path.expanduser(args.csv_path)
    if not os.path.exists(csv_path):
        print(f"Error: CSV {csv_path} not found.")
        return

    all_results = []
    os.makedirs(args.out, exist_ok=True)

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            target_p, test_p = row[0].strip(), row[1].strip()
            print(f"\n--- Processing: {os.path.basename(test_p)} ---")
            try:
                analyzer = ModelAnalyzer(target_p, test_p, debug=args.debug)
                dists = analyzer.align_and_measure(args.out)
                analyzer.save_output(dists, args.out)
                all_results.append(analyzer.results)
                print(f"Success. Chamfer: {analyzer.results['chamfer_dist']:.6f}")
            except Exception:
                print(f"FAILED: {test_p}")
                traceback.print_exc()

    if all_results:
        summary_path = os.path.join(args.out, "batch_summary.csv")
        with open(summary_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
        print(f"\nDone. Summary: {summary_path}")

if __name__ == "__main__":
    main()