import trimesh
import open3d as o3d
import numpy as np
import json
import os
import csv
import argparse
import traceback

class MetrologyAnalyzer:
    def __init__(self, target_path, test_path, voxel_size=None):
        self.target_path = os.path.expanduser(target_path)
        self.test_path = os.path.expanduser(test_path)
        self.test_name = os.path.basename(self.test_path)
        
        # Load Mesh Data
        self.target_mesh = self._load_mesh(self.target_path)
        self.test_mesh = self._load_mesh(self.test_path)
        
        # Geometry State
        extents = self.target_mesh.bounding_box.extents
        self.voxel_size = voxel_size if voxel_size else np.mean(extents) / 60
        self.t_pcd = None
        self.s_pcd = None
        self.results = {}
        self.total_applied_scale = 1.0

    def _load_mesh(self, path):
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            if hasattr(mesh, 'concatenate'):
                mesh = mesh.concatenate()
            else:
                mesh = trimesh.util.concatenate(mesh.dump())
        v = np.array(mesh.vertices, copy=True, dtype=np.float64)
        f = np.array(mesh.faces, copy=True, dtype=np.int64)
        return trimesh.Trimesh(vertices=v, faces=f, process=True)

    # --- STEP 1: DATA PREPARATION ---
    def sample_point_clouds(self, count=300000):
        """Converts meshes to high-density point clouds with normals."""
        def mesh_to_pcd(mesh):
            pts, face_idx = trimesh.sample.sample_surface(mesh, count)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.normals = o3d.utility.Vector3dVector(mesh.face_normals[face_idx])
            return pcd
        
        self.t_pcd = mesh_to_pcd(self.target_mesh)
        self.s_pcd = mesh_to_pcd(self.test_mesh)
        print("[Data] Point clouds sampled.")

    # --- STEP 2: ROUGH ALIGNMENT ---
    def apply_rough_scale_and_center(self):
        """Uses point-spread statistics for initial centering and scaling."""
        t_pts = np.asarray(self.t_pcd.points)
        s_pts = np.asarray(self.s_pcd.points)
        
        t_center = np.mean(t_pts, axis=0)
        s_center = np.mean(s_pts, axis=0)
        
        t_spread = np.mean(np.linalg.norm(t_pts - t_center, axis=1))
        s_spread = np.mean(np.linalg.norm(s_pts - s_center, axis=1))
        
        rough_scale = t_spread / s_spread
        self.s_pcd.scale(rough_scale, center=s_center)
        self.s_pcd.translate(t_center - s_center)
        
        self.total_applied_scale *= rough_scale
        print(f"[Rough] Applied initial scale: {rough_scale:.4f}")

    # --- STEP 3: GLOBAL REGISTRATION ---
    def run_global_registration(self):
        """Solves for scale, rotation, and translation via RANSAC/FPFH."""
        def get_fpfh(pcd):
            down = pcd.voxel_down_sample(self.voxel_size)
            down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=30))
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*5, max_nn=100))
            return down, fpfh

        t_down, t_fpfh = get_fpfh(self.t_pcd)
        s_down, s_fpfh = get_fpfh(self.s_pcd)

        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            s_down, t_down, s_fpfh, t_fpfh, True, self.voxel_size * 2,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 2)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        # Extract precision scale from the 3x3 component (determinant method)
        m3x3 = ransac.transformation[0:3, 0:3]
        precision_scale = np.cbrt(np.linalg.det(m3x3))
        self.total_applied_scale *= precision_scale
        
        self.s_pcd.transform(ransac.transformation)
        print(f"[Global] Applied precision scale: {precision_scale:.6f}")

    # --- STEP 4: LOCAL REFINEMENT ---
    def run_icp_refinement(self, scale_multiplier=1.0):
        """Snaps surfaces together using Point-to-Plane ICP."""
        self.t_pcd.estimate_normals()
        icp = o3d.pipelines.registration.registration_icp(
            self.s_pcd, self.t_pcd, self.voxel_size * scale_multiplier, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        self.s_pcd.transform(icp.transformation)
        print(f"[ICP] Refinement complete (multiplier: {scale_multiplier}).")


    # --- STEP 4a: RIGID REFINEMENT (Rotation & Translation Only) ---
    def run_rigid_icp_refinement(self, scale_multiplier=1.0, max_iterations=50):
        """
        Refines pose (Rotation + Translation) while keeping Scale locked.
        Useful for final polishing after Global Registration has found the scale.
        """
        if self.s_pcd is None or self.t_pcd is None:
            print("Error: Point clouds not sampled.")
            return

        self.t_pcd.estimate_normals()
        
        # We use a PointToPlane estimator which is inherently rigid (Scale = 1.0)
        # unless with_scaling is explicitly set.
        reg_p2l = o3d.pipelines.registration.registration_icp(
            self.s_pcd, self.t_pcd, self.voxel_size * scale_multiplier, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        
        self.s_pcd.transform(reg_p2l.transformation)
        print(f"[Rigid ICP] Pose refined. Scale remains locked at {self.total_applied_scale:.6f}")

    # --- STEP 4b: NON-RIGID REFINEMENT (Includes Scaling) ---
    def run_scaling_icp_refinement(self, scale_multiplier=0.5):
        """
        Specialized ICP that allows the scale to 'drift' slightly to improve fit.
        Use this only if your parts are flexible or have thermal expansion.
        """
        # Note: Open3D's standard ICP doesn't support scaling in PointToPlane natively.
        # We use the PointToPoint estimator with scaling enabled.
        reg_scaling = o3d.pipelines.registration.registration_icp(
            self.s_pcd, self.t_pcd, self.voxel_size * scale_multiplier, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        )
        
        # Extract new scale component
        m3x3 = reg_scaling.transformation[0:3, 0:3]
        incremental_scale = np.cbrt(np.linalg.det(m3x3))
        self.total_applied_scale *= incremental_scale
        
        self.s_pcd.transform(reg_scaling.transformation)
        print(f"[Scaling ICP] Scale adjusted by {incremental_scale:.6f}. Total: {self.total_applied_scale:.6f}")

    # --- STEP 5: DISTANCE COMPUTATION ---
    def compute_metrics(self):
        """Calculates final distance deviations."""
        d_s_to_t = np.asarray(self.s_pcd.compute_point_cloud_distance(self.t_pcd))
        d_t_to_s = np.asarray(self.t_pcd.compute_point_cloud_distance(self.s_pcd))

        self.results = {
            "test_model": self.test_name,
            "total_scale_factor": float(self.total_applied_scale),
            "chamfer_dist": float(np.mean(d_s_to_t**2) + np.mean(d_t_to_s**2)),
            "hausdorff_dist": float(max(np.max(d_s_to_t), np.max(d_t_to_s))),
            "mean_deviation": float(np.mean(d_s_to_t)),
            "std_deviation": float(np.std(d_s_to_t))
        }
        return d_s_to_t

    def save_output(self, distances, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        t_pts = np.asarray(self.target_mesh.vertices)
        t_spread = np.mean(np.linalg.norm(t_pts - np.mean(t_pts, axis=0), axis=1))
        
        v_max = t_spread * 0.02
        norm_dist = np.clip(distances / v_max, 0, 1)
        
        c = np.zeros((len(norm_dist), 3))
        v4 = norm_dist * 4
        c[:, 0] = np.clip(np.minimum(v4 - 1.5, -v4 + 4.5), 0, 1) # Red
        c[:, 1] = np.clip(np.minimum(v4 - 0.5, -v4 + 3.5), 0, 1) # Green
        c[:, 2] = np.clip(np.minimum(v4 + 0.5, -v4 + 2.5), 0, 1) # Blue
        
        self.s_pcd.colors = o3d.utility.Vector3dVector(c)
        base_name = os.path.splitext(self.test_name)[0]
        
        # ASCII ENCODED PLY
        o3d.io.write_point_cloud(
            os.path.join(output_dir, f"{base_name}_aligned.ply"), 
            self.s_pcd, 
            write_ascii=True
        )
        
        with open(os.path.join(output_dir, f"{base_name}_metrics.json"), "w") as f:
            json.dump(self.results, f, indent=4)




def main():
    parser = argparse.ArgumentParser(description="Professional Batch 3D Metrology Analysis")
    parser.add_argument("csv_path", help="Path to input CSV (Target_Path, Test_Path)")
    parser.add_argument("--out", default="results", help="Output directory")
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
            
            print(f"\n" + "="*50)
            print(f"PROCESSING: {os.path.basename(test_p)}")
            print("="*50)
            
            try:
                # Initialize
                analyzer = MetrologyAnalyzer(target_p, test_p)
                
                # 1. Prepare Data
                analyzer.sample_point_clouds(count=300000)
                
                # 2. Initial Rough Alignment (Centering + Spread Scaling)
                analyzer.apply_rough_scale_and_center()
                
                # 3. Global Registration (RANSAC + Umeyama Scale Solver)
                analyzer.run_global_registration()
                
                # 4. Iterative Rigid Refinement (Successive Approximation)
                # We start with a wide search buffer and narrow it down
                print("[Refinement] Starting multi-stage Rigid ICP...")
                for multiplier in [2.0, 1.0, 0.5]:
                    analyzer.run_rigid_icp_refinement(scale_multiplier=multiplier)
                
                # 5. Compute Metrics & Save Outputs
                dists = analyzer.compute_metrics()
                analyzer.save_output(dists, args.out)
                
                all_results.append(analyzer.results)
                print(f"\nSUCCESS: {analyzer.test_name}")
                print(f"Final Scale: {analyzer.results['total_scale_factor']:.6f}")
                print(f"Mean Dev:    {analyzer.results['mean_deviation']:.6f}")

            except Exception:
                print(f"\n!!! FAILED TO PROCESS: {test_p} !!!")
                traceback.print_exc()

    # Final Batch Summary
    if all_results:
        summary_path = os.path.join(args.out, "batch_summary.csv")
        with open(summary_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
        print(f"\n" + "="*50)
        print(f"BATCH COMPLETE. Summary saved to: {summary_path}")
        print("="*50)

if __name__ == "__main__":
    main()








