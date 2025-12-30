import trimesh
import open3d as o3d
import numpy as np
import json
import os
import csv
import argparse
import traceback

class ShapeAnalyzer:
    def __init__(self, target_path, test_path, voxel_size=None):
        self.target_path = os.path.expanduser(target_path)
        self.test_path = os.path.expanduser(test_path)
        self.test_name = os.path.basename(self.test_path)
        
        self.target_mesh = self._load_mesh(self.target_path)
        self.test_mesh = self._load_mesh(self.test_path)
        
        # Use a robust voxel size (1/60th of the target's average span)
        extents = self.target_mesh.bounding_box.extents
        self.voxel_size = voxel_size if voxel_size else np.mean(extents) / 60
        self.results = {}

    def _load_mesh(self, path):
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        v = np.array(mesh.vertices, copy=True, dtype=np.float64)
        f = np.array(mesh.faces, copy=True, dtype=np.int64)
        m = trimesh.Trimesh(vertices=v, faces=f, process=True)
        return m

    def _get_pcd(self, mesh):
        # High-density sampling for better correspondence
        pts, face_idx = trimesh.sample.sample_surface(mesh, 300000)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(mesh.face_normals[face_idx])
        return pcd

    def align_and_measure(self):
        # 1. ROBUST INITIAL SCALING (Point-Spread Ratio)
        # This is much better than Bounding Boxes because it uses all points.
        t_pcd = self._get_pcd(self.target_mesh)
        s_pcd = self._get_pcd(self.test_mesh)
        
        t_pts = np.asarray(t_pcd.points)
        s_pts = np.asarray(s_pcd.points)
        
        t_center = np.mean(t_pts, axis=0)
        s_center = np.mean(s_pts, axis=0)
        
        # Calculate the "Mean Radius" of each cloud
        t_spread = np.mean(np.linalg.norm(t_pts - t_center, axis=1))
        s_spread = np.mean(np.linalg.norm(s_pts - s_center, axis=1))
        
        # Apply initial rough scale and translation to get them in the same workspace
        rough_scale = t_spread / s_spread
        s_pcd.scale(rough_scale, center=s_center)
        s_pcd.translate(t_center - s_center)

        # 2. GLOBAL REGISTRATION (Umeyama/RANSAC)
        # Solve for the precise Scale, Rotation, and Translation
        t_down, t_fpfh = self._get_features(t_pcd)
        s_down, s_fpfh = self._get_features(s_pcd)

        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            s_down, t_down, s_fpfh, t_fpfh, True, self.voxel_size * 2,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 2)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        # 3. ROBUST SCALE EXTRACTION (SVD/Determinant)
        # Extract the uniform scale factor 's' from the 3x3 component of the 4x4 matrix
        # T = [sR t; 0 1]. det(sR) = s^3 * det(R). Since det(R)=1, s = cbrt(det(sR)).
        m3x3 = ransac.transformation[0:3, 0:3]
        precision_scale = np.cbrt(np.linalg.det(m3x3))
        total_scale = rough_scale * precision_scale
        
        # Apply final transform
        s_pcd.transform(ransac.transformation)

        # 4. ICP REFINEMENT (Local snap)
        icp = o3d.pipelines.registration.registration_icp(
            s_pcd, t_pcd, self.voxel_size, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        s_pcd.transform(icp.transformation)

        # 5. METRICS
        d_s_to_t = np.asarray(s_pcd.compute_point_cloud_distance(t_pcd))
        d_t_to_s = np.asarray(t_pcd.compute_point_cloud_distance(s_pcd))

        self.results = {
            "test_model": self.test_name,
            "calculated_scale": float(total_scale),
            "chamfer_dist": float(np.mean(d_s_to_t**2) + np.mean(d_t_to_s**2)),
            "hausdorff_dist": float(max(np.max(d_s_to_t), np.max(d_t_to_s))),
            "mean_deviation": float(np.mean(d_s_to_t))
        }
        self.final_pcd = s_pcd
        return d_s_to_t

    def _get_features(self, pcd):
        down = pcd.voxel_down_sample(self.voxel_size)
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*5, max_nn=100))
        return down, fpfh

    def save_output(self, distances, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Better Heatmap Scale: Normalize by the Target's "Mean Radius" (spread)
        # This makes the color relative to the physical bulk of the part.
        t_pts = np.asarray(self.target_mesh.vertices)
        t_spread = np.mean(np.linalg.norm(t_pts - np.mean(t_pts, axis=0), axis=1))
        
        # Set Red to be 2% of the model's average radius
        v_max = t_spread * 0.02
        norm_dist = np.clip(distances / v_max, 0, 1)
        
        # Jet Colormap Logic
        c = np.zeros((len(norm_dist), 3))
        v4 = norm_dist * 4
        c[:, 0] = np.clip(np.minimum(v4 - 1.5, -v4 + 4.5), 0, 1) # R
        c[:, 1] = np.clip(np.minimum(v4 - 0.5, -v4 + 3.5), 0, 1) # G
        c[:, 2] = np.clip(np.minimum(v4 + 0.5, -v4 + 2.5), 0, 1) # B
        
        self.final_pcd.colors = o3d.utility.Vector3dVector(c)
        base_name = os.path.splitext(self.test_name)[0]
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{base_name}_aligned.ply"), self.final_pcd, write_ascii=True)
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
                analyzer = ShapeAnalyzer(row[0].strip(), row[1].strip())
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