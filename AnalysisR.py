import trimesh
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import csv
import argparse

class ModelAnalyzer:
    """Core logic to compare a 'Test' model against a 'Target' (ground truth)"""

    def __init__(self, target_path, test_path, voxel_size=None):
        self.test_name = os.path.basename(test_path)
        self.target_mesh = self._load_mesh(target_path)
        self.test_mesh = self._load_mesh(test_path)
        
        # RNOTE: Should we do this here, or after alignment?
        # Scale-dependent voxel size (approx 1% of model diagonal)
        if voxel_size is None:
            self.voxel_size = self.target_mesh.bounding_box.diagonal / 100
        else:
            self.voxel_size = voxel_size
            
        self.results = {}
        self.test_pcd = None
        self.target_pcd = None

    def _load_mesh(self, path):
        """Loads STL/OBJ/PLY/STEP and handles multi-part scenes."""
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return mesh

    def align_and_measure(self):
        # 1. Scaling Test to match Target
        target_extents = self.target_mesh.bounding_box.extents
        test_extents = self.test_mesh.bounding_box.extents
        scale_factor = np.mean(target_extents / test_extents)
        self.test_mesh.apply_scale(scale_factor)

        # 2. Centering (Normalization)
        target_center = self.target_mesh.centroid
        self.target_mesh.vertices -= target_center
        self.test_mesh.vertices -= target_center

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(self.target_mesh.vertices)
        test_pcd = o3d.geometry.PointCloud()
        test_pcd.points = o3d.utility.Vector3dVector(self.test_mesh.vertices)

        # 3. Global Registration (RANSAC)
        t_down, t_fpfh = self._get_features(target_pcd)
        s_down, s_fpfh = self._get_features(test_pcd)

        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            s_down, t_down, s_fpfh, t_fpfh, True, self.voxel_size * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        # 4. Local ICP Refinement
        target_pcd.estimate_normals()
        icp = o3d.pipelines.registration.registration_icp(
            test_pcd, target_pcd, self.voxel_size, ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        self.test_pcd = test_pcd.transform(icp.transformation)
        self.target_pcd = target_pcd

        # 5. Metric Computation
        d_test_to_target = np.asarray(self.test_pcd.compute_point_cloud_distance(self.target_pcd))
        d_target_to_test = np.asarray(self.target_pcd.compute_point_cloud_distance(self.test_pcd))

        self.results = {
            "test_model": self.test_name,
            "chamfer_dist": float(np.mean(d_test_to_target**2) + np.mean(d_target_to_test**2)),
            "hausdorff_dist": float(max(np.max(d_test_to_target), np.max(d_target_to_test))),
            "mean_deviation": float(np.mean(d_test_to_target)),
            "std_deviation": float(np.std(d_test_to_target))
        }
        return d_test_to_target

    def _get_features(self, pcd):
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*5, max_nn=100))
        return pcd_down, fpfh

    def save_output(self, distances, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        max_dist = np.percentile(distances, 95)
        cmap = plt.get_cmap("jet")
        colors = cmap(np.clip(distances / max_dist, 0, 1))[:, :3]
        self.test_pcd.colors = o3d.utility.Vector3dVector(colors)

        base_name = os.path.splitext(self.test_name)[0]
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{base_name}_heatmap.ply"), self.test_pcd)
        with open(os.path.join(output_dir, f"{base_name}_metrics.json"), "w") as f:
            json.dump(self.results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Batch 3D model analysis via CSV.")
    parser.add_argument("csv_path", help="Path to CSV containing: target_path, test_path")
    parser.add_argument("--out", default="results", help="Output directory name")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file '{args.csv_path}' not found.")
        return

    all_results = []
    
    with open(args.csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            target_path, test_path = row[0].strip(), row[1].strip()
            
            print(f"\nComparing:\nTarget: {target_path}\nTest:   {test_path}")
            
            try:
                analyzer = ModelAnalyzer(target_path, test_path)
                distances = analyzer.align_and_measure()
                analyzer.save_output(distances, args.out)
                all_results.append(analyzer.results)
                print(f"Success. Chamfer: {analyzer.results['chamfer_dist']:.6f}")
            except Exception as e:
                print(f"Failed to process pair: {e}")

    # Final summary CSV for all pairs processed
    summary_path = os.path.join(args.out, "batch_summary.csv")
    if all_results:
        keys = all_results[0].keys()
        with open(summary_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
        print(f"\nDone. Master summary saved to {summary_path}")

if __name__ == "__main__":
    main()