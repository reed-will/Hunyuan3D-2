import trimesh
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import csv
import argparse

class ModelAnalyzer:
    def __init__(self, target_path, test_path, voxel_size=None, sample_threshold=10000):
        # Resolve ~/ paths
        self.target_path = os.path.expanduser(target_path)
        self.test_path = os.path.expanduser(test_path)
        self.test_name = os.path.basename(self.test_path)
        
        # FIX: Define sample_threshold so it can be used in _ensure_dense_pcd
        self.sample_threshold = sample_threshold
        
        self.target_mesh = self._load_mesh(self.target_path)
        self.test_mesh = self._load_mesh(self.test_path)
        
        if voxel_size is None:
            diag = np.linalg.norm(self.target_mesh.bounding_box.extents)
            self.voxel_size = diag / 100
        else:
            self.voxel_size = voxel_size
            
        self.results = {}
        self.test_pcd = None
        self.target_o3d_mesh = None 

    def _load_mesh(self, path):
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        mesh.vertices = np.array(mesh.vertices, copy=True)
        
        mesh.fix_normals()
        if hasattr(mesh, 'vertex_normals'):
            mesh.vertex_normals = np.array(mesh.vertex_normals, copy=True)
        return mesh

    def _ensure_dense_pcd(self, mesh, label, count=250000):
        v_count = len(mesh.vertices)
        
        if v_count < self.sample_threshold:
            print(f"[{label}] Sparse geometry ({v_count} vertices). Sampling {count} points...")
            # Sample points and the indices of the faces they came from
            points, face_indices = trimesh.sample.sample_surface(mesh, count)
            normals = mesh.face_normals[face_indices]
        else:
            print(f"[{label}] Dense geometry ({v_count} vertices).")
            points = mesh.vertices
            # Fallback if vertex_normals are missing
            normals = mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else mesh.face_normals[0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        return pcd

    def align_and_measure(self):
        # 1. Scaling
        target_extents = self.target_mesh.bounding_box.extents
        test_extents = self.test_mesh.bounding_box.extents
        scale_factor = np.mean(target_extents / test_extents)
        self.test_mesh.apply_scale(scale_factor)

        # 2. Centering
        target_center = self.target_mesh.centroid
        self.target_mesh.vertices -= target_center
        self.test_mesh.vertices -= target_center

        # Sampling
        target_pcd = self._ensure_dense_pcd(self.target_mesh, "Target", count=250000)
        self.test_pcd = self._ensure_dense_pcd(self.test_mesh, "Test", count=150000)

        # Build Target Mesh for Wireframe (using centered coords)
        # self.target_o3d_mesh = o3d.geometry.TriangleMesh()
        # self.target_o3d_mesh.vertices = o3d.utility.Vector3dVector(self.target_mesh.vertices)
        # self.target_o3d_mesh.triangles = o3d.utility.Vector3iVector(self.target_mesh.faces)

        # 4. Alignment
        t_down, t_fpfh = self._get_features(target_pcd)
        s_down, s_fpfh = self._get_features(self.test_pcd)
        
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            s_down, t_down, s_fpfh, t_fpfh, True, self.voxel_size * 2,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 2)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        icp = o3d.pipelines.registration.registration_icp(
            self.test_pcd, target_pcd, self.voxel_size, ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        self.test_pcd.transform(icp.transformation)

        # 5. Metrics
        d_test_to_target = np.asarray(self.test_pcd.compute_point_cloud_distance(target_pcd))
        d_target_to_test = np.asarray(target_pcd.compute_point_cloud_distance(self.test_pcd))

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

    def visualize_with_overlay(self, distances):
        v_max = np.percentile(distances, 98)
        v_min = np.min(distances)
        norm_dist = np.clip((distances - v_min) / (v_max - v_min), 0, 1)
        colors = plt.get_cmap("jet")(norm_dist)[:, :3]
        self.test_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.target_o3d_mesh)
        wireframe.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([self.test_pcd, wireframe], window_name=f"Overlay: {self.test_name}")

    def save_output(self, distances, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Consistent with visualization (98th percentile)
        v_max = np.percentile(distances, 98) 
        cmap = plt.get_cmap("jet")
        colors = cmap(np.clip(distances / v_max, 0, 1))[:, :3]
        self.test_pcd.colors = o3d.utility.Vector3dVector(colors)

        base_name = os.path.splitext(self.test_name)[0]
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{base_name}_heatmap.ply"), self.test_pcd)
        with open(os.path.join(output_dir, f"{base_name}_metrics.json"), "w") as f:
            json.dump(self.results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Batch 3D model analysis via CSV.")
    parser.add_argument("csv_path", help="Path to CSV")
    parser.add_argument("--out", default="results", help="Output dir")
    args = parser.parse_args()
    csv_path = os.path.expanduser(args.csv_path)

    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return

    all_results = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            target_p, test_p = row[0].strip(), row[1].strip()
            try:
                print(f"Row {test_p}")
                analyzer = ModelAnalyzer(target_p, test_p)
                dists = analyzer.align_and_measure()
                #analyzer.visualize_with_overlay(dists)
                analyzer.save_output(dists, args.out)
                all_results.append(analyzer.results)
            except Exception as e:
                print(f"Failed {test_p}: {e}")

    summary_path = os.path.join(args.out, "batch_summary.csv")
    if all_results:
        keys = all_results[0].keys()
        with open(summary_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)

if __name__ == "__main__":
    main()