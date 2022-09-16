import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from argoverse.utils.cuboid_interior import (filter_point_cloud_to_bbox_3D_vectorized)
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
from torch.utils.data import Dataset
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes


@dataclass
class LidarData:
    ply: np.ndarray
    pose: SE3
    timestamp: int
    scene_token: str
    boxes: np.ndarray
    objects: np.ndarray
    annotation_token: list


class PreProcessNuScenesDataset(Dataset):
    def __init__(
        self,
        save_fi_name,
        nusc, scene_tokens,
        channel = 'LIDAR_TOP',
        sample_every = 2,
        min_dist = 3.0,
        max_dist = 50.0,
        max_height = 4.0,
        margin = 0.6,
        max_correspondence_distance=1.0
    ):
        self.save_fi_name = save_fi_name
        self.scene_tokens = sorted(scene_tokens)
        self.nusc = nusc
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_height = max_height
        self.margin = margin
        self.max_correspondence_distance = max_correspondence_distance
        
        # ANCHOR: get all scene tokens, for each token, has sd_tokens, which stands for sample ids for a specific scene....
        self.sd_token_list = []
        for scene_token in tqdm(self.scene_tokens):
            # Get records from DB.
            scene_rec = nusc.get('scene', scene_token)
            start_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = nusc.get('sample_data', start_sample_rec['data'][channel])

            # Make list of frames
            cur_sd_rec = sd_rec
            sd_tokens = []
            sd_tokens.append(cur_sd_rec['token'])
            while cur_sd_rec['next'] != '':
                cur_sd_rec = nusc.get('sample_data', cur_sd_rec['next'])
                sd_tokens.append(cur_sd_rec['token'])
            
            sd_tokens = sd_tokens[0:-1:sample_every]
            self.sd_token_list.append(sd_tokens)
        self.iter_tokens = iter(self.scene_tokens)

    def __getitem__(self, index):
        scene_token = next(self.iter_tokens)
        sd_tokens = self.sd_token_list[index]
        
        # SECTION: 1. get all point clouds, similar to lidar sweeps in argoverse data
        lidar_sweeps = []
        for i in tqdm(range(len(sd_tokens))):
            sc_rec = self.nusc.get('sample_data', sd_tokens[i])
            sample_rec = self.nusc.get('sample', sc_rec['sample_token'])
            lidar_token = sc_rec['token']
            lidar_rec = self.nusc.get('sample_data', lidar_token)
            pc1_nusc = LidarPointCloud.from_file(f"{self.nusc.dataroot}/{lidar_rec['filename']}")

            # Points live in their own reference frame. So they need to be transformed via global to the image plane.
            # First step: transform the point cloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record1 = self.nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
            pc1_nusc.rotate(Quaternion(cs_record1['rotation']).rotation_matrix)
            pc1_nusc.translate(np.array(cs_record1['translation']))

            # Optional Filter by distance to remove the ego vehicle.
            dists_origin = np.sqrt(np.sum(pc1_nusc.points[:3, :] ** 2, axis=0))
            keep = np.logical_and(self.min_dist <= dists_origin, dists_origin <= self.max_dist)
            pc1_nusc.points = pc1_nusc.points[:, keep]

            # Second step: transform to the global frame.
            poserecord1 = self.nusc.get('ego_pose', lidar_rec['ego_pose_token'])
            pose = SE3(rotation=quat2rotmat(poserecord1["rotation"]), translation=np.array(poserecord1["translation"]))
            
            pc1 = pc1_nusc.points.T[:, :3].copy()
            # Remove points above certain height.
            pc1 = pc1[pc1[:, 2] <= self.max_height]
            # Remove point below ground. Some noisy points!
            pc1 = pc1[pc1[:, 2] > -1]

            # Get tracks.
            _, boxes1, _ = self.nusc.get_sample_data(lidar_token, use_flat_vehicle_coordinates=True)
            objects1 = [box.corners().T for box in boxes1]
            sample_annotation_tokens1 = sample_rec['anns']
            
            bundle = LidarData(pc1, pose, i, scene_token, boxes1, objects1, sample_annotation_tokens1)
            lidar_sweeps.append(bundle)
        
        # SECTION: 2. get objects, get pseudo flows
        n1 = lidar_sweeps[0].ply.shape[0]
        
        mask1_tracks_flow = []
        mask2_tracks_flow = []

        flow = np.zeros((n1, 3), dtype='float32')
        
        # ANCHOR: for annotation tokens of the reference lidar sweep (sweep 0), get object for each token
        objects1 = []
        objects1_metadata = []
        for token in lidar_sweeps[0].annotation_token:
            object1_metadata = self.nusc.get('sample_annotation', token)
            objects1_metadata.append(object1_metadata)
            object1 = get_object(object1_metadata)   # object1 is already a single object!
            objects1.append(object1)
            
        # ANCHOR: get other object instance token to see if the first frame shown in other frames
        objects2 = []
        objects2_metadata = []
        for token in lidar_sweeps[1].annotation_token:
            object2_metadata = self.nusc.get('sample_annotation', token)
            objects2_metadata.append(object2_metadata)
            object2 = get_object(object2_metadata)   # object1 is already a single object!
            objects2.append(object2)
            
        objects2_track_id_dict = {object2.track_id:object2 for object2 in objects2}

        for _ in range(len(objects1)):
            # NOTE: using boxes
            box1 = [box for box in lidar_sweeps[0].boxes if box.token == object1.token][0]
            
            # Add a margin to the cuboids. Some cuboids are very tight and might lose some points.
            box1.wlh = box1.wlh + self.margin
            bbox1_3d = box1.corners().T
            inbox_pc1, is_valid = filter_point_cloud_to_bbox_3D_vectorized(bbox1_3d, lidar_sweeps[0].ply)
            
            indices = np.where(is_valid == True)[0]
            mask1_tracks_flow.append(indices)

            if object1.track_id in objects2_track_id_dict:
                object2 = objects2_track_id_dict[object1.track_id]
                box2 = [box for box in lidar_sweeps[1].boxes if box.token == object2.token][0]
                box_pose1 = SE3(rotation=box1.rotation_matrix, translation=np.array(box1.center))
                box_pose2 = SE3(rotation=box2.rotation_matrix, translation=np.array(box2.center))
                relative_pose_1_2 = box_pose2.right_multiply_with_se3(box_pose1.inverse())
                inbox_pc1_t = relative_pose_1_2.transform_point_cloud(inbox_pc1)

                translation = inbox_pc1_t - inbox_pc1
                flow[indices, :] = translation
                
                bbox2_3d = box2.corners().T 
                inbox_pc2, is_valid2 = filter_point_cloud_to_bbox_3D_vectorized(bbox2_3d, lidar_sweeps[1].ply)
                mask2_tracks_flow.append(np.where(is_valid2 == True)[0])

        # Compensate egomotion to get rigid flow.
        map_relative_to_base = lidar_sweeps[1].pose
        map_relative_to_other = lidar_sweeps[0].pose
        other_to_base = map_relative_to_base.inverse().right_multiply_with_se3(map_relative_to_other)
        points = lidar_sweeps[0].ply
        points_t = other_to_base.transform_point_cloud(points)

        # NOTE: because of the overlaps between object bounding box, need to find unique tracks
        if not mask1_tracks_flow:
            print('mask1 tracks flow is empty!')
            mask1_tracks_flow = np.array(mask1_tracks_flow)
        else:
            mask1_tracks_flow = np.unique(np.hstack(mask1_tracks_flow))
        full_mask1 = np.arange(len(lidar_sweeps[0].ply))
        mask1_no_tracks = np.setdiff1d(full_mask1, mask1_tracks_flow, assume_unique=True)
        if not mask2_tracks_flow:
            print('mask2 tracks flow is empty!')
            mask2_tracks_flow = np.array(mask2_tracks_flow)
        else:
            mask2_tracks_flow = np.unique(np.hstack(mask2_tracks_flow))
        full_mask2 = np.arange(len(lidar_sweeps[1].ply))
        mask2_no_tracks = np.setdiff1d(full_mask2, mask2_tracks_flow, assume_unique=True)

        # NOTE: unnecessary steps with ICP, but just keep it here for consistency....
        pc1_o3d = o3d.geometry.PointCloud()
        pc1_o3d.points = o3d.utility.Vector3dVector(points_t[mask1_no_tracks])
        pc2_o3d = o3d.geometry.PointCloud()
        pc2_o3d.points = o3d.utility.Vector3dVector(lidar_sweeps[1].ply[mask2_no_tracks])
        
        # Apply point-to-point ICP
        trans_init = np.identity(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pc1_o3d, pc2_o3d, 
            self.max_correspondence_distance, 
            trans_init, 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), 
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        pc1_t_o3d = pc1_o3d.transform(reg_p2p.transformation)
        points_t_refined = np.asarray(pc1_t_o3d.points)

        rigid_flow = points_t_refined - points[mask1_no_tracks]
        flow[mask1_no_tracks] = rigid_flow

        # ANCHOR: get point clouds
        pc1 = lidar_sweeps[0].ply
        pc2 = lidar_sweeps[1].ply

        # SECTION: 3. save processed data
        dataset_path = os.path.join(self.save_fi_name, lidar_sweeps[0].scene_token)
        outfile = str(lidar_sweeps[0].timestamp) + '_' + str(lidar_sweeps[1].timestamp) + '.npz'

        full_path = os.path.join(dataset_path, outfile)
        Path(dataset_path).mkdir(parents=True, exist_ok=True)

        np.savez_compressed(full_path, pc1=pc1, pc2=pc2, flow=flow, mask1_tracks_flow=mask1_tracks_flow, mask2_tracks_flow=mask2_tracks_flow)
        
        return

    def __len__(self):
        return len(self.datapath)


def get_object(label):
    translation = np.array(label['translation'])
    quaternion = np.array(label['rotation'])
    width, length, height = label['size']   # NOTE: wlh in nuscenes!
    token = label['token']
    track_id = label['instance_token']
    
    # NOTE: don't have these in nuScenes, but just have them here for consistency
    if "occlusion" in label:
        occlusion = label["occlusion"]
    else:
        occlusion = 0

    if "label_class" in label:
        label_class = label["label_class"]
        if "name" in label_class:
            label_class = label_class["name"]
    else:
        label_class = None
        
    if "score" in label:
        score = label["score"]
    else:
        score = 1.0
        
    obj_rec = ObjectLabelRecord(
        quaternion,
        translation,
        length,
        width,
        height,
        occlusion,
        label_class,
        track_id,
        token,
        score,
    )
    
    return obj_rec


class ObjectLabelRecord:
    def __init__(
        self,
        quaternion: np.ndarray,
        translation: np.ndarray,
        length: float,
        width: float,
        height: float,
        occlusion: int,
        label_class: Optional[str] = None,
        track_id: Optional[str] = None,
        token: Optional[str] = None,
        score: float = 1.0,
    ) -> None:
        """Create an ObjectLabelRecord.

        Args:
           quaternion: Numpy vector representing quaternion, box/cuboid orientation
           translation: Numpy vector representing translation, center of box given as x, y, z.
           length: object length.
           width: object width.
           height: object height.
           occlusion: occlusion value.
           label_class: class label, see object_classes.py for all possible class in argoverse
           track_id: object track id, this is unique for each track
        """
        self.quaternion = quaternion
        self.translation = translation
        self.length = length
        self.width = width
        self.height = height
        self.occlusion = occlusion
        self.label_class = label_class
        self.track_id = track_id
        self.token = token
        self.score = score

    def as_2d_bbox(self) -> np.ndarray:
        """Construct a 2D bounding box from this label.

        Length is x, width is y, and z is height

        Alternatively could write code like::

            x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
            y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
            z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
            corners = np.vstack((x_corners, y_corners, z_corners))
        """
        bbox_object_frame = np.array(
            [
                [self.length / 2.0, self.width / 2.0, self.height / 2.0],
                [self.length / 2.0, -self.width / 2.0, self.height / 2.0],
                [-self.length / 2.0, self.width / 2.0, self.height / 2.0],
                [-self.length / 2.0, -self.width / 2.0, self.height / 2.0],
            ]
        )

        egovehicle_SE3_object = SE3(rotation=quat2rotmat(self.quaternion), translation=self.translation)
        bbox_in_egovehicle_frame = egovehicle_SE3_object.transform_point_cloud(bbox_object_frame)
        return bbox_in_egovehicle_frame

    def as_3d_bbox(self) -> np.ndarray:
        r"""Calculate the 8 bounding box corners.

        Args:
            None

        Returns:
            Numpy array of shape (8,3)

        Corner numbering::

             5------4
             |\\    |\\
             | \\   | \\
             6--\\--7  \\
             \\  \\  \\ \\
         l    \\  1-------0    h
          e    \\ ||   \\ ||   e
           n    \\||    \\||   i
            g    \\2------3    g
             t      width.     h
              h.               t.

        First four corners are the ones facing forward.
        The last four are the ones facing backwards.
        """
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = self.length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = self.width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = self.height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners_object_frame = np.vstack((x_corners, y_corners, z_corners)).T

        return corners_object_frame
    
    def as_3d_bbox_ego(self) -> np.ndarray:
        r"""Calculate the 8 bounding box corners.

        Args:
            None

        Returns:
            Numpy array of shape (8,3)

        Corner numbering::

             5------4
             |\\    |\\
             | \\   | \\
             6--\\--7  \\
             \\  \\  \\ \\
         l    \\  1-------0    h
          e    \\ ||   \\ ||   e
           n    \\||    \\||   i
            g    \\2------3    g
             t      width.     h
              h.               t.

        First four corners are the ones facing forward.
        The last four are the ones facing backwards.
        """
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = self.length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = self.width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = self.height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners_object_frame = np.vstack((x_corners, y_corners, z_corners)).T

        egovehicle_SE3_object = SE3(rotation=quat2rotmat(self.quaternion), translation=self.translation)
        corners_egovehicle_frame = egovehicle_SE3_object.transform_point_cloud(corners_object_frame)
        return corners_egovehicle_frame
    
    # def as_3d_bbox_flat(self) -> np.ndarray:
    #     x_corners = self.length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    #     y_corners = self.width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    #     z_corners = self.height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    #     corners_object_frame = np.vstack((x_corners, y_corners, z_corners)).T
        
    #     egovehicle_SE3_object = SE3(rotation=quat2rotmat(self.quaternion), translation=self.translation)
    #     corners_egovehicle_frame = egovehicle_SE3_object.transform_point_cloud(corners_object_frame)
        
    #     yaw = Quaternion(self.quaternion).yaw_pitch_roll[0]
    #     box =translate(-np.array(self.translation))
    #     box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
    #     return box

    def render_clip_frustum_cv2(
        self,
        img: np.ndarray,
        corners: np.ndarray,
        planes: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        camera_config: CameraConfig,
        colors: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]] = (
            BLUE_RGB,
            RED_RGB,
            GREEN_RGB,
        ),
        linewidth: int = 2,
    ) -> np.ndarray:
        r"""We bring the 3D points into each camera, and do the clipping there.

        Renders box using OpenCV2. Roughly based on
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes_utils/data_classes.py

        ::

                5------4
                |\\    |\\
                | \\   | \\
                6--\\--7  \\
                \\  \\  \\ \\
            l    \\  1-------0    h
             e    \\ ||   \\ ||   e
              n    \\||    \\||   i
               g    \\2------3    g
                t      width.     h
                 h.               t.

        Args:
            img: Numpy array of shape (M,N,3)
            corners: Numpy array of shape (8,3) in camera coordinate frame.
            planes: Iterable of 5 clipping planes. Each plane is defined by 4 points.
            camera_config: CameraConfig object
            colors: tuple of RGB 3-tuples, Colors for front, side & rear.
                defaults are    0. blue (0,0,255) in RGB and (255,0,0) in OpenCV's BGR
                                1. red (255,0,0) in RGB and (0,0,255) in OpenCV's BGR
                                2. green (0,255,0) in RGB and BGR alike.
            linewidth: integer, linewidth for plot

        Returns:
            img: Numpy array of shape (M,N,3), representing updated image
        """

        def draw_rect(selected_corners: np.ndarray, color: Tuple[int, int, int]) -> None:
            prev = selected_corners[-1]
            for corner in selected_corners:
                draw_clipped_line_segment(
                    img,
                    prev.copy(),
                    corner.copy(),
                    camera_config,
                    linewidth,
                    planes,
                    color,
                )
                prev = corner

        # Draw the sides in green
        for i in range(4):
            # between front and back corners
            draw_clipped_line_segment(
                img,
                corners[i],
                corners[i + 4],
                camera_config,
                linewidth,
                planes,
                colors[2][::-1],
            )

        # Draw front (first 4 corners) in blue
        draw_rect(corners[:4], colors[0][::-1])
        # Draw rear (last 4 corners) in red
        draw_rect(corners[4:], colors[1][::-1])

        # grab the top vertices
        center_top = np.mean(corners[TOP_VERT_INDICES], axis=0)
        uv_ct, _, _, _ = proj_cam_to_uv(center_top.reshape(1, 3), camera_config)
        uv_ct = uv_ct.squeeze().astype(np.int32)  # cast to integer

        if label_is_closeby(center_top) and uv_coord_is_valid(uv_ct, img):
            top_left = (uv_ct[0] - BKGRND_RECT_OFFS_LEFT, uv_ct[1] - BKGRND_RECT_OFFS_UP)
            bottom_right = (uv_ct[0] + BKGRND_RECT_OFFS_LEFT, uv_ct[1] + BKGRND_RECT_OFFS_DOWN)
            img = draw_alpha_rectangle(img, top_left, bottom_right, EMERALD_RGB, alpha=BKGRND_RECT_ALPHA)
            add_text_cv2(img, text=str(self.label_class), x=uv_ct[0] - TEXT_OFFS_LEFT, y=uv_ct[1], color=WHITE_BGR)

        # Draw blue line indicating the front half
        center_bottom_forward = np.mean(corners[2:4], axis=0)
        center_bottom = np.mean(corners[[2, 3, 7, 6]], axis=0)
        draw_clipped_line_segment(
            img,
            center_bottom,
            center_bottom_forward,
            camera_config,
            linewidth,
            planes,
            colors[0][::-1],
        )

        return img
    

def uv_coord_is_valid(uv: np.ndarray, img: np.ndarray) -> bool:
    """Check if 2d-point lies within 3-channel color image boundaries"""
    h, w, _ = img.shape
    return bool(uv[0] >= 0 and uv[1] >= 0 and uv[0] < w and uv[1] < h)


def label_is_closeby(box_point: np.ndarray) -> bool:
    """Check if 3d cuboid pt (in egovehicle frame) is within range from
    egovehicle to prevent plot overcrowding.
    """
    return bool(np.linalg.norm(box_point) < MAX_RANGE_THRESH_PLOT_CATEGORY)


def draw_alpha_rectangle(
    img: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color_rgb: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Alpha blend colored rectangle into image. Corner coords given as (x,y) tuples"""
    img_h, img_w, _ = img.shape
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = 1
    return vis_mask(img, mask, np.array(list(color_rgb[::-1])), alpha)


if __name__ == "__main__":
    dataset_path = '/dataset/nuscenes'   # NOTE: path to the original dataset
    
    # ANCHOR: hyperparameters
    channel = 'LIDAR_TOP'
    sample_every = 2   # nuscenes default: 20 Hz, we use 10 Hz here
    min_dist = 3.0
    max_dist = 50.0
    max_height = 4.0
    margin = 0.6
    max_correspondence_distance = 1.0
    
    scenes = create_splits_scenes()
    split = 'val'
    scenes = scenes[split]
    save_fi_name = ''
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataset_path, verbose=True)
    scene_tokens = set()
    for scene in nusc.scene:
        if scene['name'] in scenes:
            scene_tokens.add(scene['token'])

    dataset = PreProcessNuScenesDataset(
        save_fi_name,
        nusc, scene_tokens,
        channel,
        sample_every,
        min_dist, max_dist,
        max_height,
        margin,
        max_correspondence_distance
    )
    
    # NOTE: for each scene, get sampels for pseudo scene flow
    for i, _ in enumerate(dataset):
        print('id is {}, working on scene {}'.format(i, dataset.scene_tokens[i]))
        # pass
