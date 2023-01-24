#!/usr/bin/env python3

import typing
import rclpy
from rclpy.node import Node
from tf_transformations import quaternion_from_matrix
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Pose, PoseArray

import torch
import numpy as np
from Encoder import Encoder
from Model import Model


# TODO(aris): Use depth image instead of pointcloud, it is computationally less intensive and is
#  already aligned and registered to the RGB image.


class PoseNode(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")
        self.get_logger().info(f"Starting {self.get_name()} node")
        self._rgb = None  # type: typing.Optional[np.array]
        self._depth = None  # type: typing.Optional[np.array]
        self._points = None  # type: typing.Optional[PointCloud2]

        # Load Parameters
        self.declare_parameter("detector_weight_path", "./weights/detector.pt")
        self.declare_parameter("detector_repository", "./")
        self.declare_parameter("detector_confidence", 0.2)
        self.declare_parameter("encoder_weights", "./weights/encoder.npy")
        self.declare_parameter("pose_estimator_weights", "./weights/pose_estimator.pt")

        # Load Inference Model
        # Load YOLO detector
        self._detector = torch.hub.load(self.get_parameter("detector_repository").value,
                                        "custom",
                                        path=self.get_parameter("detector_weight_path").value,
                                        source="local")
        self._detector.conf = self.get_parameter("detector_confidence").value
        # Load AutoEncoder
        self._device = torch.device("cuda:0")
        self._encoder = Encoder(self.get_parameter("encoder_weights").value).to(self._device)
        self._encoder.eval()  # set model to inference mode
        # Load Pose Estimator
        self._model, self._num_views = \
            self._load_checkpoint(self.get_parameter("pose_estimator_weights").value)
        self._model.eval()  # set model to inference mode

        # Create ROS stuff
        self._bridge = CvBridge()
        # Create Pubs/Subs
        self._pose_est_pub = self.create_publisher(PoseArray, "pose_estimation", 10)
        transport = self.declare_parameter("transport", "raw").value
        if transport == "raw":
            self._rgb_sub = self.create_subscription(Image, "rgb/image_raw",
                                                     self.image_callback, 10)
        elif transport == "compressed":
            self._rgb_sub = self.create_subscription(CompressedImage, "rgb/image_raw/compressed",
                                                     self.compressed_image_callback, 10)

        # self._depth_sub = self.create_subscription(Image, "depth/image_raw",
        #                                            self.depth_callback, 10)
        self._points_sub = self.create_subscription(PointCloud2, "points2",
                                                    self.points_callback, 10)

    def _load_checkpoint(self, model_path: str):
        checkpoint = torch.load(model_path)
        num_views = int(checkpoint['model']['l3.bias'].shape[0] / (6 + 1))
        model = Model(num_views=num_views).cuda()
        model.load_state_dict(checkpoint['model'])
        self.get_logger().info(f"Loaded the checkpoint: {model_path}")
        return model, num_views

    def image_callback(self, msg: Image):
        self._rgb = self._bridge.imgmsg_to_cv2(msg, "rgb8")
        self.run_inference(msg)

    def compressed_image_callback(self, msg: CompressedImage):
        self._rgb = self._bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
        self.run_inference(msg)

    def depth_callback(self, msg: Image):
        # self._depth = self._bridge.imgmsg_to_cv2(msg)
        pass

    def points_callback(self, msg: PointCloud2):
        self._points = msg

    def run_inference(self, img_msg: typing.Union[Image, CompressedImage]):
        predictions = self._process_scene(self._rgb)
        msg = PoseArray()
        msg.header = img_msg.header
        for i, prediction in enumerate(predictions):
            transformation, bbox = self._get_depth(prediction)
            if transformation[2] < 10:  # Distance on Z!
                self.get_logger().info(transformation)
            pose = Pose()
            pose.position.x = transformation[0]
            pose.position.y = transformation[1]
            pose.position.z = transformation[2]

            mat = PoseNode.rt_to_mat(prediction['rot'], transformation)
            self.get_logger().debug(mat)
            qt = quaternion_from_matrix(mat)
            pose.orientation.x = qt[0]
            pose.orientation.y = qt[1]
            pose.orientation.z = qt[2]
            pose.orientation.w = qt[3]
            msg.poses.append(pose)
        self._pose_est_pub.publish(msg)

    def _process_crop(self, detection):
        # Disable gradients for the encoder
        with torch.no_grad():
            img = cv2.resize(detection['im'], (128, 128), interpolation=cv2.INTER_NEAREST)
            img_max = np.max(img)
            img_min = np.min(img)
            img = (img - img_min) / (img_max - img_min)
            img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(self._device)
            code = self._encoder(img.float())
            code = code.detach().cpu().numpy()[0]
            norm_code = code / np.linalg.norm(code)
        batch_codes = torch.tensor(np.stack([norm_code]), device=self._device, dtype=torch.float32)
        predicted_poses = self._model(batch_codes)
        confs = predicted_poses[:, :self._num_views]
        index = torch.argmax(confs)  # which pose has the highest confidence
        pose_start = self._num_views  # +3 if with translation
        pose_end = pose_start + 6
        curr_pose = predicted_poses[:, pose_start:pose_end]
        predicted_rotations = PoseNode.compute_rotation_matrix_from_ortho6d(curr_pose)
        return predicted_rotations.cpu().detach().numpy()[0]

    def _process_scene(self, image: np.array):
        detections = self._detector(image)

        cropped = detections.crop(save=False)
        for detection in cropped:
            predicted_rotations = self._process_crop(detection)
            predicted_rotations = PoseNode.conv_r_pytorch2opengl_np(predicted_rotations)
            detection['rot'] = predicted_rotations
        return cropped

    def _get_depth(self, detection: typing.Dict):
        box = detection['box']
        tlx = box[0].cpu()
        tly = box[1].cpu()
        brx = box[2].cpu()
        bry = box[3].cpu()
        bbox = [tlx, tly, brx, bry]
        mid_x = int((tlx + brx) / 2)
        mid_y = int((tly + bry) / 2)
        if not self._points:
            return None, bbox
        temp = point_cloud2.read_points(self._points, field_names=['x', 'y', 'z'],
                                        skip_nans=True, uvs=[(mid_x, mid_y)])
        transformation = list(list(temp)[0])
        return transformation, bbox

    @staticmethod
    def compute_rotation_matrix_from_ortho6d(poses: np.array):
        x_raw = poses[:, 0:3]  # batch*3
        y_raw = poses[:, 3:6]  # batch*3

        x = PoseNode.normalize_vector(x_raw)  # batch*3
        z = PoseNode.cross_product(x, y_raw)  # batch*3
        z = PoseNode.normalize_vector(z)  # batch*3
        y = PoseNode.cross_product(z, x)  # batch*3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        return torch.cat((x, y, z), 2)  # batch*3*3

    @staticmethod
    def normalize_vector(v: np.array):
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        return v / v_mag

    @staticmethod
    def cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        return torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    @staticmethod
    def conv_r_pytorch2opengl_np(r):
        # Convert R matrix from pytorch to opengl format
        xy_flip = np.eye(3, dtype=float)
        xy_flip[0, 0] = -1.0
        xy_flip[1, 1] = -1.0
        r_opengl = np.dot(r, xy_flip)
        return np.transpose(r_opengl)

    @staticmethod
    def conv_r_opengl2pytorch_np(r):
        # Convert R matrix from opengl to pytorch format
        xy_flip = np.eye(3, dtype=float)
        xy_flip[0, 0] = -1.0
        xy_flip[1, 1] = -1.0
        r_pytorch = np.transpose(r)
        return np.dot(r_pytorch, xy_flip)

    @staticmethod
    def rt_to_mat(rot, tran):
        mat = np.concatenate((rot, np.array(tran).reshape(3, 1)), axis=1)
        return np.concatenate((mat, [[0, 0, 0, 1]]), axis=1)


if __name__ == "__main__":
    rclpy.init()
    node = PoseNode()
    rclpy.spin(node)
