#! /usr/bin/python3
import torch
import numpy as np
import torch.nn as nn
import cv2
from Encoder import Encoder
from Model import Model

import rospy
#from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2

# local copy to avoid relying on utils due to name clash
def loadCheckpoint(model_path):
    # Load checkpoint and parameters
    checkpoint = torch.load(model_path)

    # Load model
    num_views = int(checkpoint['model']['l3.bias'].shape[0]/(6+1))
    model = Model(num_views=num_views).cuda()

    model.load_state_dict(checkpoint['model'])

    print("Loaded the checkpoint: \n" + model_path)
    return model, num_views

# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3

    return out

# again copy to avoid importing utils
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3

    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

class Inference():
    def __init__(self):
        detector_weight_path = "/home/hampus/vision/yolov3/runs/train/exp4/weights/best.pt"
        detector_model = "/home/hampus/vision/yolov3"
        encoder_weights = "/home/hampus/vision/AugmentedAutoencoder/multi-pose/data/encoder/obj1-18/encoder.npy"
        model_path = "/home/hampus/vision/AugmentedAutoencoder/multi-pose/output/test/models/model-epoch0.pt"

        device = torch.device("cuda:0")

        # load yolo detector
        detector = torch.hub.load('/home/hampus/vision/yolov3', 'custom', path="/home/hampus/vision/yolov3/runs/train/exp5/weights/best.pt", source='local')

        # load AE autoencoder
        encoder = Encoder(encoder_weights).to(device)
        encoder.eval()

        # load pose estimator
        model, num_views = loadCheckpoint(model_path)
        model = model.eval() # Set model to eval mode

        self.device = device
        self.detector = detector
        self.encoder = encoder
        self.model = model
        self.num_views = num_views

    def process_crop(self, detection):
        # Disable gradients for the encoder
        with torch.no_grad():
            # detect object in scene, get bboxes, crop and pass to AE for each object
            #pred = self.detector.predict(scene_image)
            #print(type(scene_image))
            #print(scene_image.shape)
            #pred = self.detector.run([self.detector.get_outputs()[0].name], {self.detector.get_inputs()[0].name: scene_image.astype(np.float32)})[0]
            #pred = detector(scene_img)

            resize=(128,128)
            interpolation=cv2.INTER_NEAREST

            image = detection['im']

            img = cv2.resize(image, resize, interpolation = interpolation)

            # Convert images to AE codes
            codes = []

            # Normalize image
            img_max = np.max(img)
            img_min = np.min(img)
            img = (img - img_min)/(img_max - img_min)

            # Run image through encoder
            img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).to(self.device)
            #print(img.shape)
            code = self.encoder(img.float())
            code = code.detach().cpu().numpy()[0]
            norm_code = code / np.linalg.norm(code)

        # Predict poses from the codes
        batch_codes = torch.tensor(np.stack([norm_code]), device=self.device, dtype=torch.float32)
        predicted_poses = self.model(batch_codes)
        confs = predicted_poses[:,:self.num_views]
        # which pose has highest conf
        index = torch.argmax(confs)
        pose_start = self.num_views # +3 if with translation
        pose_end = pose_start + 6
        curr_pose = predicted_poses[:,pose_start:pose_end]
        Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
        return Rs_predicted.cpu().detach().numpy()[0]

    def process_scene(self, image):
        detections = self.detector(image)

        crops_det = detections.crop(save=False) # set to True to debug crops
        for crop_det in crops_det:
            Rs_predicted = self.process_crop(crop_det)
            crop_det['rot'] = Rs_predicted
        return crops_det

class Pose_estimation_rosnode():
    def __init__(self):
        #self.bridge = CvBridge()
        self.inference = Inference()
        self.depth = None
        self.points = None
        self.pub = rospy.Publisher('pose_estimation', String, queue_size=10)
        rospy.init_node('pose_estimation_hampus', anonymous=True)
        rospy.Subscriber('/realsense/aligned_depth_to_color/image_raw', Image, callback=self.depth_callback)
        rospy.Subscriber('/realsense/rgb/image_raw', Image, callback=self.run_callback)
        rospy.Subscriber('/realsense/depth/points', PointCloud2, callback=self.points_callback)

        rospy.spin()

    def mid_depth(self, detection):
        box = detection['box']
        tlx = box[0].cpu()
        tly = box[1].cpu()
        brx = box[2].cpu()
        bry = box[3].cpu()

        bbox = [tlx, tly, brx, bry]

        mid_x = (tlx + brx)/2
        mid_y = (tly + bry)/2

        if self.points == None:
            return None, bbox

        #T = self.points[mid_x, mid_y]
        T = 0

        # temp stupid estimate of x and y
        #x_pix = mid_x - self.image.shape[0]/2
        #y_pix = mid_y - self.image.shape[1]/2

        return T, bbox

    def depth_callback(self, depth_data):
        #self.depth = self.bridge.imgmsg_to_cv2(data, 'passthrough')
        self.depth = np.frombuffer(depth_data.data, dtype=np.uint8).reshape(depth_data.height, depth_data.width, -1)

    def points_callback(self, point_data):
        #self.points = self.bridge.imgmsg_to_cv2(data, 'passthrough')
        print("points_callback")
        self.points = np.frombuffer(point_data.data, dtype=np.uint8).reshape(point_data.height, point_data.width, -1)
        print(self.points.shape)

    def run_callback(self, image_data):
        #image = self.bridge.imgmsg_to_cv2(data, 'passthrough')
        image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)

        pred = inference.process_scene(image)

        rot = []
        Ts = []
        for p in pred:
            T, bbox = self.mid_depth(p)

            rot.append(p['rot'])
            Ts.append(T)

        rot = np.array(rot)
        Ts = np.array(Ts)
        #print(ret)
        self.pub.publish("{} {}".format(np.array_str(rot), np.array_str(Ts)))

# main method only used for testing
if __name__ == '__main__':

    inference = Inference()

    #test_img_path = '/home/hampus/vision/AugmentedAutoencoder/multi-pose/detection_data/images/1.png'
    #bgr = cv2.imread(test_img_path)

    #pred = inference.process_scene(bgr)
    #print(pred)

    pose_estimation_rosnode = Pose_estimation_rosnode()
