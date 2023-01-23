# Pose Estimation

## Installation
```bash
sudo apt install ros-${ROS_DISTRO}-tf-transformations

cd ${PROJECT_FOLDER}
virtualenv --system-site-packages .env
source .env/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt coremltools onnx gsutil notebook wandb>=0.12.2 transforms3d
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
## Modify for numpy
transforms3d includes a number of np.float references, you must change them to np.float_ or just 
float

## Execution
In a sourced virtual environment (source .env/bin/activate), you should also source your ROS2 
installation
```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
```
then, just navigate into the project folder and run the ros2 node like this
```bash
cd ${PROJECT_FOLDER}
./pose_estimate/ros2_node.py
```
