.PHONY: image

image: weights/detector.pt weights/encoder.npy weights/pose_estimator.pt
	bash ./build.sh

weights/detector.pt:
	curl -L https://github.com/robberthofmanfm/yolo/releases/download/v0.0.1/obj19_detector.pt -o weights/detector.pt

weights/encoder.npy:
	curl -L https://github.com/robberthofmanfm/yolo/releases/download/v0.0.1/encoder.npy -o weights/encoder.npy

weights/pose_estimator.pt:
	curl -L https://github.com/robberthofmanfm/yolo/releases/download/v0.0.1/obj_19_pose_estimator_model-epoch199.pt -o weights/pose_estimator.pt
