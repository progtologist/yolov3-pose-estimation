#!/bin/bash
#SHARED_FOLDER=$(dirname $(readlink -f $0) | rev | cut -d'/' -f3- | rev)
SHARED_FOLDER="$PWD/temp"

docker run --gpus all -ti --rm --network=host \
                    -e NVIDIA_VISIBLE_DEVICES=all \
                    -e NVIDIA_DRIVER_CAPABILITIES=all \
                    --volume="/etc/group:/etc/group:ro" \
                    --volume="/etc/passwd:/etc/passwd:ro" \
                    --volume="/etc/shadow:/etc/shadow:ro" \
                    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
                    -v ${SHARED_FOLDER}:/shared-folder \
                    -e DISPLAY=$DISPLAY \
                    --env QT_X11_NO_MITSHM=1 \
                    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" yolo_ros bash

#--user=$( id -u $USER ):$( id -g $USER )
