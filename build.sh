

docker build --force-rm --build-arg UID=$( id -u $USER) --build-arg GID=$( id -g $USER) --build-arg USER=$USER -t yolo_ros .
