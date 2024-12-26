# um-food-pickup-robot

roscore
# 在不同的终端中运行以下命令
roslaunch rchomeedu_vision multi_astra.launch
roslaunch astra_camera astra.launch
roslaunch usb_cam usb_cam-test.launch
roslaunch opencv_apps face_detection.launch image:=/camera/rgb/image_raw
roslaunch kids_module say_hello.launch

rosrun your_package face_recognition_ros.py
