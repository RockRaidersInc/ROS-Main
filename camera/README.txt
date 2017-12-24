Author(s):
    - Owen Xie

The camera node uses OpenCV 3, which comes default with ROS Kinectic

To use the package, you can either use rosrun or roslaunch
the camera.launch file, such as:

    rosrun camera camera_stream camera _camera:=[camera #, default 0]

           -- OR --

    roslaunch camera.launch

Editing camera.launch:

Add this within the <group> tags to add a camera.

<node pkg="camera" type="camera_stream" name="camera_[index of camera]">
    <param name="camera" value="[index of camera]">
</node>

Don't forget to replcae [index of camera]
