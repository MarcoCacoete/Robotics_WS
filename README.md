

I have been running my program on level 3 complexity using: 
*ros2 launch uol_tidybot tidybot.launch.py world:=level_2_3.world

Usually with 5 green and red blocks using: 
* ros2 run  uol_tidybot generate_objects --ros-args -p red:=false -p n_objects:=5
* ros2 run  uol_tidybot generate_objects --ros-args -p red:=true -p n_objects:=5


To run the program on gazebo please enter the following in the terminal:

* cd /workspaces/Robotics_WS
* colcon build --symlink-install --packages-select limo_chaser
* source install/setup.bash
* ros2 run limo_chaser limo_chaser        (to run in gazebo)
* ros2 run limo_chaser limo_chaser --bot  (to run on real robot)


