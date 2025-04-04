so2Nodes - Executable code, can communicate with other nodes, through topics, can subscribe to topics which is data being sent between them.

git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"

for github on school pcs.

Commands: 
- Start with ros2 
    - launch uol_tidybot tidybot.launch.py:
    - node 
        - list
     - topic 
        - list
    - param
        - list
            - node name
    - run
        - topic name Ex: Ros2 run scan scan


Plan: 
* Priority rule, if the block matches the wall behind it, push that block until wall distance is close to 0.
To make sure block is still being pushed, pull robot back some distance when block goes out of sight, then push once more for x distance.
* if a block is in sight, and on obstacles in its path, head for the object, use lidar range at certain angles to create cone to detect obstacles. Avoid them by re adjusting location.
* If no blocks are initially visible, rotate 360. If none are still visible advance to close to a wall or obstacle, rotate to keep parallel with object, navigate around object, keeping distance to it in same range, until the conditions above are present.

ros2 launch uol_tidybot tidybot.launch.py world:=level_2_1.world

ros2 launch uol_tidybot tidybot.launch.py world:=level_2_2.world

ros2 launch uol_tidybot tidybot.launch.py world:=level_2_3.world

ros2 run  uol_tidybot generate_objects --ros-args -p red:=false -p n_objects:=5
ros2 run  uol_tidybot generate_objects --ros-args -p red:=true -p n_objects:=5


# 1. FIRST-TIME SETUP (runs once)
cd /workspaces/Robotics_WS
colcon build --symlink-install --packages-select limo_chaser
source install/setup.bash

# 2. AFTER EDITING Assignment.py (normal workflow):
# Option A: If you only changed Python code (no new imports):
ros2 run limo_chaser limo_chaser --bot

# Option B: If you added new imports or dependencies:
colcon build --symlink-install --packages-select limo_chaser
source install/setup.bash
ros2 run limo_chaser limo_chaser --bot

# 3. NUCLEAR OPTION (if something breaks):
rm -rf build/limo_chaser install/limo_chaser && \
colcon build --symlink-install --packages-select limo_chaser && \
source install/setup.bash && \
ros2 run limo_chaser limo_chaser --bot

# 4. VERIFICATION COMMANDS:
# Check node is running:
ros2 node list
# Check topics:
ros2 topic list
# Monitor velocity commands:
ros2 topic echo /cmd_vel
# Check Python symlinks:
ls -l install/limo_chaser/lib/python3.8/site-packages/limo_chaser/Assignment.py