Nodes - Executable code, can communicate with other nodes, through topics, can subscribe to topics which is data being sent between them.

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
* 