#!/usr/bin/env python

# A general ROS 2 Python template for creating a node with subscribers, publishers, and timers
# Fill in your specific use case in the marked sections

import rclpy
from rclpy.node import Node
# Import your required ROS message types (e.g., sensor_msgs.msg, geometry_msgs.msg)
# Add more as needed for your use case
from std_msgs.msg import String  # Example message type; replace or add others

class MyNode(Node):
    def __init__(self):
        # Initialize the node with a unique name
        super().__init__('my_node')  # Replace 'my_node' with your node's name

        # Declare variables you’ll use across callbacks
        self.my_variable = 0.0  # Example; replace with variables for your use case

        # ---- Publisher Setup ----
        # Create a publisher to send data to a topic
        # Replace 'topic_name' with your topic, and String with your message type
        self.publisher = self.create_publisher(String, 'topic_name', 10)
        # '10' is the queue size (how many messages to buffer)

        # ---- Timer Setup ----
        # Create a timer to call a function periodically (e.g., to publish data)
        timer_period = 0.5  # seconds; adjust as needed
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # ---- Subscriber Setup ----
        # Create a subscription to listen to a topic
        # Replace 'input_topic' with your topic, String with your message type,
        # and 'subscriber_callback' with your callback function name
        self subscription = self.create_subscription(
            String, 'input_topic', self.subscriber_callback, 10
        )

        # Log a message to confirm the node is running
        self.get_logger().info('Node has started successfully!')

    def subscriber_callback(self, msg):
        # This function is called whenever a message is received on the subscribed topic
        # 'msg' contains the data from the topic
        # Replace this with your data processing logic
        self.get_logger().info(f'Received: {msg.data}')  # Example for String msg
        
        # ---- YOUR PROCESSING LOGIC HERE ----
        # Example: Store or process the incoming data
        self.my_variable = msg.data  # Adapt based on your message type and needs

    def timer_callback(self):
        # This function is called periodically based on the timer_period
        # Use it to publish data or perform recurring tasks
        
        # ---- YOUR PUBLISHING LOGIC HERE ----
        # Example: Create and publish a message
        msg = String()  # Replace String with your message type
        msg.data = f'Value: {self.my_variable}'  # Replace with your data
        self.publisher.publish(msg)
        
        # Optional: Log what’s happening
        self.get_logger().info(f'Published: {msg.data}')

def main(args=None):
    # Main function to start the ROS node
    print('Starting the ROS 2 node...')
    
    # Initialize the ROS 2 system
    rclpy.init(args=args)
    
    # Create an instance of your node
    node = MyNode()
    
    # Keep the node running and processing callbacks
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        node.get_logger().info('Shutting down node...')
    
    # Clean up: Destroy the node and shut down ROS 2
    node.destroy_node()
    rclpy.shutdown()
    print('Node stopped.')

if __name__ == '__main__':
    # Run the main function if this script is executed directly
    main()