#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import Marker


class FollowTheGap(Node):
    def __init__(self):
        super().__init__('follow_the_gap')

        # Paraméterek
        self.declare_parameter('safe_dist', 1.0)
        self.declare_parameter('fov', 180.0)
        self.declare_parameter('speed', 0.2)
        self.declare_parameter('p_turn', 0.6)

        self.safe_dist = self.get_parameter('safe_dist').value
        self.fov = np.radians(self.get_parameter('fov').value)
        self.speed = self.get_parameter('speed').value
        self.p_turn = self.get_parameter('p_turn').value

        # ROS interfészek
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/ftg_target', 10)

        # Path követéshez 
        self.path = Path()
        self.max_path_size = 1500

        self.path_pub = self.create_publisher(Path, "/path", 10)

        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)


    # Path frissítése odometria alapján
    def odom_callback(self, msg: Odometry):
        # Path memória limit (régi pontok törlése)
        if len(self.path.poses) >= self.max_path_size:
            del self.path.poses[0:int(self.max_path_size * 0.2)]

        # Új pozíció hozzáadása
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        self.path.header = msg.header
        self.path.poses.append(pose)

        # Path publikálása
        self.path_pub.publish(self.path)


    # LIDAR adat feldolgozása
    def scan_callback(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=float)
        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = msg.range_max

        N = len(ranges)
        inc = msg.angle_increment  # rad/minta

        # fél FOV mintaszámban kiszámolva
        k = int(round((self.fov / 2.0) / inc))
        front_ranges = np.concatenate([ranges[-k:], ranges[:k]])

        # Szögek kiszámolása
        angles = msg.angle_min + np.arange(N) * inc
        front_angles = np.concatenate([angles[-k:], angles[:k]])

        # Legnagyobb rés előszámolása
        if len(front_ranges) == 0:
            self.get_logger().warn("Nincs érvényes adat!")
            return

        gap_start, gap_end = self.find_largest_gap(front_ranges)

        if gap_start is None:
            self.get_logger().warn("Nincs érvényes rés!")
            return

        gap_center_idx = (gap_start + gap_end) // 2
        angle = front_angles[gap_center_idx]
        start_idx = gap_start - 1 if gap_start > 0 else gap_start
        end_idx = gap_end + 1 if gap_end < len(front_ranges) - 1 else gap_end

        start_dist = front_ranges[start_idx]
        end_dist = front_ranges[end_idx]
        mean_dist = (start_dist + end_dist) / 2.0
        clamped_dist = np.clip(mean_dist, 0.5, 1.0)
        x = clamped_dist * np.cos(angle)
        y = clamped_dist * np.sin(angle)

        # Marker
        marker = Marker()
        marker.header.frame_id = "base_scan"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0
        self.marker_pub.publish(marker)

        # Irányítás
        twist = Twist()
        twist.linear.x = self.speed
        angle_drive = math.atan2(y, x)
        twist.angular.z = self.p_turn * angle_drive
        self.cmd_pub.publish(twist)
        

    # Legnagyobb rés keresése
    def find_largest_gap(self, data):
        nonzero = (data > self.safe_dist)
        max_len = 0
        max_start = max_end = None
        start = None

        for i, ok in enumerate(nonzero):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                length = i - start
                if length > max_len:
                    max_len = length
                    max_start, max_end = start, i - 1
                start = None

        if start is not None:
            length = len(data) - start
            if length > max_len:
                max_len = length
                max_start, max_end = start, len(data) - 1

        return max_start, max_end


def main(args=None):
    rclpy.init(args=args)
    node = FollowTheGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()