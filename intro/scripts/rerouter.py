#! /usr/bin/python2

import math
import rospy
from turtlesim.msg import Pose
from turtlesim.srv import Spawn
from geometry_msgs.msg import Twist

cmd_vel_pub = None


class Rerouter:
    def __init__(self):
        self.leo_pub = rospy.Publisher('/leo/cmd_vel', Twist, queue_size=10)
        self.leo = None
        self.turtle1 = None

    def record_leo_pos(self, msg):
        self.leo = msg
        self.route_if_needed()

    def record_turle1_pos(self, msg):
        self.turtle1 = msg
        self.route_if_needed()

    def route_if_needed(self):
        if self.leo is None or self.turtle1 is None:
            return
        leo, self.leo = self.leo, None
        turtle1, self.turtle1 = self.turtle1, None
        x0, y0 = leo.x, leo.y
        x1, y1 = turtle1.x, turtle1.y
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        if distance < 1.0:
            return
        dx = x1 - x0
        dy = y1 - y0
        theta = math.atan2(dy, dx)
        delta = theta - leo.theta
        twist = Twist()
        twist.linear.x = 1.0
        twist.angular.z = delta
        self.leo_pub.publish(twist)


rospy.wait_for_service('/spawn')
spawn_call = rospy.ServiceProxy('/spawn', Spawn)
spawn_call(5.0, 5.0, 3.1415, 'leo')

rospy.init_node('leo_rerouter')
rerouter = Rerouter()
rospy.Subscriber('/leo/pose', Pose, rerouter.record_leo_pos)
rospy.Subscriber('/turtle1/pose', Pose, rerouter.record_turle1_pos)
rospy.spin()

