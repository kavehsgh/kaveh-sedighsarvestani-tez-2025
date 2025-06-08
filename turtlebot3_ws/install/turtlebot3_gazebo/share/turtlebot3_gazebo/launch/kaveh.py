import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
import tf_transformations


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")
        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.navigator = BasicNavigator()
        self.navigator.waitUntilNav2Active()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        detected_objects_image, target_detected = self.detect_red_objects(cv_image)
        if target_detected:
            cv2.putText(detected_objects_image, "Target Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.set_goal(3.0, -1.0, 0.0)  # Setting a new goal at position (0,0) with 0 rotation

        resized_image = cv2.resize(detected_objects_image, (720, 405))
        cv2.imshow("Object Detection", resized_image)
        cv2.waitKey(1)

    def detect_red_objects(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = (0, 16, 172)
        upper_red = (0, 255, 255)
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_detected = False
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 10**3:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                target_detected = True
        return image, target_detected

    def set_goal(self, x, y, z_rot):
        goal_pose = self.create_pose_stamped(x, y, z_rot)
        self.navigator.goToPose(goal_pose)

    def create_pose_stamped(self, x, y, z_rot):
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, z_rot)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation.x = q[0]
        goal_pose.pose.orientation.y = q[1]
        goal_pose.pose.orientation.z = q[2]
        goal_pose.pose.orientation.w = q[3]
        return goal_pose


def main(args=None):
    rclpy.init(args=args)
    object_detection_node = ObjectDetectionNode()
    rclpy.spin(object_detection_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()