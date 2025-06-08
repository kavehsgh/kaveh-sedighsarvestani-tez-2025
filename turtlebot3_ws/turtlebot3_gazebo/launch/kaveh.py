#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy
from rclpy.duration import Duration

import time
import math
import cv2
from cv_bridge import CvBridge
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import (
    PointStamped,
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
)
import tf2_ros
import tf2_geometry_msgs  # zorunlu import
import tf_transformations
from nav2_simple_commander.robot_navigator import BasicNavigator


class RedObjectNavNode(Node):
    def __init__(self):
        super().__init__("red_object_nav_node")

        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        # RGB aboneliği
        self.rgb_msg = None
        self.create_subscription(
            Image, "/intel_realsense_r200_rgb/image_raw", self.rgb_callback, qos
        )

        # Depth aboneliği
        self.depth_msg = None
        self.create_subscription(
            Image,
            "/intel_realsense_r200_depth/depth/image_raw",
            self.depth_callback,
            qos,
        )

        # CameraInfo aboneliği → intrinsics + dist coeffs + undistort map
        self.depth_cam_info = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.map1 = None
        self.map2 = None
        self.depth_frame_id = None
        self.create_subscription(
            CameraInfo,
            "/intel_realsense_r200_depth/depth/camera_info",
            self.depth_info_callback,
            qos,
        )

        # AMCL başlangıç pozu publisher
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.initial_pose_sent = False

        # /amcl_pose aboneliği
        self.current_amcl_pose = None
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self.amcl_pose_callback, qos
        )

        # cmd_vel publisher (dönüş kontrolü için)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # TF buffer ve listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Nav2 BasicNavigator
        self.navigator = BasicNavigator()
        self.navigator.lifecycleStartup()
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 ACTIVE oldu.")

        # Timer: 1 Hz
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Durum bayrakları ve parametreler
        # 0: Nesne tespit ve dur, 1: Görsel hizalama, 2: Derinlik hesap ve ilk hedefi gönder,
        # 3: İlk hedefe yaklaşma + büyük kontur kontrolü, 4: Başlangıca dönüş, 5: Tamamlandı
        self.state = 0
        self.processing = False

        # HSV kırmızı aralığı
        self.lower_red = np.array([0, 16, 172])
        self.upper_red = np.array([0, 255, 255])

        self.br = CvBridge()

        # Kamera yüksekliği (zeminden metrede)
        self.camera_height = 0.22

        # Başlangıç pozunu saklama
        self.start_pose = None

        # Birinci hedef koordinatları
        self.first_goal_x = None
        self.first_goal_y = None

        # Görsel merkez toleransı (%30)
        self.center_tolerance_frac = 0.3

        self.get_logger().info("RedObjectNavNode başlatıldı (1 Hz).")

    # RGB callback
    def rgb_callback(self, msg: Image):
        self.rgb_msg = msg

    # Depth callback
    def depth_callback(self, msg: Image):
        self.depth_msg = msg

    # CameraInfo callback
    def depth_info_callback(self, msg: CameraInfo):
        if self.depth_cam_info is None:
            self.depth_cam_info = msg
            self.depth_frame_id = msg.header.frame_id

            fx = msg.k[0]
            fy = msg.k[4]
            cx = msg.k[2]
            cy = msg.k[5]
            self.camera_matrix = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )

            self.dist_coeffs = np.array(msg.d, dtype=np.float32)

            if self.rgb_msg is not None:
                rgb_w = self.rgb_msg.width
                rgb_h = self.rgb_msg.height
            else:
                rgb_w = int(cx * 2)
                rgb_h = int(cy * 2)

            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.camera_matrix,
                (rgb_w, rgb_h),
                cv2.CV_32FC1,
            )

            self.get_logger().info(
                f"Depth intrinsics alındı:\n"
                f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}\n"
                f"  dist_coeffs={self.dist_coeffs.tolist()}\n"
                f"  depth_frame_id='{self.depth_frame_id}'"
            )

    # AMCL pozunu saklamak için callback
    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_amcl_pose = msg

    # Ana döngü (1 Hz)
    def timer_callback(self):
        if (
            (self.rgb_msg is None)
            or (self.depth_msg is None)
            or (self.depth_cam_info is None)
        ):
            return

        # Başlangıç pozunu bir kez gönder ve kaydet
        if not self.initial_pose_sent:
            self.send_initial_pose(0.0, 0.0, 0.0)
            time.sleep(0.5)
            if self.current_amcl_pose is None:
                self.get_logger().warn("amcl_pose henüz gelmedi, bekleniyor...")
                return
            self.start_pose = PoseStamped()
            self.start_pose.header.frame_id = "map"
            self.start_pose.header.stamp = self.current_amcl_pose.header.stamp
            self.start_pose.pose = self.current_amcl_pose.pose.pose
            self.initial_pose_sent = True
            self.get_logger().info(
                "AMCL başlangıç pozu gönderildi ve start_pose kaydedildi."
            )
            return

        if self.processing:
            return

        # -------- STATE 0: Nesne tespit et ve dur --------
        if self.state == 0:
            self.processing = True
            try:
                rgb_image = self.br.imgmsg_to_cv2(self.rgb_msg, "bgr8")
                hsv_full = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
                mask_full = cv2.inRange(hsv_full, self.lower_red, self.upper_red)

                if cv2.countNonZero(mask_full) < 50:
                    self.get_logger().debug("Çok az kırmızı piksel, nesne yok.")
                    return

                contours, _ = cv2.findContours(
                    mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    self.get_logger().debug("Kontur yok.")
                    return

                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                area = w * h
                if area < 7000:
                    self.get_logger().debug(f"Noise: alan={area:.0f} < 7000.")
                    return

                # Nesne algılandı: robotu durdur
                stop_msg = Twist()
                stop_msg.linear.x = 0.0
                stop_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(stop_msg)
                self.get_logger().info(
                    "Nesne algılandı, robot durdu. State → 1 (hizalama)."
                )
                # State 1'e geç
                self.state = 1

            except Exception as ex:
                self.get_logger().error(f"timer_callback (state 0) hatası: {ex}")

            finally:
                self.processing = False

        # -------- STATE 1: Görsel hizalama --------
        elif self.state == 1:
            self.processing = True
            try:
                rgb_image = self.br.imgmsg_to_cv2(self.rgb_msg, "bgr8")
                hsv_full = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
                mask_full = cv2.inRange(hsv_full, self.lower_red, self.upper_red)

                contours, _ = cv2.findContours(
                    mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    self.get_logger().debug("State 1: Kontur yok, bekleniyor...")
                    return

                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                area = w * h
                if area < 7000:
                    self.get_logger().debug(f"State 1: Noise: alan={area:.0f} < 7000.")
                    return

                rgb_h = self.rgb_msg.height
                rgb_w = self.rgb_msg.width
                u_center = rgb_w // 2

                u_rgb = int(x + w / 2)
                tol_pixels = int((rgb_w / 2) * self.center_tolerance_frac)

                du = u_rgb - u_center

                # Eğer nesne tolerans içinde değilse, yavaşça dön
                if abs(du) > tol_pixels:
                    twist = Twist()
                    # Küçük açısal hız: 0.1 rad/s
                    twist.angular.z = 0.1 * (1 if du < 0 else -1)
                    self.cmd_vel_pub.publish(twist)
                    self.get_logger().info(
                        f"State 1: Hizalanıyor, Δu={du}, tol={tol_pixels}"
                    )
                else:
                    # Hizalama tamam: dur ve bir sonraki state'e geç
                    stop_msg = Twist()
                    stop_msg.linear.x = 0.0
                    stop_msg.angular.z = 0.0
                    self.cmd_vel_pub.publish(stop_msg)
                    self.get_logger().info(
                        "State 1: Nesne ortalandı. State → 2 (derinlik hesap)."
                    )
                    self.state = 2

            except Exception as ex:
                self.get_logger().error(f"timer_callback (state 1) hatası: {ex}")

            finally:
                self.processing = False

        # -------- STATE 2: Derinlik hesapla ve birinci hedefi gönder --------
        elif self.state == 2:
            self.processing = True
            try:
                # Derinlik medyanı hesaplama (daha önceki adımlar)
                rgb_image = self.br.imgmsg_to_cv2(self.rgb_msg, "bgr8")
                hsv_full = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
                mask_full = cv2.inRange(hsv_full, self.lower_red, self.upper_red)

                contours, _ = cv2.findContours(
                    mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    self.get_logger().debug("State 2: Kontur yok, bekleniyor...")
                    return

                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)

                rgb_h = self.rgb_msg.height
                rgb_w = self.rgb_msg.width
                depth_h = self.depth_msg.height
                depth_w = self.depth_msg.width

                x_d = int(x * depth_w / rgb_w)
                y_d = int(y * depth_h / rgb_h)
                w_d = int(w * depth_w / rgb_w)
                h_d = int(h * depth_h / rgb_h)

                pad = 5
                x0 = max(x_d + pad, 0)
                y0 = max(y_d + pad, 0)
                x1 = min(x_d + w_d - pad, depth_w - 1)
                y1 = min(y_d + h_d - pad, depth_h - 1)

                depth_image = self.br.imgmsg_to_cv2(self.depth_msg, "32FC1")
                if x1 <= x0 or y1 <= y0:
                    u_depth = int((x + w / 2) * depth_w / rgb_w)
                    v_depth = int((y + h / 2) * depth_h / rgb_h)
                    z_val = float(depth_image[v_depth, u_depth])
                else:
                    sub_region = depth_image[y0:y1, x0:x1]
                    valid = sub_region[np.isfinite(sub_region) & (sub_region > 0.0)]
                    if valid.size == 0:
                        u_depth = int((x + w / 2) * depth_w / rgb_w)
                        v_depth = int((y + h / 2) * depth_h / rgb_h)
                        z_val = float(depth_image[v_depth, u_depth])
                    else:
                        z_val = float(np.median(valid))

                if np.isnan(z_val) or (z_val <= 0.0):
                    self.get_logger().warn("State 2: Medyan derinlik geçersiz.")
                    return

                u_rgb = int(x + w / 2)
                v_rgb = int(y + h / 2)
                px = np.array([[[u_rgb, v_rgb]]], dtype=np.float32)
                und = cv2.undistortPoints(
                    px, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix
                )
                u_undist = und[0][0][0]
                v_undist = und[0][0][1]

                u_depth = int(u_undist * depth_w / rgb_w)
                v_depth = int(v_undist * depth_h / rgb_h)
                if not (0 <= u_depth < depth_w and 0 <= v_depth < depth_h):
                    u_depth = int((x + w / 2) * depth_w / rgb_w)
                    v_depth = int((y + h / 2) * depth_h / rgb_h)

                fx = self.depth_cam_info.k[0]
                fy = self.depth_cam_info.k[4]
                cx = self.depth_cam_info.k[2]
                cy = self.depth_cam_info.k[5]

                X_cam = (u_depth - cx) * z_val / fx
                Y_cam = (v_depth - cy) * z_val / fy
                Z_cam = z_val
                self.get_logger().info(
                    f"State 2: Kamera koordinatında: X={X_cam:.3f}, Y={Y_cam:.3f}, Z={Z_cam:.3f}"
                )

                # TF ile map’e dönüştür
                pt_cam = PointStamped()
                pt_cam.header.frame_id = "camera_depth_optical_frame"
                pt_cam.header.stamp = rclpy.time.Time().to_msg()
                pt_cam.point.x = X_cam
                pt_cam.point.y = Y_cam
                pt_cam.point.z = Z_cam

                try:
                    self.tf_buffer.can_transform(
                        "map",
                        pt_cam.header.frame_id,
                        pt_cam.header.stamp,
                        timeout=Duration(seconds=1.0),
                    )
                    pt_map = self.tf_buffer.transform(
                        pt_cam, "map", timeout=Duration(seconds=1.0)
                    )
                except Exception as e:
                    self.get_logger().warn(f"TF hatası: {e}")
                    return

                x_map = pt_map.point.x
                y_map = pt_map.point.y
                self.get_logger().info(
                    f"State 2: Dünya koordinatında nesne: x={x_map:.3f}, y={y_map:.3f}"
                )

                # İlk hedefi gönder
                self.first_goal_x = x_map
                self.first_goal_y = y_map + 2  # Hedefi biraz ileriye kaydır

                # Robotu durdur
                stop_msg = Twist()
                stop_msg.linear.x = 0.0
                stop_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(stop_msg)

                # Nav2’ye hedef gönder
                goal = PoseStamped()
                goal.header.frame_id = "map"
                goal.header.stamp = self.get_clock().now().to_msg()
                goal.pose.position.x = self.first_goal_x
                goal.pose.position.y = self.first_goal_y
                goal.pose.position.z = 0.0
                q = tf_transformations.quaternion_from_euler(0.0, 0.0, 0.0)
                goal.pose.orientation.x = q[0]
                goal.pose.orientation.y = q[1]
                goal.pose.orientation.z = q[2]
                goal.pose.orientation.w = q[3]

                self.get_logger().info("State 2: Birinci hedef gönderiliyor...")
                self.navigator.goToPose(goal)
                self.get_logger().info("State 2: Nav2 goToPose() çağrıldı.")
                self.state = 3

            except Exception as ex:
                self.get_logger().error(f"timer_callback (state 2) hatası: {ex}")

            finally:
                self.processing = False

        # -------- STATE 3: İlk hedefe yaklaşırken büyük kontur kontrolü --------
        elif self.state == 3:
            self.processing = True
            try:
                # 3.1) Eğer Nav2 hâlâ hedefte değilse
                if not self.navigator.isTaskComplete():
                    # RGB→HSV, mask, contur
                    rgb_image = self.br.imgmsg_to_cv2(self.rgb_msg, "bgr8")
                    hsv_full = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
                    mask_full = cv2.inRange(hsv_full, self.lower_red, self.upper_red)

                    contours, _ = cv2.findContours(
                        mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest)
                        area = w * h

                        # 3.2) Kontur alanı 80000 ≥ ise “yaklaştı”
                        if area >= 80000:
                            self.get_logger().info(
                                f"State 3: Yaklaştı, kontur alanı={area:.0f} ≥ 80000."
                            )
                            # Nesne dünya konumunu yeniden hesapla ve yazdır
                            rgb_h = self.rgb_msg.height
                            rgb_w = self.rgb_msg.width
                            depth_h = self.depth_msg.height
                            depth_w = self.depth_msg.width

                            x_d = int(x * depth_w / rgb_w)
                            y_d = int(y * depth_h / rgb_h)
                            w_d = int(w * depth_w / rgb_w)
                            h_d = int(h * depth_h / rgb_h)

                            pad = 5
                            x0 = max(x_d + pad, 0)
                            y0 = max(y_d + pad, 0)
                            x1 = min(x_d + w_d - pad, depth_w - 1)
                            y1 = min(y_d + h_d - pad, depth_h - 1)

                            depth_image = self.br.imgmsg_to_cv2(self.depth_msg, "32FC1")
                            if x1 <= x0 or y1 <= y0:
                                u_depth = int((x + w / 2) * depth_w / rgb_w)
                                v_depth = int((y + h / 2) * depth_h / rgb_h)
                                z_val = float(depth_image[v_depth, u_depth])
                            else:
                                sub_region = depth_image[y0:y1, x0:x1]
                                valid = sub_region[
                                    np.isfinite(sub_region) & (sub_region > 0.0)
                                ]
                                if valid.size == 0:
                                    u_depth = int((x + w / 2) * depth_w / rgb_w)
                                    v_depth = int((y + h / 2) * depth_h / rgb_h)
                                    z_val = float(depth_image[v_depth, u_depth])
                                else:
                                    z_val = float(np.median(valid))

                            if not (np.isnan(z_val) or z_val <= 0.0):
                                u_rgb = int(x + w / 2)
                                v_rgb = int(y + h / 2)
                                px = np.array([[[u_rgb, v_rgb]]], dtype=np.float32)
                                und = cv2.undistortPoints(
                                    px,
                                    self.camera_matrix,
                                    self.dist_coeffs,
                                    P=self.camera_matrix,
                                )
                                u_undist = und[0][0][0]
                                v_undist = und[0][0][1]

                                u_depth = int(u_undist * depth_w / rgb_w)
                                v_depth = int(v_undist * depth_h / rgb_h)
                                if not (
                                    0 <= u_depth < depth_w and 0 <= v_depth < depth_h
                                ):
                                    u_depth = int((x + w / 2) * depth_w / rgb_w)
                                    v_depth = int((y + h / 2) * depth_h / rgb_h)

                                fx = self.depth_cam_info.k[0]
                                fy = self.depth_cam_info.k[4]
                                cx = self.depth_cam_info.k[2]
                                cy = self.depth_cam_info.k[5]

                                X_cam = (u_depth - cx) * z_val / fx
                                Y_cam = (v_depth - cy) * z_val / fy
                                Z_cam = z_val

                                pt_cam = PointStamped()
                                pt_cam.header.frame_id = "camera_depth_optical_frame"
                                pt_cam.header.stamp = rclpy.time.Time().to_msg()
                                pt_cam.point.x = X_cam
                                pt_cam.point.y = Y_cam
                                pt_cam.point.z = Z_cam

                                try:
                                    self.tf_buffer.can_transform(
                                        "map",
                                        pt_cam.header.frame_id,
                                        pt_cam.header.stamp,
                                        timeout=Duration(seconds=1.0),
                                    )
                                    pt_map_close = self.tf_buffer.transform(
                                        pt_cam, "map", timeout=Duration(seconds=1.0)
                                    )
                                    self.get_logger().info(
                                        f"State 3: Yakın nesne konumu (map): "
                                        f"x={pt_map_close.point.x:.3f}, y={pt_map_close.point.y:.3f}"
                                    )
                                except Exception as e:
                                    self.get_logger().warn(f"State 3 TF hatası: {e}")

                            # 3.3) Nav2 görevini iptal et ve başla dönüş
                            # try:
                            #     self.navigator.cancelTask()
                            # except:
                            #     pass

                            if self.start_pose is not None:
                                self.navigator.goToPose(self.start_pose)
                                self.get_logger().info(
                                    "State 3: Başlangıca dönüş hedefi gönderildi."
                                )
                                self.state = 4
                            else:
                                self.get_logger().warn(
                                    "State 3: start_pose bilinmiyor, dönüş yapılamıyor."
                                )
                                self.state = 5

                            return

                else:
                    # Eğer Nav2 zaten birinci hedefe tamamladıysa, state=4
                    self.state = 4

            except Exception as ex:
                self.get_logger().error(f"timer_callback (state 3) hatası: {ex}")

            finally:
                self.processing = False

        # -------- STATE 4: Başlangıç noktasına dönüş --------
        elif self.state == 4:
            # Nav2’nin dönüşü tamamlamasını bekle
            if self.navigator.isTaskComplete():
                self.get_logger().info(
                    "State 4: Başlangıç noktasına ulaşıldı. Görev tamamlandı."
                )
                self.state = 5

        # -------- STATE 5: Tamamlandı --------
        elif self.state == 5:
            pass

    def send_initial_pose(self, x: float, y: float, z_rot: float):
        """
        AMCL’e başlangıç pozunu bildirir (PoseWithCovarianceStamped).
        """
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, z_rot)
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.frame_id = "map"
        init_msg.header.stamp = self.get_clock().now().to_msg()
        init_msg.pose.pose.position.x = x
        init_msg.pose.pose.position.y = y
        init_msg.pose.pose.position.z = 0.0
        init_msg.pose.pose.orientation.x = q[0]
        init_msg.pose.pose.orientation.y = q[1]
        init_msg.pose.pose.orientation.z = q[2]
        init_msg.pose.pose.orientation.w = q[3]
        init_msg.pose.covariance = [0.0] * 36
        self.initial_pose_pub.publish(init_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RedObjectNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("RedObjectNavNode kapanıyor.")
        rclpy.shutdown()


if __name__ == "__main__":
    main()

