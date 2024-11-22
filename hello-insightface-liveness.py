"""
# 人脸活体检测
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class LivenessDetector:
    def __init__(self, blink_threshold=0.2, flow_threshold=1.0, consecutive_frames=3):
        """
        活体检测器初始化。
        - blink_threshold: EAR 阈值，用于眨眼检测。
        - flow_threshold: 光流检测的运动幅度阈值。
        - consecutive_frames: 连续帧 EAR 阈值的最小帧数。
        """
        self.blink_threshold = blink_threshold
        self.flow_threshold = flow_threshold
        self.consecutive_frames = consecutive_frames
        self.blink_count = 0
        self.frame_counter = 0
        self.prev_gray = None
        self.face_analyzer = self.initialize_insightface()
        self.is_live = False

    def initialize_insightface(self):
        """
        初始化 InsightFace 人脸分析模块。
        """
        app = FaceAnalysis(root='./', allowed_modules=None, providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app

    def calculate_eye_aspect_ratio(self, eye_points):
        """
        计算眼睛纵横比（EAR）。
        - eye_points: 眼部关键点坐标数组。
        """
        print(eye_points[1])
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[1] - eye_points[2])
        return (A + B) / (2.0 * C)

    def detect_blink(self, eye_aspect_ratio):
        """
        检测眨眼行为。
        - eye_aspect_ratio: 当前帧的眼睛纵横比。
        """
        if eye_aspect_ratio < self.blink_threshold:
            # 输出eye_aspect_ratio和blink_threshold和blink_count和字符串“眨眼”
            print(eye_aspect_ratio, self.blink_threshold, self.blink_count, "眨眼")
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consecutive_frames:
                self.blink_count += 1
            self.frame_counter = 0

        if self.blink_count >= 2:
            print("连续眨眼次数超过阈值，判定为活体")
        return self.blink_count

    def detect_liveness_optical_flow(self, prev_gray, current_gray, roi):
        """
        检测人脸区域的光流运动幅度。
        - prev_gray: 前一帧灰度图。
        - current_gray: 当前帧灰度图。
        - roi: 人脸区域 (x, y, w, h)。
        """
        x, y, w, h = roi
        h, w = min(h, prev_gray.shape[0] - y), min(w, prev_gray.shape[1] - x)  # 边界检查
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_roi = flow[y:y + h, x:x + w]
        magnitude, _ = cv2.cartToPolar(flow_roi[..., 0], flow_roi[..., 1])
        return np.mean(magnitude) > self.flow_threshold

    def process_frame(self, frame):
        """
        处理单帧图像，执行人脸检测和活体检测。
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_analyzer.get(frame_rgb)

        for face in faces:
            # 获取人脸边界框和关键点
            x1, y1, x2, y2 = map(int, face.bbox)
            w, h = x2 - x1, y2 - y1
            roi = (x1, y1, w, h)

            kps = face.kps.astype(np.int32)  # 人脸关键点
            kps68 = face.landmark_3d_68.astype(np.int32)  # 人脸68个关键点
            left_eye = kps68[36:42, :]  # 左眼
            right_eye = kps68[42:48, :]  # 右眼

            # 眨眼检测
            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            print(ear)
            blink_count = self.detect_blink(ear)

            # 光流检测
            is_live_optical_flow = False
            if self.prev_gray is not None:
                is_live_optical_flow = self.detect_liveness_optical_flow(self.prev_gray, gray, roi)

            print("流光检测 ：", is_live_optical_flow)

            # 综合判断
            is_live = (blink_count > 0) and is_live_optical_flow
            self.is_live = is_live

            label = "Live" if self.is_live else "Fake"
            color = (0, 255, 0) if is_live else (0, 0, 255)

            # 绘制结果
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        self.prev_gray = gray
        return frame


def main():
    """
    主函数，启动摄像头进行实时检测。
    """
    detector = LivenessDetector(blink_threshold=0.8)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.process_frame(frame)
        cv2.imshow("Liveness Detection", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
