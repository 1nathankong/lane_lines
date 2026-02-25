import cv2
import numpy as np
import time
from collections import deque, namedtuple

LaneState = namedtuple("LaneState", ["coeffs", "type", "color"])

class LaneTracker:
    """Tracks a single lane (left or right) owns all per-lane buffers and state."""

    def __init__(self, buffer_size, alpha, line_threshold):
        self.alpha = alpha
        self.line_threshold = line_threshold
        self.buffer = deque(maxlen=buffer_size)
        self.type_history = deque(maxlen=buffer_size)
        self.color_history = deque(maxlen=buffer_size)
        self.prev = None  # LaneState | None

    def _fit_curve(self, points):
        """Adaptive fit: quadratic only when curvature is significant, otherwise linear."""
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        quad = np.polyfit(ys, xs, 2)
        curve_effect = abs(quad[0]) * (ys.max() - ys.min()) ** 2
        if curve_effect > 50:
            return quad

        lin = np.polyfit(ys, xs, 1)
        return np.array([0.0, lin[0], lin[1]])

    def _classify_type(self, segments):
        """ Classify using coverage ratio. Broken lines follow a 1:3 standard , solid lines are near 100%. Threshold at 50%. """
        if len(segments) < 2:
            self.type_history.append(0.0)
        else:
            # this is broken not able to indentify broken lines in the middle of the freeway. 
            segs = np.array(segments)
            y_min = segs[:, [1, 3]].min()
            y_max = segs[:, [1, 3]].max()
            span = y_max - y_min
            covered = sum(abs(s[3] - s[1]) for s in segments)
            self.type_history.append(covered / span if span > 0 else 1.0)

        return "SOLID" if np.mean(self.type_history) >= self.line_threshold else "DOTTED"

    def _classify_color(self, segments, mask_y, width, height):
        """Detect lane color by sampling a region around each segment midpoint, smoothed with a temporal vote across recent frames."""
        radius = 5
        if len(segments) > 0:
            yellow_px, total_px = 0, 0
            for seg in segments:
                mid_x = np.clip(int((seg[0] + seg[2]) / 2), radius, width - 1 - radius)
                mid_y = np.clip(int((seg[1] + seg[3]) / 2), radius, height - 1 - radius)
                patch = mask_y[mid_y - radius:mid_y + radius + 1, mid_x - radius:mid_x + radius + 1]
                yellow_px += np.count_nonzero(patch)
                total_px += patch.size
            self.color_history.append(yellow_px / total_px if total_px > 0 else 0.0)

        if len(self.color_history) == 0:
            return "WHITE"
        return "YELLOW" if np.mean(self.color_history) > 0.15 else "WHITE"

    def update(self, points, segments, mask_y, width, height):
        """Run one frame of tracking. Returns LaneState or None if not enough data."""
        self.buffer.append(points)
        all_points = [p for frame_pts in self.buffer for p in frame_pts]

        if len(all_points) < 6:
            return self.prev  # may be None

        coeffs = self._fit_curve(all_points)
        line_type = self._classify_type(segments)
        color = self._classify_color(segments, mask_y, width, height)

        # EMA smooth against previous frame
        if self.prev is not None:
            coeffs = self.alpha * coeffs + (1 - self.alpha) * self.prev.coeffs

        self.prev = LaneState(coeffs, line_type, color)
        return self.prev


class LaneDetector:

    def __init__(self, buffer_size=5, alpha=0.2, debug=False, line_threshold=0.5):
        self.debug = debug
        self.roi_vertices = None
        self.gpu_roi_mask = None

        # HSV ranges
        self.lower_yellow = np.array([15, 38, 115])
        self.upper_yellow = np.array([35, 204, 255])
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])

        # GPU objects
        self.gpu_clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
        self.gpu_canny = cv2.cuda.createCannyEdgeDetector(30, 100)

        # Per-lane trackers
        self.left = LaneTracker(buffer_size, alpha, line_threshold)
        self.right = LaneTracker(buffer_size, alpha, line_threshold)

    def _apply_roi(self, gpu_edges):
        """Applies a trapezoidal mask on the GPU."""
        height, width = gpu_edges.size()[::-1]
        if self.gpu_roi_mask is None:
            self.roi_vertices = np.array([
                [int(width * 0.1), height],
                [int(width * 0.40), int(height * 0.667)],
                [int(width * 0.58), int(height * 0.667)],
                [int(width * 0.9), height]
            ], dtype=np.int32)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [self.roi_vertices], 255)
            self.gpu_roi_mask = cv2.cuda_GpuMat()
            self.gpu_roi_mask.upload(mask)

        return cv2.cuda.bitwise_and(gpu_edges, self.gpu_roi_mask)

    def _detect_edges(self, frame, color_mask):
        """Color-mask the frame, then run GPU edge detection + ROI crop."""
        masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)

        gpu_masked = cv2.cuda_GpuMat()
        gpu_masked.upload(masked_frame)
        gpu_gray = cv2.cuda.cvtColor(gpu_masked, cv2.COLOR_BGR2GRAY)

        gpu_enhanced = self.gpu_clahe.apply(gpu_gray, cv2.cuda_Stream.Null())
        gpu_blurred = self.gpu_blur.apply(gpu_enhanced)
        gpu_edges = self.gpu_canny.detect(gpu_blurred)
        return self._apply_roi(gpu_edges)

    def _sort_lines(self, lines, mid_x):
        """Split Hough lines into left/right points and segments by slope + position."""
        left_pts, right_pts = [], []
        left_segs, right_segs = [], []

        if lines is None:
            return left_pts, left_segs, right_pts, right_segs

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue
            m = (y2 - y1) / (x2 - x1)
            mx = (x1 + x2) // 2

            if -1.5 < m < -0.5 and mx < mid_x:
                left_pts.extend([(x1, y1), (x2, y2)])
                left_segs.append([x1, y1, x2, y2])
            elif 0.5 < m < 1.50 and mx > mid_x:
                right_pts.extend([(x1, y1), (x2, y2)])
                right_segs.append([x1, y1, x2, y2])

        return left_pts, left_segs, right_pts, right_segs

    def _draw_overlay(self, frame, lane_results, plot_y, width):
        """Draw lane curves, fill the lane zone, and add HUD text."""
        overlay = frame.copy()
        curve_pts = {}

        for side, state in lane_results.items():
            plot_x = np.clip(np.polyval(state.coeffs, plot_y).astype(int), 0, width - 1)
            pts = np.column_stack((plot_x, plot_y)).reshape((-1, 1, 2))
            curve_pts[side] = np.column_stack((plot_x, plot_y))

            line_color = (0, 255, 255) if state.color == "YELLOW" else (255, 255, 255)
            cv2.polylines(overlay, [pts], False, line_color, 5)

        if "Left" in curve_pts and "Right" in curve_pts:
            lane_poly = np.vstack([curve_pts["Left"], curve_pts["Right"][::-1]]).astype(np.int32)
            lane_overlay = overlay.copy()
            cv2.fillPoly(lane_overlay, [lane_poly], (0, 180, 0))
            cv2.addWeighted(lane_overlay, 0.3, overlay, 0.7, 0, overlay)

        y_offset = 60
        for side, state in lane_results.items():
            label = f"{side}: {state.color} {state.type}"
            cv2.putText(overlay, label, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        return overlay

    def process_frame(self, frame):
        height, width = frame.shape[:2]

        # Color filtering (CPU) — bottom third only, sky excluded
        sky_cutoff = int(height * 2 / 3)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_w = cv2.inRange(hsv, self.lower_white, self.upper_white)
        color_mask = cv2.bitwise_or(mask_y, mask_w)
        color_mask[:sky_cutoff, :] = 0
        mask_y[:sky_cutoff, :] = 0

        # Edge detection (GPU)
        gpu_roi_edges = self._detect_edges(frame, color_mask)

        # Debug images
        debug_imgs = {}
        if self.debug:
            roi_visual = frame.copy()
            cv2.polylines(roi_visual, [self.roi_vertices], True, (255, 0, 0), 4)
            roi_overlay = roi_visual.copy()
            cv2.fillPoly(roi_overlay, [self.roi_vertices], (0, 255, 255))
            cv2.addWeighted(roi_overlay, 0.3, roi_visual, 0.7, 0, roi_visual)
            debug_imgs["DEBUG 1 - ROI Overlay"] = roi_visual
            #debug_imgs["DEBUG 2 - White+Yellow Mask"] = color_mask
            debug_imgs["DEBUG 3 - Masked Color"] = cv2.bitwise_and(frame, frame, mask=color_mask)
            debug_imgs["DEBUG 4 - Canny Edges (ROI only)"] = gpu_roi_edges.download()

        # Hough lines + sort
        edges = gpu_roi_edges.download()
        lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 50, minLineLength=30, maxLineGap=30)
        left_pts, left_segs, right_pts, right_segs = self._sort_lines(lines, width // 2)

        # Update each lane tracker
        lane_results = {}
        for side, tracker, pts, segs in [
            ("Left", self.left, left_pts, left_segs),
            ("Right", self.right, right_pts, right_segs),
        ]:
            state = tracker.update(pts, segs, mask_y, width, height)
            if state is not None:
                lane_results[side] = state

        # Draw
        plot_y = np.linspace(int(height * 0.667), height, 50, dtype=int)
        overlay = self._draw_overlay(frame, lane_results, plot_y, width)

        return overlay, debug_imgs


## down here to test my code
if __name__ == "__main__":
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("CUDA not detected."); exit()

    # ── Toggle debug mode here ──
    DEBUG = True

    detector = LaneDetector(debug=DEBUG)
    #cap = cv2.VideoCapture(r"c:\Users\nkgMe\documents\lane_lines\project_video.mp4")
    #cap = cv2.VideoCapture(r"c:\Users\nkgMe\documents\lane_lines\challenge_video.mp4")
    cap = cv2.VideoCapture(r"c:\Users\nkgMe\documents\lane_lines\harder_challenge_video.mp4")

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if not ret: break
        res, debug_imgs = detector.process_frame(frame)
        fps = 1 / (time.time() - start)
        cv2.putText(res, f"GPU FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Car Logic Prototype", res)

        for title, img in debug_imgs.items():
            cv2.imshow(title, img)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()
