import cv2
import numpy as np
import csv
import mediapipe as mp

# ---- Initialize MediaPipe Pose ----
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ---- NET HEIGHT CALIBRATION ----
NET_HEIGHT_in = 88 # womens net
net_top_y = None
net_bottom_y = None
pixels_per_in = None
calibration_step = 0  # 0 = click net, 1 = click ground, 2 = done

def mouse_callback(event, x, y, flags, param):
    global net_top_y, net_bottom_y, pixels_per_in, calibration_step
    if event == cv2.EVENT_LBUTTONDOWN:
        if calibration_step == 0:
            net_top_y = y
            calibration_step = 1
            print(f"✓ Net top set at y={y}. Now click on the GROUND.")
        elif calibration_step == 1:
            net_bottom_y = y
            net_height_pixels = abs(net_bottom_y - net_top_y)
            pixels_per_in = net_height_pixels / NET_HEIGHT_in
            calibration_step = 2
            print(f"✓ Calibration complete! {pixels_per_in:.2f} pixels per in")

# ---- Open video file ----
video_path = "volleyball.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video file!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Default if can't detect
print(f"Video FPS: {fps}")

cv2.namedWindow("Volleyball Jump Detection")
cv2.setMouseCallback("Volleyball Jump Detection", mouse_callback)

print("\n" + "="*60)
print("STEP 1: CALIBRATION")
print("="*60)
print("Click on the TOP of the net, then click on the GROUND")
print("\nCONTROLS:")
print("  SPACEBAR = Pause/Unpause")
print("  R = Reset tracking")
print("  ESC = Exit")
print("="*60 + "\n")

# ---- CSV for jump heights ----
csv_file = open("jump_positions.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "jump_height_in"])

# ---- Variables ----
frame_count = 0
jump_threshold = 0.015  # Lowered back down
prev_ankle_y = None
baseline_ankle_y = None  # Track standing position
jump_heights = []
jump_display_counter = 0
last_jump_height = 0
total_jumps = 0

# Smoothing variables
ankle_history = []
history_size = 5  # Back to 5 for more responsiveness

# Jump detection improvements
min_jump_height = 10  # Lowered minimum
max_jump_height = 120  # Realistic maximum
frames_since_last_jump = 0
min_frames_between_jumps = 10  # Reduced cooldown

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frames_since_last_jump += 1
    frame_height, frame_width = frame.shape[:2]
    
    # Draw calibration points
    if net_top_y is not None:
        cv2.circle(frame, (frame_width//2, net_top_y), 4, (0, 255, 0), -1)
        cv2.putText(frame, "Net", (frame_width//2 + 8, net_top_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    if net_bottom_y is not None:
        cv2.circle(frame, (frame_width//2, net_bottom_y), 4, (0, 255, 0), -1)
        cv2.putText(frame, "Ground", (frame_width//2 + 8, net_bottom_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)

    jump_height_in = 0
    
    # Status box background - MOVED TO BOTTOM LEFT
    status_box_y = frame_height - 130
    cv2.rectangle(frame, (10, status_box_y), (310, frame_height - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, status_box_y), (310, frame_height - 10), (255, 255, 255), 1)

    # Calibration status
    if calibration_step == 0:
        cv2.putText(frame, "STEP 1: Click NET TOP", (15, status_box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    elif calibration_step == 1:
        cv2.putText(frame, "STEP 2: Click GROUND", (15, status_box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    else:
        cv2.putText(frame, f"Calibrated: {pixels_per_in:.2f} px/in", (15, status_box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # Pose detection status
    if results.pose_landmarks:
        cv2.putText(frame, "Pose: DETECTED", (15, status_box_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
        ankle_y = (left_ankle + right_ankle) / 2
        
        # Add to history for smoothing
        ankle_history.append(ankle_y)
        if len(ankle_history) > history_size:
            ankle_history.pop(0)
        
        # Use smoothed ankle position
        smoothed_ankle_y = sum(ankle_history) / len(ankle_history)
        
        # Initialize baseline
        if baseline_ankle_y is None:
            baseline_ankle_y = smoothed_ankle_y
        else:
            # Update baseline slowly when person is standing still
            if abs(smoothed_ankle_y - baseline_ankle_y) < 0.005:  # Tighter threshold
                baseline_ankle_y = baseline_ankle_y * 0.98 + smoothed_ankle_y * 0.02  # Slower update

        cv2.putText(frame, f"Ankle: {smoothed_ankle_y:.3f}", (15, status_box_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"Baseline: {baseline_ankle_y:.3f}", (15, status_box_y + 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if prev_ankle_y is not None:
            ankle_diff = prev_ankle_y - smoothed_ankle_y
            cv2.putText(frame, f"Movement: {ankle_diff:.4f}", (15, status_box_y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Show why jump isn't detected (if movement is significant)
            if abs(ankle_diff) > 0.01:
                potential_jump_in = ankle_diff * frame_height * pixels_per_in if pixels_per_in else 0
                
                debug_msg = ""
                if calibration_step != 2:
                    debug_msg = "Not calibrated"
                elif ankle_diff <= jump_threshold:
                    debug_msg = f"Below threshold ({ankle_diff:.4f} < {jump_threshold})"
                elif jump_display_counter > 0:
                    debug_msg = "Already showing jump"
                elif frames_since_last_jump <= min_frames_between_jumps:
                    debug_msg = f"Cooldown ({frames_since_last_jump}/{min_frames_between_jumps})"
                elif potential_jump_in < min_jump_height:
                    debug_msg = f"Too small ({potential_jump_in:.1f}in < {min_jump_height}in)"
                elif potential_jump_in > max_jump_height:
                    debug_msg = f"Too large ({potential_jump_in:.1f}in > {max_jump_height}in)"
                else:
                    debug_msg = f"Would be: {potential_jump_in:.1f}in - SHOULD DETECT!"
                
                cv2.putText(frame, debug_msg, (15, status_box_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            
            # Detect jump - improved conditions
            if (calibration_step == 2 and 
                ankle_diff > jump_threshold and 
                jump_display_counter == 0 and
                frames_since_last_jump > min_frames_between_jumps):  # Cooldown period
                
                jump_height_normalized = ankle_diff
                jump_height_in = jump_height_normalized * frame_height * pixels_per_in
                
                # Only count realistic jumps with stricter filtering
                if min_jump_height <= jump_height_in <= max_jump_height:
                    last_jump_height = jump_height_in
                    total_jumps += 1
                    frames_since_last_jump = 0  # Reset cooldown
                    
                    print(f"Frame {frame_count}: JUMP #{total_jumps} - {jump_height_in:.1f} in")
                    csv_writer.writerow([frame_count, jump_height_in])
                    jump_display_counter = int(fps * 3)  # Display for 3 seconds
                else:
                    print(f"Rejected jump: {jump_height_in:.1f}in (outside {min_jump_height}-{max_jump_height}in range)")

        prev_ankle_y = smoothed_ankle_y
        jump_heights.append(jump_height_in)
        if len(jump_heights) > 200:
            jump_heights.pop(0)
    else:
        cv2.putText(frame, "Pose: NOT DETECTED", (15, status_box_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(frame, "Move closer or adjust", (15, status_box_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # Stats
    cv2.putText(frame, f"Jumps: {total_jumps}", (15, status_box_y + 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Display jump detection - cleaner overlay
    if jump_display_counter > 0:
        # Draw semi-transparent box
        overlay = frame.copy()
        cv2.rectangle(overlay, (30, 140), (250, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text
        cv2.putText(frame, "JUMP DETECTED!", (40, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Height: {last_jump_height:.1f} in", (40, 188),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        jump_display_counter -= 1

    # ---- Draw jump height graph ----
    graph_height = 100
    graph_width = 200
    graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

    if len(jump_heights) > 1 and max(jump_heights) > 0:
        max_jump = max(max(jump_heights), 50)  # At least 50in scale
        for i in range(1, len(jump_heights)):
            y1 = graph_height - int((jump_heights[i-1] / max_jump) * graph_height * 0.9)
            y2 = graph_height - int((jump_heights[i] / max_jump) * graph_height * 0.9)
            y1 = max(0, min(graph_height-1, y1))
            y2 = max(0, min(graph_height-1, y2))
            cv2.line(graph, (i-1, y1), (i, y2), (0, 255, 0), 2)

    # Overlay graph
    x_offset = frame_width - graph_width - 20
    y_offset = frame_height - graph_height - 20
    frame[y_offset:y_offset+graph_height, x_offset:x_offset+graph_width] = graph

    # Max jump stat
    if len(jump_heights) > 0 and max(jump_heights) > 0:
        cv2.putText(frame, f"Max: {max(jump_heights):.1f}in", (x_offset, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Controls reminder
    cv2.putText(frame, "SPACE=Pause | R=Reset | ESC=Exit", (frame_width - 280, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Show video
    cv2.imshow("Volleyball Jump Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACEBAR
        print("⏸ PAUSED - Press any key to continue...")
        cv2.waitKey(0)
    elif key == ord('r') or key == ord('R'):  # R = Reset
        print("Tracking reset")
        prev_ankle_y = None
        baseline_ankle_y = None
        ankle_history = []
        frames_since_last_jump = 0

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("PROCESSING COMPLETE!")
print("="*60)
print(f"Total frames processed: {frame_count}")
print(f"Total jumps detected: {total_jumps}")
if len(jump_heights) > 0 and max(jump_heights) > 0:
    print(f"Maximum jump height: {max(jump_heights):.1f} in")
print(f"Results saved to: jump_positions.csv")
print("="*60)
