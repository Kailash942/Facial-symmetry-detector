'''
Facial Symmetry Detection
Author: Kailash Dalui (M.Tech Robotics & AI)

This project detects up to 2 faces using Mediapipe and calculates
a simple facial symmetry score in real-time using the webcam.
'''

# Import required libraries
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe FaceMesh for face landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,     # Use video stream (not just images)
    max_num_faces=2,             # Detect up to 2 faces
    refine_landmarks=True,       # More accurate landmarks (eyes, lips, etc.)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no frame is captured

    # Convert the frame from BGR (OpenCV) to RGB (Mediapipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Check if at least one face is detected
    if results.multi_face_landmarks:
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks, start=1):
            h, w, _ = frame.shape

            # Convert Mediapipe landmarks to pixel coordinates
            pts = np.array([
                (int(l.x * w), int(l.y * h)) 
                for l in face_landmarks.landmark
            ])

            # Draw small green dots on all landmarks
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Define pairs of left-right points to compare symmetry
            symmetry_pairs = [
                (33, 263),   # eye corners
                (133, 362),  # inner eyes
                (61, 291),   # mouth corners
                (199, 429),  # cheeks
                (2, 152)     # chin
            ]

            diffs = []  # store differences
            for (l, r) in symmetry_pairs:
                lx, ly = pts[l]
                rx, ry = pts[r]

                # Mirror the right landmark across the nose center
                nose_x = pts[1][0]   # Nose tip (x position)
                mirrored_rx = 2 * nose_x - rx

                # Distance between left and mirrored-right point
                diff = np.linalg.norm([lx - mirrored_rx, ly - ry])
                diffs.append(diff)

                # Draw blue line between left and right landmark
                cv2.line(frame, (lx, ly), (rx, ry), (255, 0, 0), 1)

            # Calculate symmetry score
            score = max(0, 100 - np.mean(diffs))

            # Display symmetry score for each face separately
            y_offset = 50 * face_id   # avoid overlapping text
            cv2.putText(frame, f"Face {face_id} Symmetry: {score:.2f}%",
                        (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

    # Show the video with landmarks and scores
    cv2.imshow("Facial Symmetry Detector", frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
