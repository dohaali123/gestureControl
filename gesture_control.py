import cv2
import mediapipe as mp
import pyautogui  # For scrolling actions
import time


# Initialize Mediapipe Hand and Face Mesh solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variables for zoom timing
last_zoom_time = 0  # Last zoom action time

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with both MediaPipe solutions
    results_hands = hands.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)

    # ---- Hand Gesture: Thumbs Up/Down for Scrolling ----
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get specific hand landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Check for thumbs up (thumb above index MCP and pinky)
            if thumb_tip.y < index_mcp.y and thumb_tip.y < pinky_tip.y:
                cv2.putText(frame, "Thumbs Up - Scroll Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.scroll(10)  # Scroll up
            # Check for thumbs down (thumb below index MCP and pinky)
            elif thumb_tip.y > index_mcp.y and thumb_tip.y > pinky_tip.y:
                cv2.putText(frame, "Thumbs Down - Scroll Down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pyautogui.scroll(-10)  # Scroll down

    # ---- Face Gesture: Eyebrow Movement for Zooming ----
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # Get coordinates of key landmarks
            h, w, _ = frame.shape

            # Right eyebrow and eye landmarks
            right_eyebrow_top = face_landmarks.landmark[65]  # Right eyebrow top
            right_eye_top = face_landmarks.landmark[159]  # Right eye top

            # Convert normalized coordinates to pixel values
            right_eyebrow_top_y = int(right_eyebrow_top.y * h)
            right_eye_top_y = int(right_eye_top.y * h)

            # Measure the distance between eyebrow and eye
            eyebrow_eye_distance = right_eye_top_y - right_eyebrow_top_y

            # Debug: Display distance
            cv2.putText(frame, f"Eyebrow Distance: {eyebrow_eye_distance}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Thresholds for zooming
            if eyebrow_eye_distance > 25 and (time.time() - last_zoom_time > 0.5):  # Zoom Out
                cv2.putText(frame, "Zoom Out", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pyautogui.hotkey('ctrl', '-')  # Zoom out
                last_zoom_time = time.time()
            elif eyebrow_eye_distance < 10 and (time.time() - last_zoom_time > 0.5):  # Zoom In
                cv2.putText(frame, "Zoom In", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.hotkey('ctrl', '+')  # Zoom in
                last_zoom_time = time.time()

    # Show the frame with annotations
    cv2.imshow("Gesture Control: Scroll and Zoom", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
