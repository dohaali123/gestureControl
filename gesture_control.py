import cv2
import mediapipe as mp
import pyautogui  # For scrolling actions

# Initialize Mediapipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks for reference
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

    # Show the frame with annotations
    cv2.imshow("Thumbs Up/Down Gesture Control", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
