import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start the webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

with mp_hands.Hands(
    model_complexity=1,  # Higher value for better accuracy but slower performance
    max_num_hands=2,     # Detect up to two hands
    min_detection_confidence=0.5,  # Minimum confidence for detection
    min_tracking_confidence=0.5    # Minimum confidence for tracking
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the image horizontally for a mirror-like effect
        # Convert the frame from BGR (OpenCV) to RGB (MediaPipe)
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(frame_rgb)

        # Draw the hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Display the resulting frame
        cv2.imshow('Hand Detection', frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
