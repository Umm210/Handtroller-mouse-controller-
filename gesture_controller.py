import cv2
import mediapipe as mp
import mouse_actions

controller_running = False
cap = None


def main():
    global cap, controller_running
    controller_running = True
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # camera frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = mouse_actions.hands.process(frameRGB)

            # list of hand landmarks
            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # only detecting one hand
                draw.draw_landmarks(frame, hand_landmarks, mouse_actions.mpHands.HAND_CONNECTIONS)  # draw landmarks
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            mouse_actions.detect_gesture(frame, landmark_list, processed)

            # Display the frame in a window
            cv2.imshow('Hand Controller', frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
