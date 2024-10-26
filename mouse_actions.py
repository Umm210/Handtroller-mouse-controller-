import mediapipe as mp
import pyautogui
import hand_tracking
from pynput.mouse import Button, Controller

mouse = Controller()
screen_width, screen_height = pyautogui.size()

# hands tracking values
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# smoothing factors
smoothing_factor = 0.7
prev_x, prev_y = 0, 0  # initial positions


def smooth_coordinates(x, y):
    # smoothing
    global prev_x, prev_y
    smoothed_x = prev_x + smoothing_factor * (x - prev_x)
    smoothed_y = prev_y + smoothing_factor * (y - prev_y)
    prev_x, prev_y = smoothed_x, smoothed_y
    return int(smoothed_x), int(smoothed_y)


def move_mouse(index_finger_tip):
    # mouse movement happens where the index is
    if index_finger_tip is not None:
        # converting hand coords to screen coords
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)

        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))

        # Smoothing movement before the mouse moves
        smoothed_x, smoothed_y = smooth_coordinates(x, y)
        pyautogui.moveTo(smoothed_x, smoothed_y)


def find_finger_tip(processed):
    # getting the index tip location if detected
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None


def is_left_click(landmark_list, thumb_index_dist):
    return (
            hand_tracking.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 < thumb_index_dist and
            hand_tracking.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
    )


left_click_held = False  # left click hold check


def detect_gesture(frame, landmark_list, processed):
    global left_click_held

    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)  # locate the index tip
        thumb_index_dist = hand_tracking.get_distance([landmark_list[4], landmark_list[5]])  # dist btn index and thumb
        move_mouse(index_finger_tip)

        # left click detection
        if is_left_click(landmark_list, thumb_index_dist):
            if not left_click_held:
                mouse.click(Button.left)
                left_click_held = True  # click is being held
        else:
            if left_click_held:
                mouse.release(Button.left)
                left_click_held = False  # not holding anymore



