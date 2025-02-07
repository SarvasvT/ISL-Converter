# isl_detection_module.py

import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string

def detect_sign_language():
    # Load the saved model from file
    model = keras.models.load_model("model.h5")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    alphabet =  ['1','2','3','4','5','6','7','8','9']
    alphabet += list(string.ascii_uppercase)

    def calc_landmark_list(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def pre_process_landmark(landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))
        def normalize_(n):
            return n / max_value
        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list

    cap = cv2.VideoCapture(0)
    results_list = []

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            if not success:
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            debug_image = copy.deepcopy(image)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    df = pd.DataFrame(pre_processed_landmark_list).transpose()
                    predictions = model.predict(df, verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)
                    label = alphabet[predicted_classes[0]]
                    results_list.append(label)
                    if len(results_list) >= 1:  # Example condition to break after processing one frame
                        cap.release()
                        cv2.destroyAllWindows()
                        return results_list

    cap.release()
    cv2.destroyAllWindows()
    return results_list
