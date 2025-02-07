import cv2
from flask import Flask, render_template, request, Response
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import whisper
from dummy2 import convert_text_to_isl
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string

# Load models
whisper_model = whisper.load_model("base")
isl_model = keras.models.load_model("Indian-Sign-Language-Detection/model.h5")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

def sign_output():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)

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
            temp_landmark_list[index][0] -= base_x
            temp_landmark_list[index][1] -= base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(map(abs, temp_landmark_list))
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
        return temp_landmark_list

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    df = pd.DataFrame(pre_processed_landmark_list).transpose()
                    predictions = isl_model.predict(df, verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)
                    label = alphabet[predicted_classes[0]]
                    cv2.putText(image, label, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 255), 7)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/part1', methods=['GET', 'POST'])
def part1():
    transcribed_text = None
    if request.method == 'POST':
        input_text = request.form.get('giventext')
        if input_text:
            print(f"User input: {input_text}")
            convert_text_to_isl(input_text)
        if 'fileInput' in request.files:
            file = request.files['fileInput']
            if file and file.filename != '':
                filename = 'audio.wav'
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                print(f"File uploaded and saved to {file_path}")
                result = whisper_model.transcribe(file_path)
                print("Transcribed Text:", result["text"])
                transcribed_text = result["text"]
                convert_text_to_isl(result["text"])
    return render_template('audio_to_isl.html', transcribed_text=transcribed_text)

@app.route('/part2')
def part2():
    return render_template('isl_to_text.html')

@app.route('/video_feed')
def video_feed():
    return Response(sign_output(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
