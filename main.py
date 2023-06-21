import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer

eye_model = load_model('models/eye-model.h5')
mouth_model = load_model('models/mouth-model.h5')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
eye_r = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
eye_l = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

eye_classes = ['Closed', 'Open']
mouth_classes = ['No_Yawn', 'Yawn']

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

mixer.init()
sound = mixer.Sound('alarm.wav')

score = 0
count = 0
r_pred = ''
l_pred = ''

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(grey, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = eye_l.detectMultiScale(grey)
    right_eye = eye_r.detectMultiScale(grey)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (224, 224))
        r_eye = np.expand_dims(r_eye, axis=0)
        r_probs = eye_model.predict(r_eye)
        r_pred = np.rint(r_probs)
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (224, 224))
        l_eye = np.expand_dims(l_eye, axis=0)
        l_probs = eye_model.predict(l_eye)
        l_pred = np.rint(l_probs)
        break

    if l_pred == 0 and r_pred == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        sound.play()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        sound.stop()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
