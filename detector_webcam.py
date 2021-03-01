from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('model2-020.model')
categories = ["anger", "fear", "happy", "sadness", "surprise"]
IMG_SIZE = 48

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))
        result = model.predict(reshaped)

        print(result)
        accuracy = "{:.2f}".format(np.amax(result)*100)
        #accuracy = "%0.2f" % 3

        label = np.argmax(result, axis=1)[0]
        print(accuracy)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w + 50, y), (255, 0, 0), -1)
        cv2.putText(frame, categories[label] + " " + accuracy + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()