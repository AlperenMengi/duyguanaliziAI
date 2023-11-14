import cv2
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

def deneme():
    print('deneme fonksiyonu içi ------------------- ')
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    model = load_model("gulumseme_tespiti_modeli.h5")
    cap = cv2.VideoCapture(0)
    print('tespit 12. satır: ',cap)

    smileCount = 0  
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            face_image = cv2.resize(roi_gray, (150, 150))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

            face_image = image.img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            face_image /= 255.0

            prediction = model.predict(face_image)

            if prediction[0][0] > 0.5:
                cv2.putText(frame, "smile", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                smileCount+=1
                print(smileCount)
                if(smileCount > 10):
                    cap.release()
                    cv2.destroyAllWindows()   
                    return True
            else:
                cv2.putText(frame, "neutral", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Pencere", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()