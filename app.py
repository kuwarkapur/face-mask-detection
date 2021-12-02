import tensorflow as tf
from flask import Flask,Response
model=tf.keras.models.load_model('face_mask.h5',compile=False)

app=Flask(__name__)

import cv2 
@app.route('/')
def index():
   return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def frames():
    b=1
    haar = cv2.CascadeClassifier('D:\haarcascade_frontalface_default.xml')
    dims=224
    offset = 5
    frame=cv2.VideoCapture(0)
    while(frame):
        ret,img=frame.read()
        faces = haar.detectMultiScale(img, 1.3, 5)
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            face_offset = img[y - offset:y + h + offset, x - offset:x + w + offset]
            face_selection = cv2.resize(face_offset, (500, 500))
            # Create resized image using the calculated dimentions
            resized_image = cv2.resize(face_selection,(dims,dims),interpolation=cv2.INTER_AREA)
            resized_image=tf.expand_dims(resized_image,axis=0)
            c=model.predict(resized_image)
            b=tf.round(c)
        if b==0:
            cv2.putText(img,"with_mask",(100,100),4,1,250,4)
        else:
            cv2.putText(img,"without_mask",(100,100),4,1,250,4)
    
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')



if '__main__'==__name__:
    app.run(debug=True)