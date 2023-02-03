from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier()
face_cascade.load('haarcascade_frontalface_alt.xml')


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    video = cv2.VideoCapture(0)
    while True:
        success, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 3)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            faceROI = frame_gray[y:y + h, x:x + w]
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)