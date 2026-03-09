import io
import cv2
from flask import Flask, Response

app=Flask(__name__)

def generate_frames():
    #with picamera.PiCamera() as camera:
        #camera.resolution=(640,480)
        #camera.framerate=24
        #stream=io.BytesIO()
        cap=cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_HEOGHT, 480)
        #cap.set(cv2.CAP_PROP_FPS, 24)
        
        try:
            while True:
                ret,frame=cap.read()
                if not ret:
                    break
                
                ok,buffer=cv2.imencode(".jpg",frame)
                if not ok:
                    continue
                
                frame_bytes=buffer.tobytes()
                
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')
        finally:
            cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
