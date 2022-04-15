#app.py
from flask import Flask, render_template, Response, request,flash,redirect,url_for
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread


from werkzeug.utils import secure_filename

global capture,rec_frame, rec, out

capture=0
rec=0
#TEMPLATE_DIR = os.path.abspath('../templates')
#STATIC_DIR = os.path.abspath('../static')
app = Flask(__name__, template_folder='templates')

'''
for ip camera use - 
rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
'''



camera = cv2.VideoCapture(0)
 
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config["CACHE_TYPE"] = "null"

ALLOWED_EXTENSIONS = set(['mp3','wav'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No audio selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Audio file successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - mp3/wav')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_audio(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')



if __name__ == "__main__":
    app.run()
#