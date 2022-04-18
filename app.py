#!/usr/bin/env python
import imghdr
from importlib import import_module
import os
import cv2
import js2py
play = """
        var audio= document.getElementById('audio');
        var playPauseBTN = document.getElementById('playPauseBTN');
        var count=0;
        function playPause(){
            if(count == 0){
                count = 1;
                audio.play();
                playPauseBTN.innerHTML = "Pause &#9208;"
            }else{
                count = 0;
                audio.pause();
                playPauseBTN.innerHTML = "Play &#9658;"
                }
            }
            """

from flask import Flask, render_template, Response, redirect, request

from werkzeug.utils import secure_filename

import soundfile as sf
import sounddevice as sd

import numpy as np
from pydub import AudioSegment
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# Get current volume 
currentVolumeDb = volume.GetMasterVolumeLevel()

filename = 'audio'
path = 'static/uploads/' + filename + '.wav' # path from my computer 
audio_file, fs = sf.read(path, dtype='float32')  

def Mute():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    setup_mute =-65.25
    volume.SetMasterVolumeLevel(setup_mute, None)


def play(audio_file):
    sd.play(audio_file)

def stop():
    sd.stop()

def volume_20():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    setup_20 = -23.654823303222656
    volume.SetMasterVolumeLevel(setup_20, None)    

def volume_40():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    setup_40 = -13.61940860748291
    volume.SetMasterVolumeLevel(setup_40, None)

def volume_60():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    setup_60 = -7.626824855804443
    volume.SetMasterVolumeLevel(setup_60, None)

def volume_80():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    setup_80 = -3.339998245239258
    volume.SetMasterVolumeLevel(setup_80, None)


def volume_max():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    setup_max = 0.0
    volume.SetMasterVolumeLevel(setup_max, None)    

from py_mini_racer import py_mini_racer
ctx = py_mini_racer.MiniRacer()
# ctx.eval("""
# var audio= document.getElementById('audio');
# var playPauseBTN = document.getElementById('playPauseBTN');
# var count=0;



# function playPause(){
# 	if(count == 0){
# 		count = 1;
# 		audio.play();
#         playPauseBTN.innerHTML = "Pause &#9208;"
#     }else{
#         count = 0;
# 		audio.pause();
#         playPauseBTN.innerHTML = "Play &#9658;"
#     }
# }

# function Stop(){
#     playPause()
#     audio.pause();
#     audio.currentTime= 0;
#     playPauseBTN.innerHTML = "Play &#9658;"

# }

# function Volume_20(){
#     audio.volume = 0.2;
# }
# function Volume_40(){
#     audio.volume = 0.4;
# }
# function Volume_60(){
#     audio.volume = 0.6;
# }

# function Volume_80(){
#     audio.volume = 0.8;
# }

# function Volume_max(){
#     audio.volume = 1;
# }
# function Mute(){
#     audio.volume = 0;
# }


# """)

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config["CACHE_TYPE"] = "null"

ALLOWED_EXTENSIONS = set(['mp3','wav']) 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        # flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        # flash('No audio selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # flash('Audio file successfully uploaded and displayed below')
        # subprocess.run(['python', 'detect.py', '--source', '0','--weight','last.pt','--img','416','--save-txt','--save-conf'])
        return render_template('index.html', filename=filename)
    else:
        # flash('Allowed image types are - mp3/wav')
        return redirect(request.url)


def gen(camera):
    """Video streaming generator function."""
    ctr = 0
    while True:
        frame, func = camera.get_frame()
        object = func
        print(object)
        ctr += 1
        if object == "Play" and ctr > 100:
        #     js2py.eval_js("""var audio= document.getElementById('audio');
        # var playPauseBTN = document.getElementById('playPauseBTN');
        # var count=0;
        # function playPause(){
        #     if(count == 0){
        #         count = 1;
        #         audio.play();
        #         playPauseBTN.innerHTML = "Pause &#9208;"
        #     }else{
        #         count = 0;
        #         audio.pause();
        #         playPauseBTN.innerHTML = "Play &#9658;"
        #         }
        #     }
        #     playPause()""")
            # ctx.call("playPause()")
            play(audio_file)
            print("playyyy")
            ctr = 0 
        elif object == "Stop":
            stop()
        elif object == "Mute":
            Mute()  
        elif object == "Volume-20":
            volume_20()  
        elif object == "Volume-40":
            volume_40()  
        elif object == "Volume-60":
            volume_60()  
        elif object == "Volume-80":
            volume_80()              
        else:
            volume_max()    
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/display/<filename>')
def display_audio(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run()
