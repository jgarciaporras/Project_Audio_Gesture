import soundfile as sf
import sounddevice as sd

import numpy as np
from pydub import AudioSegment
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math


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


def Mute():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    setup_mute =-65.25
    volume.SetMasterVolumeLevel(setup_mute, None)