#!/usr/bin/env python
import pyaudio
import socket
import aubio
from aubio import onset, tempo, pitch
import numpy as np
import click
import time
import json
import dataclasses
import threading
from dataclasses import dataclass

# https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses/51286749#51286749
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

@dataclass
class PitchResult:
    pitch: float
    confidence: float

def process_pitch(signal):
    global g_pitch
    pitch = float(g_pitch(signal)[0])
    confidence = float(g_pitch.get_confidence())
    return PitchResult(pitch=pitch, confidence=confidence)

@dataclass
class TempoResult:
    bpm: float
    confidence: float
    is_beat: bool

def process_tempo(signal):
    global g_tempo
    tempo_info = g_tempo(signal)
    bpm = float(g_tempo.get_bpm())
    confidence = float(g_tempo.get_confidence())
    is_beat = True if tempo_info[0] else False
    return TempoResult(bpm=bpm, confidence=confidence, is_beat=is_beat)

@dataclass
class OnsetResult:
    is_onset: bool

def process_onset(signal):
    global g_onset
    onset = g_onset(signal)[0]
    if onset > 0:
        is_onset = True
    else:
        is_onset = False
    return OnsetResult(is_onset=is_onset)

@dataclass
class Result:
    timestamp: float
    onset: OnsetResult
    tempo: TempoResult
    pitch: PitchResult

def processor(in_data):
    global g_sock
    global g_options

    signal = np.frombuffer(in_data, dtype=np.float32)
    onset_info = g_onset(signal)

    pitch_result = process_pitch(signal)
    tempo_result = process_tempo(signal)
    onset_result = process_onset(signal)
    res = Result(onset=onset_result, pitch=pitch_result, tempo=tempo_result, timestamp=time.time())
    res_json = json.dumps(json.loads(json.dumps(res, cls=EnhancedJSONEncoder), parse_float=lambda x: round(float(x), 3)))
    g_sock.sendto(bytes(res_json, "utf-8"), (g_options["ip"], g_options["port"]))

def pyaudio_callback(in_data, frame_count, time_info, status):
    p = threading.Thread(target=processor, args={in_data})
    p.start()
    return None, pyaudio.paContinue

@click.command()
@click.option("--ip", default="127.0.0.1", help="ip of the receiver")
@click.option("--port", default=9000, help="port of the receiver")
def process(ip, port):
    global g_onset
    global g_tempo
    global g_pitch
    global g_sock
    global g_options

    g_options = {
        "ip": ip,
        "port": port
    }

    p = pyaudio.PyAudio()
    samplerate = 44100
    fft_size = 1024
    hop_size = int(fft_size / 2)
    buf_size = hop_size
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=samplerate,
        input=True,
        frames_per_buffer=buf_size,
        stream_callback=pyaudio_callback
    )
    g_onset = onset("default", fft_size, hop_size, samplerate)
    g_tempo = tempo("specdiff", fft_size, hop_size, samplerate)
    g_pitch = pitch("yin", fft_size, hop_size, samplerate)
    g_sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        time.sleep(1)

if __name__ == "__main__":
    process()
