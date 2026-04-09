import os
import wave
import torch
import pyaudio
import webrtcvad
import litert_lm
import subprocess
import numpy as np
from aksharamukha import transliterate
from openwakeword.model import Model

# --- SETTINGS ---
os.environ['ORT_LOGGING_LEVEL'] = '3' # Suppress ONNX warnings
WAKE_WORD = "alexa" # Or "hey_jarvis"
LLM_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/616f4124e6ff216292f16e7f73ff33b5ba9a4dd4/gemma-4-E2B-it.litertlm")

# --- INITIALIZATION ---
engine = litert_lm.Engine(LLM_PATH, audio_backend=litert_lm.Backend.CPU)
tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='indic', speaker='v3_indic')
oww_model = Model(wakeword_models=[WAKE_WORD])
vad = webrtcvad.Vad(3)
pa = pyaudio.PyAudio()

def play_cue():
    subprocess.run(['aplay', 'beep.wav'], stderr=subprocess.DEVNULL)

def listen_and_process(stream, conversation):
    """Wait for wake word, then record speech, then get LLM response."""
    
    # 1. WAIT FOR WAKE WORD
    print(f"Listening for '{WAKE_WORD}'...")
    while True:
        data = stream.read(1280, exception_on_overflow=False)
        audio_frame = np.frombuffer(data, dtype=np.int16)
        prediction = oww_model.predict(audio_frame)
        if prediction[WAKE_WORD] > 0.5:
            print(f"Wake word detected!")
            break

    # 2. TRIGGER CUE AND RECORD VAD
    play_cue()
    print("Recording speech...")
    frames = []
    num_silent_frames = 0
    speech_detected = False
    
    # Settings for VAD (using 30ms frames to match 16kHz)
    chunk_vad = 480 
    max_silence_frames = 30 # ~1 second of silence to stop

    while True:
        data = stream.read(chunk_vad, exception_on_overflow=False)
        is_speech = vad.is_speech(data, 16000)

        if not speech_detected:
            if is_speech:
                speech_detected = True
                frames.append(data)
        else:
            frames.append(data)
            num_silent_frames = 0 if is_speech else num_silent_frames + 1
            if num_silent_frames > max_silence_frames:
                break

    # Save audio for Gemma
    temp_file = os.path.abspath("input.wav")
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    
    return temp_file

def run_app(): 
    # Open a single persistent stream for efficiency
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                     input=True, frames_per_buffer=1280)
    
    with engine.create_conversation() as conversation: 
        subprocess.run(['aplay', 'silence.wav'], stderr=subprocess.DEVNULL)
        playback_process = None

        while True:
            # Wait for previous response to finish playing
            if playback_process and playback_process.poll() is None:
                playback_process.wait()

            try:
                # Get audio input via Wake Word + VAD
                audio_path = listen_and_process(stream, conversation)
                
                print("--- Gemma is processing ---") 
                user_message = { 
                    "role": "user", 
                    "content": [ 
                        {"type": "audio", "path": audio_path}, 
                        {"type": "text", "text": "à¤¸à¤‚à¤•à¥à¤·à¥‡à¤ª à¤®à¥‡à¤‚ à¤‰à¤¤à¥à¤¤à¤° à¤¦à¥‡à¤‚à¥¤"} 
                    ] 
                } 

                response = conversation.send_message(user_message) 
                answer = response["content"][0]["text"] 
                print(f"Gemma: {answer}") 

                # TTS & Async Playback
                roman_text = transliterate.process('Devanagari', 'ISO', answer) 
                audio_out = tts_model.apply_tts(roman_text, speaker='hindi_male') 
                audio_int16 = (audio_out.numpy().flatten() * 32767).astype(np.int16) 

                playback_process = subprocess.Popen(
                    ['aplay', '-t', 'raw', '-f', 'S16_LE', '-r', '48000', '-c', '1'],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                playback_process.stdin.write(audio_int16.tobytes())
                playback_process.stdin.close() 

            except Exception as e: 
                print(f"Error: {e}")

if __name__ == "__main__": 
    try:
        run_app()
    finally:
        pa.terminate()
