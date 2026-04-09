import os
import wave
import time
import torch
import pyaudio
import webrtcvad
import litert_lm
import subprocess
import numpy as np
from aksharamukha import transliterate
from openwakeword.model import Model

# --- SETTINGS ---
os.environ['ORT_LOGGING_LEVEL'] = '3'  # Suppress ONNX warnings
WAKE_WORD = "alexa"
LLM_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/616f4124e6ff216292f16e7f73ff33b5ba9a4dd4/gemma-4-E2B-it.litertlm")

# --- INITIALIZATION ---
print("Initializing engines...")
engine = litert_lm.Engine(LLM_PATH, audio_backend=litert_lm.Backend.CPU)
tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='indic', speaker='v3_indic')
oww_model = Model(wakeword_models=[WAKE_WORD])
vad = webrtcvad.Vad(3)
pa = pyaudio.PyAudio()

def play_cue():
    """Signifies the assistant is listening."""
    subprocess.run(['aplay', 'beep.wav'], stderr=subprocess.DEVNULL)

def play_sleep_cue():
    """Signifies the session has ended and it's back to Wake Word mode."""
    # Using your converted wav file
    subprocess.run(['aplay', 'sleeping.wav'], stderr=subprocess.DEVNULL)

def flush_stream(stream):
    """Clear any pending audio in the buffer to prevent ghost triggers."""
    while stream.get_read_available() > 0:
        stream.read(stream.get_read_available(), exception_on_overflow=False)

def wait_for_wake_word(stream):
    print(f"--- Passive Mode: Waiting for '{WAKE_WORD}' ---")
    while True:
        data = stream.read(1280, exception_on_overflow=False)
        audio_frame = np.frombuffer(data, dtype=np.int16)
        prediction = oww_model.predict(audio_frame)
        if prediction[WAKE_WORD] > 0.5:
            print("Wake word detected!")
            oww_model.reset()
            return True

def record_vad_with_timeout(stream, timeout=10):
    flush_stream(stream)
    print(f"--- Active Session: Listening (Timeout in {timeout}s) ---")
    frames = []
    num_silent_frames = 0
    speech_detected = False
    start_time = time.time()
    
    chunk_vad = 480 
    max_silence_frames = 30 # ~1 second of silence

    while True:
        if not speech_detected and (time.time() - start_time) > timeout:
            return None

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

    temp_file = os.path.abspath("input.wav")
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    return temp_file

def run_app(): 
    # Buffer size 4096 prevents overflow during heavy LLM computation
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                     input=True, frames_per_buffer=4096)
    
    is_session_active = False
    
    with engine.create_conversation() as conversation: 
        # keep-alive for Bluetooth headset
        subprocess.run(['aplay', 'silence.wav'], stderr=subprocess.DEVNULL)
        playback_process = None

        while True:
            # Ensure assistant is done talking before checking for next input
            if playback_process and playback_process.poll() is None:
                playback_process.wait()

            # 1. State Management
            if not is_session_active:
                wait_for_wake_word(stream)
                is_session_active = True

            # 2. Audio Capture
            play_cue() 
            audio_path = record_vad_with_timeout(stream, timeout=10)

            if audio_path is None:
                print("Session timed out.")
                play_sleep_cue()
                is_session_active = False
                continue

            # 3. Processing & Intent Check
            try:
                print("Processing...")
                user_message = { 
                    "role": "user", 
                    "content": [ 
                        {"type": "audio", "path": audio_path}, 
                        {"type": "text", "text": "संक्षेप में उत्तर दें।"} 
                    ] 
                } 

                response = conversation.send_message(user_message) 
                answer = response["content"][0]["text"]
                
                # Check for Manual Sleep Intent ("ok bye")
                exit_triggers = ["ok bye", "stop now", "alvida", "अलविदा", "बस"]
                if any(trigger in answer.lower() for trigger in exit_triggers):
                    print(f"Gemma: {answer}")
                    print("Intent: Shutdown. Ending session...")
                    
                    # Say the final goodbye before sleeping
                    roman_text = transliterate.process('Devanagari', 'ISO', answer) 
                    audio_out = tts_model.apply_tts(roman_text, speaker='hindi_male') 
                    audio_int16 = (audio_out.numpy().flatten() * 32767).astype(np.int16) 
                    
                    p = subprocess.Popen(['aplay', '-t', 'raw', '-f', 'S16_LE', '-r', '48000', '-c', '1'], stdin=subprocess.PIPE)
                    p.communicate(input=audio_int16.tobytes())
                    
                    play_sleep_cue()
                    is_session_active = False
                    continue 

                print(f"Gemma: {answer}") 

                # 4. TTS and Playback
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
                is_session_active = False

if __name__ == "__main__": 
    try:
        run_app()
    finally:
        pa.terminate()
