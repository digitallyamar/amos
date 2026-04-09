import os
import wave
import torch
import pyaudio
import webrtcvad
import litert_lm
import subprocess
import collections
import numpy as np
from aksharamukha import transliterate

# --- GLOBAL INITIALIZATION ---
LLM_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/616f4124e6ff216292f16e7f73ff33b5ba9a4dd4/gemma-4-E2B-it.litertlm")

# 1. Initialize Engine with Audio Backend (Crucial for Multimodal)
# The audio_backend=litert_lm.Backend.CPU is necessary for native audio handling
engine = litert_lm.Engine(LLM_PATH, audio_backend=litert_lm.Backend.CPU)

# 2. Load TTS model globally
tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='indic', speaker='v3_indic')

def play_cue():
    # Option A: Play a system beep sound (if you have one)
    subprocess.run(['aplay', 'beep.wav'], stderr=subprocess.DEVNULL)
    
    # # Option B: Generate a quick 200ms 1000Hz beep via speaker-test
    # subprocess.run(['speaker-test', '-t', 'sine', '-f', '1000', '-l', '1', '-p', '0', '-X'], 
    #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.2)


def record_audio(filename="input.wav", duration=4): 
    # Capture at 16kHz for Gemma native audio processing 
    subprocess.run(['arecord', '-f', 'S16_LE', '-r', '16000', '-c', '1', '-d', str(duration), filename], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
    # Return the absolute path as required by the API 
    return os.path.abspath(filename)


def record_audio_vad(filename="input.wav", sample_rate=16000):
    vad = webrtcvad.Vad(3)  # Aggressiveness mode (0-3). 3 is most aggressive.
    pa = pyaudio.PyAudio()
    
    # Standard settings for webrtcvad
    frame_duration_ms = 30  # ms
    chunk_size = int(sample_rate * frame_duration_ms / 1000)
    
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                     input=True, frames_per_buffer=chunk_size)

    print("Listening for speech...")
    frames = []
    num_silent_frames = 0
    speech_detected = False
    
    # 1.0 second of silence to trigger stop
    max_silence_frames = int(1000 / frame_duration_ms) 

    while True:
        data = stream.read(chunk_size, exception_on_overflow=False)
        is_speech = vad.is_speech(data, sample_rate)

        if not speech_detected:
            if is_speech:
                speech_detected = True
                print("Speech started...")
                frames.append(data)
        else:
            frames.append(data)
            if not is_speech:
                num_silent_frames += 1
            else:
                num_silent_frames = 0

            # Stop if the user stops speaking for the threshold duration
            if num_silent_frames > max_silence_frames:
                print("Speech ended.")
                break

    # Clean up and save
    stream.stop_stream()
    stream.close()
    pa.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    return os.path.abspath(filename)

def run_app(): 
    with engine.create_conversation() as conversation: 
        subprocess.run(['aplay', 'silence.wav'], stderr=subprocess.DEVNULL)
        while True:
            print("\n--- Ready! Speak after the beep ---")
            play_cue() # <--- TRIGGER CUE HERE
            # Step 1: Capture 
            audio_path = record_audio_vad() 
            # Step 2: Correct Multi-modal Structure 
            # We use "path" instead of "audio" to satisfy the 'path or blob' requirement 
            user_message = { 
                "role": "user", 
                "content": [ 
                    {"type": "audio", "path": audio_path}, 
                    {"type": "text", "text": "संक्षेप में उत्तर दें।"} 
                ] 
            } 
            print("--- Gemma is processing audio ---") 
            try: 
                response = conversation.send_message(user_message) 
                # Correct response extraction for modern litert-lm schemas 
                answer = response["content"][0]["text"] 
                print(f"Gemma: {answer}") 
                # Step 3: Speak 
                roman_text = transliterate.process('Devanagari', 'ISO', answer) 
                audio_out = tts_model.apply_tts(roman_text, speaker='hindi_male') 
                audio_int16 = (audio_out.numpy().flatten() * 32767).astype(np.int16) 
                process = subprocess.Popen( ['aplay', '-t', 'raw', '-f', 'S16_LE', '-r', '48000', '-c', '1'], stdin=subprocess.PIPE )
                process.communicate(input=audio_int16.tobytes()) 
            except Exception as e: 
                print(f"Error during inference: {e}")


if __name__ == "__main__": 
    run_app()