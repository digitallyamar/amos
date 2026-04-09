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

engine = litert_lm.Engine(LLM_PATH, audio_backend=litert_lm.Backend.CPU)
tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='indic', speaker='v3_indic')

def play_cue():
    subprocess.run(['aplay', 'beep.wav'], stderr=subprocess.DEVNULL)

def record_audio_vad(filename="input.wav", sample_rate=16000):
    vad = webrtcvad.Vad(3)
    pa = pyaudio.PyAudio()
    frame_duration_ms = 30
    chunk_size = int(sample_rate * frame_duration_ms / 1000)
    
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                     input=True, frames_per_buffer=chunk_size)

    print("Listening for speech...")
    frames = []
    num_silent_frames = 0
    speech_detected = False
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
            if num_silent_frames > max_silence_frames:
                print("Speech ended.")
                break

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
        # Wake up Bluetooth pods
        subprocess.run(['aplay', 'silence.wav'], stderr=subprocess.DEVNULL)
        
        # Track the playback process globally within the loop
        playback_process = None

        while True:
            # --- ASYNC CHECK ---
            # If the assistant is still talking from the previous turn, 
            # wait for it to finish before showing the "Ready" prompt.
            if playback_process and playback_process.poll() is None:
                playback_process.wait()

            print("\n--- Ready! Speak after the beep ---")
            play_cue() 
            
            audio_path = record_audio_vad() 
            
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
                answer = response["content"][0]["text"] 
                print(f"Gemma: {answer}") 

                # TTS Processing
                roman_text = transliterate.process('Devanagari', 'ISO', answer) 
                audio_out = tts_model.apply_tts(roman_text, speaker='hindi_male') 
                audio_int16 = (audio_out.numpy().flatten() * 32767).astype(np.int16) 

                # --- NON-BLOCKING PLAYBACK ---
                # We start the process and write to stdin immediately.
                # Because we don't call .communicate(), Python continues execution 
                # as soon as the data is handed off to the OS audio buffer.
                playback_process = subprocess.Popen(
                    ['aplay', '-t', 'raw', '-f', 'S16_LE', '-r', '48000', '-c', '1'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                playback_process.stdin.write(audio_int16.tobytes())
                playback_process.stdin.close() # Signal end of data to aplay

            except Exception as e: 
                print(f"Error during inference: {e}")

if __name__ == "__main__": 
    run_app()
