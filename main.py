import os
import torch
import litert_lm
import subprocess
import numpy as np
from aksharamukha import transliterate

# --- GLOBAL INITIALIZATION ---
LLM_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/616f4124e6ff216292f16e7f73ff33b5ba9a4dd4/gemma-4-E2B-it.litertlm")

# 1. Initialize Engine with Audio Backend (Crucial for Multimodal)
# The audio_backend=litert_lm.Backend.CPU is necessary for native audio handling
engine = litert_lm.Engine(LLM_PATH, audio_backend=litert_lm.Backend.CPU)

# 2. Load TTS model globally
tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='indic', speaker='v3_indic')

def record_audio(filename="input.wav", duration=4): 
    # Capture at 16kHz for Gemma native audio processing 
    subprocess.run(['arecord', '-f', 'S16_LE', '-r', '16000', '-c', '1', '-d', str(duration), filename], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
    # Return the absolute path as required by the API 
    return os.path.abspath(filename)

def run_app(): 
    with engine.create_conversation() as conversation: 
        while True: 
            # Step 1: Capture 
            audio_path = record_audio() 
            # Step 2: Correct Multi-modal Structure 
            # We use "path" instead of "audio" to satisfy the 'path or blob' requirement 
            user_message = { 
                "role": "user", "content": [ 
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