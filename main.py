import os
import sys
import torch
import litert_lm
import subprocess
import multiprocessing
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from aksharamukha import transliterate

# 1. Standard environment suppression
os.environ['GLOG_minloglevel'] = '2'

# 2. Loading Silero TTS Model (Indic)
tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='indic',
                                     speaker='v3_indic')

def inference_worker(model_path, prompt, queue):
    with open(os.devnull, 'w') as fnull:
        os.dup2(fnull.fileno(), sys.stdout.fileno())
        os.dup2(fnull.fileno(), sys.stderr.fileno())
        try:
            with litert_lm.Engine(model_path) as engine:
                with engine.create_conversation() as conversation:
                    response = conversation.send_message(prompt)
                    text_result = response["content"][0]["text"]
                    queue.put(text_result)
        except Exception as e:
            queue.put(f"Error: {e}")

def record_audio(filename="input.wav", duration=5):
    print(f"--- Recording for {duration}s (Talk now!) ---")
    # Capture raw audio from Bluetooth mic via arecord
    subprocess.run(['arecord', '-f', 'S16_LE', '-r', '48000', '-c', '1', '-d', str(duration), filename], check=True)

def transcribe_hindi(filename="input.wav"):
    print("--- Transcribing Hindi Speech ---")
    # Using 'small-hindi' fine-tuned for better accuracy
    # 'int8' quantization saves memory on Pi 5
    model = WhisperModel("collabora/faster-whisper-small-hindi", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(filename, language="hi", beam_size=5)
    text = "".join([segment.text for segment in segments])
    return text

def run_app():
    llm_path = os.path.expanduser("~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/616f4124e6ff216292f16e7f73ff33b5ba9a4dd4/gemma-4-E2B-it.litertlm")
    
    # --- STEP 1: Listen ---
    record_audio("input_capture.wav", duration=5)
    system_prompt = "अपना उत्तर संक्षिप्त रखें।"
    hindi_prompt = transcribe_hindi("input_capture.wav")
    print(f"You said: {hindi_prompt}")

    if not hindi_prompt.strip():
        print("No speech detected. Exiting.")
        return

    # --- STEP 2: Think (LLM) ---
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=inference_worker, args=(llm_path, system_prompt + hindi_prompt, queue))
    p.start()
    answer = queue.get()
    p.join()
    print(f"Gemma Response: {answer}")

    # --- STEP 3: Speak (TTS) ---
    sample_rate = 48000 
    roman_text = transliterate.process('Devanagari', 'ISO', answer)
    audio = tts_model.apply_tts(roman_text, speaker='hindi_male')
    
    audio_numpy = audio.numpy().flatten()
    audio_int16 = (audio_numpy * 32767).astype(np.int16)
    
    print("--- Playing Response via Bluetooth ---")
    process = subprocess.Popen(
        ['aplay', '-t', 'raw', '-f', 'S16_LE', '-r', str(sample_rate), '-c', '1'],
        stdin=subprocess.PIPE
    )
    process.communicate(input=audio_int16.tobytes())

if __name__ == "__main__":
    run_app()
