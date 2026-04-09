import os
import sys
import torch
import litert_lm
import subprocess
import multiprocessing
import soundfile as sf
import sounddevice as sd
from kokoro import KPipeline
from dotenv import load_dotenv
from aksharamukha import transliterate

# Make sure sudo apt-get install libportaudio2 is done

# 1. Standard environment suppression
os.environ['GLOG_minloglevel'] = '2'

'''
# Kokoro support
load_dotenv()
token = os.getenv("HF_TOKEN")
'''

# 2. Loading model
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='indic',
                                     speaker='v3_indic')


def inference_worker(model_path, prompt, queue):
    """
    Runs in a separate process. Redirects entire output to devnull 
    at the OS level to swallow C++ logs.
    """
    with open(os.devnull, 'w') as fnull:
        os.dup2(fnull.fileno(), sys.stdout.fileno())
        os.dup2(fnull.fileno(), sys.stderr.fileno())
        
        try:
            with litert_lm.Engine(model_path) as engine:
                with engine.create_conversation() as conversation:
                    response = conversation.send_message(prompt)
                    # FIX: Access the first element [0] of the content list
                    text_result = response["content"][0]["text"]
                    queue.put(text_result)
        except Exception as e:
            queue.put(f"Error: {e}")

def run_app():
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/616f4124e6ff216292f16e7f73ff33b5ba9a4dd4/gemma-4-E2B-it.litertlm"
    )

    prompt = "अपना उत्तर संक्षिप्त रखें।"
    prompt += "बेंगलुरु किस लिए प्रसिद्ध है?"
    
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=inference_worker, args=(model_path, prompt, queue))
    p.start()
    
    # This will now get the clean string
    answer = queue.get()
    p.join()

    print(f"Question: {prompt}")
    print(f"Response: {answer}")

    ## TTS
    # Note: Silero v3/v4 models often output at 16000, 24000, or 48000.
    # If 16000 sounded slow, try 24000 or 48000 here
    sample_rate = 48000
    roman_text = transliterate.process('Devanagari', 'ISO', answer)
    print(f"Transliterated: {roman_text}")

    # 2. Generate the audio tensor
    audio = model.apply_tts(roman_text, speaker='hindi_male')

    # 3. Convert Tensor to NumPy and save
    # We use .numpy() to convert it and sf.write to save it
    audio_numpy = audio.numpy().flatten()

    # Play the audio reply
    # sd.play(audio_numpy, sample_rate)
    # print("Audio played successfully")
    # Open aplay as a subprocess and feed it the bytes via stdin
    # Silero usually outputs float32, so we convert to int16 for aplay
    audio_int16 = (audio_numpy * 32767).astype('int16')
    audio_bytes = audio_int16.tobytes()

    # Open aplay as a subprocess and feed it the bytes via stdin
    print("Playing audio via aplay pipe...")
    # Use -r 48000 if 16000 sounds slow; -c 1 specifies mono
    process = subprocess.Popen(
        ['aplay', '-t', 'raw', '-f', 'S16_LE', '-r', '48000'],
        stdin=subprocess.PIPE
    )
    process.communicate(input=audio_bytes)

    # Write to external wav audio output file
    sf.write('output.wav', audio_numpy, sample_rate)
    print("Audio saved successfully as output.wav")


    '''
    ## Kokoro TTS
    pipeline = KPipeline(lang_code='h')

    # 4?? Generate, display, and save audio files in a loop.
    generator = pipeline(
        roman_text, voice='hf_alpha', # <= change voice here
        speed=1, split_pattern=r'\n+'
    )

    for i, (gs, ps, audio) in enumerate(generator):
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes
        sf.write(f'{i}_kokoro.wav', audio, 24000) # save each audio file
    '''

if __name__ == "__main__":
    run_app()
