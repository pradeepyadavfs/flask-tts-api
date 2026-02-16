"""
TTS Script using Hugging Face - Fixed Version
Loads models ONCE at startup for stability
"""

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
import numpy as np
import os

# Global variables for loaded models (loaded once at startup)
processor = None
model = None
vocoder = None
speaker_embeddings = None
models_loaded = False

def load_models():
    """
    Load all TTS models ONCE at startup.
    This prevents memory issues and slow requests.
    """
    global processor, model, vocoder, speaker_embeddings, models_loaded
    
    if models_loaded:
        print("Models already loaded, skipping...")
        return
    
    print("Loading processor...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    
    print("Loading TTS model...")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    
    print("Loading vocoder...")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    print("Loading speaker embeddings...")
    from datasets import load_dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    models_loaded = True
    print("All models loaded successfully!")

def generate_speech(text, output_file="output.wav"):
    """
    Generate speech from text using pre-loaded Hugging Face TTS model.
    
    Args:
        text: Input text (max 100 characters recommended)
        output_file: Output WAV file path
    
    Returns:
        bool: True if successful, False otherwise
    """
    global processor, model, vocoder, speaker_embeddings
    
    try:
        # Check if models are loaded
        if not models_loaded:
            load_models()
        
        print(f"Processing text: {text}")
        
        # Process the input text
        inputs = processor(text=text, return_tensors="pt")
        
        # Generate speech
        print("Generating speech...")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        
        # Save as WAV file
        print(f"Saving to {output_file}...")
        speech_np = speech.cpu().numpy()
        sf.write(output_file, speech_np, samplerate=16000)
        
        print(f"Done! Audio saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"ERROR in generate_speech: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Main execution - for testing only
if __name__ == "__main__":
    # Load models first
    load_models()
    
    # Test generation
    text = "Hello world, this is a test of text to speech."
    generate_speech(text, "output.wav")

