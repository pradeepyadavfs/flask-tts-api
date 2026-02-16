"""
Flask API for TTS Generation - Minimal Version
Loads TTS model ONCE at startup
"""

from flask import Flask, request, jsonify
import os
import traceback
import soundfile as sf

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import Twilio call module
from twilio_call import make_call, get_audio_public_url, check_twilio_configured

app = Flask(__name__)

# Global variables for loaded models (loaded once at startup)
tts_processor = None
tts_model = None
tts_vocoder = None
speaker_embeddings = None
models_loaded = False

def load_models():
    """Load all TTS models ONCE at startup."""
    global tts_processor, tts_model, tts_vocoder, speaker_embeddings, models_loaded
    
    if models_loaded:
        print("Models already loaded, skipping...")
        return
    
    print("Loading TTS models at startup...")
    
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    import torch
    import numpy as np
    
    print("  Loading processor...")
    tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    
    print("  Loading TTS model...")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    
    print("  Loading vocoder...")
    tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    print("  Loading speaker embeddings...")
    speaker_embeddings = torch.zeros(1, 512)  # Placeholder embedding
    print("  Using placeholder speaker embeddings")
    
    models_loaded = True
    print("All TTS models loaded successfully!")

# Load models ONCE at startup
print("=" * 50)
load_models()
print("=" * 50)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "twilio_configured": check_twilio_configured()
    }), 200

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate TTS audio from text and optionally make a phone call.
    Expected JSON: {"text": "Your text here", "make_call": true}
    Returns: {"success": true, "file": "output.wav", "call_sid": "..."}
    """
    global tts_processor, tts_model, tts_vocoder, speaker_embeddings
    
    try:
        print("\n[REQUEST] /generate called")
        
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "message": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        print(f"[REQUEST] Text: {text}")
        
        # Check if call should be made (default: true if Twilio configured)
        make_call_flag = data.get('make_call', check_twilio_configured())
        target_number = data.get('target_number')
        
        # Validate text length
        if len(text) > 100:
            return jsonify({
                "success": False,
                "message": f"Text too long ({len(text)} chars). Max 100 characters."
            }), 400
        
        # Generate speech using pre-loaded models (no reloading)
        print("[GENERATING] Processing text...")
        inputs = tts_processor(text=text, return_tensors="pt")
        
        print("[GENERATING] Generating speech...")
        speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=tts_vocoder)
        
        print("[GENERATING] Saving to output.wav...")
        speech_np = speech.cpu().numpy()
        sf.write("output.wav", speech_np, samplerate=16000)
        
        print("[SUCCESS] Audio generated successfully")
        
        # Build response
        response_data = {
            "success": True,
            "file": "output.wav"
        }
        
        # Make phone call if requested
        if make_call_flag and check_twilio_configured():
            print("[TWILIO] Getting audio public URL...")
            audio_url = get_audio_public_url("output.wav")
            
            print(f"[TWILIO] Initiating call to {target_number or 'default'}...")
            call_result = make_call(audio_url, target_number)
            
            if call_result.get("success"):
                response_data["call_sid"] = call_result["call_sid"]
                response_data["call_status"] = call_result.get("status", "initiated")
                print(f"[TWILIO] Call initiated: {call_result['call_sid']}")
            else:
                response_data["call_error"] = call_result.get("error", "Unknown error")
                print(f"[TWILIO] Call failed: {call_result.get('error')}")
        elif make_call_flag:
            print("[TWILIO] Twilio not configured, skipping call")
            response_data["call_error"] = "Twilio not configured"
        
        return jsonify(response_data), 200
            
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("\nStarting Flask API on port 5050...")
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)
