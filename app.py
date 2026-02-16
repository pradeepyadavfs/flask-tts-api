"""
Flask API for TTS Generation - Render Ready Version
"""

from flask import Flask, request, jsonify, send_from_directory
import os
import traceback
import soundfile as sf

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import Twilio call module
from twilio_call import make_call, get_audio_public_url, check_twilio_configured

# Create static/audio directory for generated audio files
AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'static', 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)

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
        "status": "healthy"
    }), 200

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate TTS audio from text and optionally make a phone call.
    Expected JSON: {"text": "Your text here", "make_call": true}
    Returns: {"success": true, "file": "filename.wav", "call_sid": "..."}
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
        
        # Generate unique filename for this request
        import uuid
        filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(AUDIO_DIR, filename)
        
        # Generate speech using pre-loaded models (no reloading)
        print("[GENERATING] Processing text...")
        inputs = tts_processor(text=text, return_tensors="pt")
        
        print("[GENERATING] Generating speech...")
        speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=tts_vocoder)
        
        print(f"[GENERATING] Saving to {filename}...")
        speech_np = speech.cpu().numpy()
        sf.write(output_path, speech_np, samplerate=16000)
        
        print("[SUCCESS] Audio generated successfully")
        
        # Build response - use the public URL path
        audio_url = f"/audio/{filename}"
        response_data = {
            "success": True,
            "file": audio_url
        }
        
        # Make phone call if requested
        if make_call_flag and check_twilio_configured():
            # Get the public base URL from environment or construct one
            public_base_url = os.environ.get('PUBLIC_BASE_URL', '')
            if public_base_url:
                # Use the public base URL if configured
                full_audio_url = public_base_url.replace('/audio.wav', f'/{filename}')
            else:
                # For local development, use the relative path
                full_audio_url = audio_url
            
            print(f"[TWILIO] Initiating call to {target_number or 'default'}...")
            call_result = make_call(full_audio_url, target_number)
            
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

@app.route('/audio/<filename>', methods=['GET'])
def serve_audio(filename):
    """Serve generated audio files"""
    # Security: only allow audio files
    if not filename.endswith('.wav'):
        return jsonify({"error": "Invalid file type"}), 400
    
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting Flask API on 0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

