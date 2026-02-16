"""
Twilio Call Integration Module
Handles outbound phone calls with generated audio playback
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Twilio configuration from environment variables
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '')
TARGET_PHONE_NUMBER = os.environ.get('TARGET_PHONE_NUMBER', '')

# Placeholder URL for audio - replace with actual S3/cloud storage in production
PUBLIC_BASE_URL = os.environ.get('PUBLIC_BASE_URL', 'https://your-public-server.com')


def get_audio_public_url(filename: str = "output.wav") -> str:
    """
    Generate public URL for the audio file.
    For now, returns placeholder URL - ready for S3/cloud storage integration.
    
    Args:
        filename: The audio file name
        
    Returns:
        Public URL string
    """
    return f"{PUBLIC_BASE_URL}/{filename}"


def make_call(audio_url: str, target_number: Optional[str] = None) -> Dict[str, Any]:
    """
    Initiate outbound Twilio call and play audio file.
    
    Args:
        audio_url: Public URL of the audio file to play
        target_number: Optional override for target phone number
        
    Returns:
        Dict with call_sid and status, or error information
    """
    # Use provided number or fall back to environment variable
    to_number = target_number or TARGET_PHONE_NUMBER
    
    # Validate configuration
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.warning("Twilio credentials not configured")
        return {
            "success": False,
            "error": "Twilio not configured",
            "call_sid": None
        }
    
    if not TWILIO_PHONE_NUMBER:
        logger.warning("Twilio phone number not configured")
        return {
            "success": False,
            "error": "Twilio phone number not set",
            "call_sid": None
        }
    
    if not to_number:
        logger.warning("Target phone number not configured")
        return {
            "success": False,
            "error": "Target phone number not set",
            "call_sid": None
        }
    
    try:
        from twilio.rest import Client
        from twilio.twiml import Response
        
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Create TwiML response with audio playback
        twiml_response = Response()
        twiml_response.play(audio_url)
        
        # Initiate the call
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            twiml=str(twiml_response)
        )
        
        logger.info(f"Call initiated successfully: {call.sid}")
        
        return {
            "success": True,
            "call_sid": call.sid,
            "status": call.status,
            "to": to_number,
            "from": TWILIO_PHONE_NUMBER
        }
        
    except ImportError:
        logger.error("Twilio SDK not installed. Run: pip install twilio")
        return {
            "success": False,
            "error": "Twilio SDK not installed",
            "call_sid": None
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate call: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "call_sid": None
        }


def check_twilio_configured() -> bool:
    """Check if Twilio is properly configured."""
    return bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER)

