from backend.app.services.email_service import email_service

@staticmethod
def send_email_otp(email: str, otp_code: str, purpose: str, user_name: Optional[str] = None) -> bool:
    """Send OTP email for given purpose"""
    try:
        print(f"[DEBUG] Sending OTP to {email} Via EmailService")
        purpose_mapping = {
            "email_verification": "registration",
            "password_reset": "password_reset",
            "2fa": "login",
            "login": "login"
        }
        
        email_purpose = purpose_mapping.get(purpose, "login")
        
        result = email_service.send_otp_email(email, otp_code, email_purpose)
        print(f"[DEBUG] EmailService result: {result}")
        
        return result
    except Exception as e:
        print(f"[ERROR] Failed to send OTP email: {e}")
        return False