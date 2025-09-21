import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        self.from_name = os.getenv("FROM_NAME", "Intelligent Document Assistant")
    
    def send_otp_email(self, to_email: str, otp_code: str, purpose: str = "login") -> bool:
        """Send OTP verification email"""
        try:
            if not self.smtp_username or not self.smtp_password:
                logger.warning("SMTP credentials not configured, printing OTP to console")
                print(f"\n{'='*60}")
                print(f"[DEV MODE] OTP EMAIL")
                print(f"To: {to_email}")
                print(f"Purpose: {purpose}")
                print(f"OTP Code: {otp_code}")
                print(f"{'='*60}\n")
                return True
            
            subject = self._get_otp_subject(purpose)
            html_body = self._get_otp_html_body(otp_code, purpose)
            text_body = self._get_otp_text_body(otp_code, purpose)
            
            return self._send_email(to_email, subject, html_body, text_body)
            
        except Exception as e:
            logger.error(f"Failed to send OTP email to {to_email}: {str(e)}")
            return False
    
    def _send_email(self, to_email: str, subject: str, html_body: str, text_body: str) -> bool:
        """Send email via SMTP"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = to_email
            
            # Add text and HTML parts
            text_part = MIMEText(text_body, "plain")
            html_part = MIMEText(html_body, "html")
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"OTP email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"SMTP error: {str(e)}")
            return False
    
    def _get_otp_subject(self, purpose: str) -> str:
        """Get email subject based on purpose"""
        subjects = {
            "login": "Your Login Verification Code",
            "registration": "Welcome! Verify Your Email",
            "password_reset": "Password Reset Verification Code"
        }
        return subjects.get(purpose, "Your Verification Code")
    
    def _get_otp_html_body(self, otp_code: str, purpose: str) -> str:
        """Get HTML email body"""
        action = {
            "login": "sign in to your account",
            "registration": "complete your registration", 
            "password_reset": "reset your password"
        }.get(purpose, "verify your identity")
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #4f46e5; color: white; text-align: center; padding: 20px; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 8px 8px; }}
                .otp-code {{ font-size: 32px; font-weight: bold; text-align: center; 
                            background: #1f2937; color: white; padding: 20px; 
                            border-radius: 8px; margin: 20px 0; letter-spacing: 8px; }}
                .footer {{ text-align: center; margin-top: 20px; color: #6b7280; font-size: 14px; }}
                .warning {{ background: #fef3c7; border: 1px solid #f59e0b; color: #92400e; 
                            padding: 15px; border-radius: 6px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Intelligent Document Assistant</h1>
                </div>
                <div class="content">
                    <h2>Verification Code</h2>
                    <p>Hi there! Use this code to {action}:</p>
                    
                    <div class="otp-code">{otp_code}</div>
                    
                    <div class="warning">
                        <strong>Security Notice:</strong><br>
                        • This code expires in 10 minutes<br>
                        • Don't share this code with anyone<br>
                        • If you didn't request this, please ignore this email
                    </div>
                    
                    <p>If you have any questions, feel free to reach out to our support team.</p>
                </div>
                <div class="footer">
                    <p>This is an automated email. Please do not reply to this message.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_otp_text_body(self, otp_code: str, purpose: str) -> str:
        """Get plain text email body"""
        action = {
            "login": "sign in to your account",
            "registration": "complete your registration",
            "password_reset": "reset your password"
        }.get(purpose, "verify your identity")
        
        return f"""
        Intelligent Document Assistant - Verification Code

        Hi there! Use this code to {action}:

        {otp_code}

        SECURITY NOTICE:
        • This code expires in 10 minutes
        • Don't share this code with anyone  
        • If you didn't request this, please ignore this email

        If you have any questions, feel free to reach out to our support team.

        ---
        This is an automated email. Please do not reply to this message.
        """

# Global email service instance
email_service = EmailService()