# test_smtp.py
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get SMTP settings from environment
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USERNAME)

def test_smtp_connection():
    try:
        print(f"Testing SMTP connection to {SMTP_SERVER}:{SMTP_PORT}")
        print(f"Username: {SMTP_USERNAME}")
        print(f"From Email: {FROM_EMAIL}")
        
        if not SMTP_USERNAME or not SMTP_PASSWORD:
            print("❌ Missing SMTP_USERNAME or SMTP_PASSWORD in .env file")
            return False
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Send test email
        msg = MIMEText("Test email from your RAG system")
        msg['Subject'] = 'SMTP Test'
        msg['From'] = FROM_EMAIL
        msg['To'] = "satya.bulusuv@gmail.com"  # Your email from the logs
        
        server.send_message(msg)
        server.quit()
        
        print("✅ SMTP test successful!")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ SMTP Authentication failed: {e}")
        print("Check your Gmail App Password")
        return False
    except Exception as e:
        print(f"❌ SMTP test failed: {e}")
        return False

if __name__ == "__main__":
    test_smtp_connection()