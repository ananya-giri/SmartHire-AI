import os
import smtplib
from email.message import EmailMessage
from crewai.tools import tool

@tool("send_email")
def send_email(to_email: str, subject: str, content: str) -> str:
    """Useful for automatically sending an email to a candidate.
    Requires to_email, subject, and content."""
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")
    
    if not sender_email or not sender_password:
        # Fallback mock sending if credentials not set by the user
        print("\n" + "="*40)
        print("MOCK EMAIL SEND - (Set SMTP_EMAIL and SMTP_PASSWORD in .env to actually send)")
        print(f"TO: {to_email}\nSUBJECT: {subject}\nBODY:\n{content}")
        print("="*40 + "\n")
        return "Simulated success! Email 'sent'. Tell user to set SMTP_EMAIL and SMTP_PASSWORD in .env for actual sending."
    
    try:
        msg = EmailMessage()
        msg.set_content(content)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to_email

        # Send the message via our own SMTP server (Assuming Gmail for standard testing)
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return f"Email specifically sent successfully to {to_email}!"
    except Exception as e:
        return f"Failed to send email to {to_email}. Error: {e}"

def get_email_tool():
    return send_email
