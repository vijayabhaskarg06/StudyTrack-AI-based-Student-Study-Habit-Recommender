from django.core.mail import send_mail
from django.conf import settings

def send_email_otp(email, otp):
    send_mail(
        subject="Verify Your Email - StudyTrack",
        message=f"""
Hello,

Your email verification OTP is: {otp}

This OTP is valid for 5 minutes.
Do not share this OTP with anyone.

– StudyTrack AI Team
""",
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[email],
        fail_silently=False,
    )

def send_reset_password_otp(email, otp):
    send_mail(
        subject="Reset Your Password – StudyTrack",
        message=f"""
Hello,

We received a request to reset your StudyTrack account password.

Your password reset OTP is:

{otp}

This OTP is valid for 5 minutes.
If you did not request a password reset, please ignore this email.
Do NOT share this OTP with anyone.

– StudyTrack AI Team
""",
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[email],
        fail_silently=False,
    )

