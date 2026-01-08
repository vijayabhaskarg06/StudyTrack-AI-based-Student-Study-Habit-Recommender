# accounts/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    is_email_verified = models.BooleanField(default=False)
    email_otp = models.CharField(max_length=255, blank=True, null=True)
    otp_expiry = models.DateTimeField(blank=True, null=True)