from django.urls import path
from . import views

urlpatterns = [
    path("train", views.train_view, name="train"),
    path('single-predict/', views.single_predict, name='single_predict'),
    path('bulk-predict/', views.bulk_predict, name='bulk_predict'),
    path('bulk-predict/download-csv/', views.download_csv, name='download_csv'),
    path('bulk-predict/download-report/', views.download_pdf_report, name='download_report'),
    path('bulk-predict/download-recommendations-csv/', views.download_recommendations_csv, name='download_recommendations_csv'),
    
    path('',views.landing,name="landing"),
    path('register/', views.register, name='register'),
    path('verify-email/',views.verify_email,name='verify_email'),
    path('resend-otp/', views.resend_otp, name='resend_otp'),
    path('reset-password/', views.reset_password, name='reset_password'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('index', views.index, name='index'),
]
