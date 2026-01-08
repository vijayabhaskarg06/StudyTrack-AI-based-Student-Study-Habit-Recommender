# views.py
import base64
import json
import os
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.http import FileResponse, HttpResponse
from django.shortcuts import render

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)
from reportlab.platypus.flowables import HRFlowable


# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')

# Load models globally (cache them)
try:
    LR_MODEL = joblib.load(os.path.join(MODELS_DIR, 'linear_regression_model.pkl'))
    KMEANS_MODEL = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    SCALER = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
except FileNotFoundError as e:
    print(f"Model loading error: {e}. Please ensure models are in {MODELS_DIR}")
    LR_MODEL = KMEANS_MODEL = SCALER = None


# ============== HELPER FUNCTIONS ==============
def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_base64_plot(fig):
    """Convert matplotlib figure to base64 string."""
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def predict_student_data(features_array):
    """Predict test marks and cluster for given features."""
    if LR_MODEL is None or SCALER is None or KMEANS_MODEL is None:
        raise ValueError("Models not loaded properly")
    
    # Predict test marks
    test_marks = LR_MODEL.predict(features_array)[0]
    
    # Scale features for clustering (only first 3 features)
    scaled_features = SCALER.transform(features_array[:, :3])
    cluster_label = KMEANS_MODEL.predict(scaled_features)[0]
    
    return float(test_marks), int(cluster_label)


def get_recommendation(study_hours, sleep_hours, social_media):
    """Generate personalized recommendation based on student habits."""
    if sleep_hours < 6:
        return "Increase your sleep hours to at least 7-8 hours for better cognitive function."
    elif social_media > 3:
        return "Reduce social media usage to under 3 hours daily to improve focus."
    elif study_hours < 4:
        return "Increase study hours gradually, aiming for 4-6 hours daily with breaks."
    else:
        return "Keep up your balanced habits! Maintain consistency for best results."


# ============== PLOT GENERATION FUNCTIONS ==============
def plot_clusters(df):
    """Generate cluster visualization scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot
    scatter = sns.scatterplot(
        x='StudyHours', 
        y='SleepHours', 
        hue='Cluster', 
        data=df, 
        palette='tab10',
        s=100,
        ax=ax
    )
    
    ax.set_title("Student Clusters Distribution", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Study Hours (per day)", fontsize=12)
    ax.set_ylabel("Sleep Hours (per day)", fontsize=12)
    ax.legend(title='Cluster', title_fontsize=11, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add average lines
    ax.axhline(y=df['SleepHours'].mean(), color='gray', linestyle='--', alpha=0.7, label=f"Avg Sleep: {df['SleepHours'].mean():.1f}h")
    ax.axvline(x=df['StudyHours'].mean(), color='gray', linestyle='--', alpha=0.7, label=f"Avg Study: {df['StudyHours'].mean():.1f}h")
    
    ax.legend()
    plt.close(fig)
    return generate_base64_plot(fig)


def plot_histogram(df):
    """Generate predicted test marks distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create histogram with KDE
    sns.histplot(df['PredictedTestMarks'], bins=15, kde=True, color='steelblue', alpha=0.7, ax=ax)
    
    # Add vertical line for mean
    mean_score = df['PredictedTestMarks'].mean()
    ax.axvline(x=mean_score, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_score:.1f}')
    
    ax.set_title("Predicted Test Marks Distribution", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Test Marks (Predicted)", fontsize=12)
    ax.set_ylabel("Number of Students", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.close(fig)
    return generate_base64_plot(fig)


def plot_recommendations(df):
    """Generate recommendations count plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Count recommendations and sort
    rec_counts = df['Recommendation'].value_counts()
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(rec_counts)), rec_counts.values, color='lightcoral', alpha=0.8)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, rec_counts.values)):
        ax.text(count + 0.5, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=10)
    
    ax.set_yticks(range(len(rec_counts)))
    ax.set_yticklabels([rec[:40] + "..." if len(rec) > 40 else rec for rec in rec_counts.index])
    ax.set_xlabel("Number of Students", fontsize=12)
    ax.set_title("Recommendations Distribution", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.close(fig)
    return generate_base64_plot(fig)


def index(request):
    """Main view for single student prediction and CSV upload."""
    context = {
        'prediction': None,
        'cluster_label': None,
        'recommendation': None,
        'csv_results': None,
        'cluster_graph': None,
        'plot_histogram_graph': None,
        'plot_recommendations_graph': None,
        'has_csv_data': False
    }
    
    if request.method == 'POST':
        # Single student submission
        if 'single_submit' in request.POST:
            try:
                # Get form data
                study_hours = float(request.POST.get('study_hours', 0))
                sleep_hours = float(request.POST.get('sleep_hours', 0))
                social_media = float(request.POST.get('social_media', 0))
                exercise = float(request.POST.get('exercise', 0))
                
                # Create features array
                student_features = np.array([[study_hours, sleep_hours, social_media, exercise]])
                
                # Make predictions
                prediction, cluster_label = predict_student_data(student_features)
                recommendation = get_recommendation(study_hours, sleep_hours, social_media)
                
                # Update context
                context.update({
                    'prediction': round(prediction, 2),
                    'cluster_label': cluster_label,
                    'recommendation': recommendation,
                    'form_data': request.POST
                })
                
            except ValueError as e:
                context['error'] = f"Invalid input: {str(e)}"
            except Exception as e:
                context['error'] = f"Prediction error: {str(e)}"
        
        # CSV upload processing
        elif 'csv_submit' in request.POST and 'csv_file' in request.FILES:
            try:
                csv_file = request.FILES['csv_file']
                
                # Validate file extension
                if not csv_file.name.endswith('.csv'):
                    context['error'] = "Please upload a CSV file"
                    return render(request, 'recommender/index.html', context)
                
                # Read and process CSV
                df = pd.read_csv(csv_file)
                required_columns = ['StudyHours', 'SleepHours', 'SocialMedia', 'Exercise']
                
                # Validate columns
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    context['error'] = f"Missing columns: {', '.join(missing_columns)}"
                    return render(request, 'recommender/index.html', context)
                
                results = []
                for _, row in df.iterrows():
                    # Create features array
                    features = np.array([[row['StudyHours'], row['SleepHours'], 
                                         row['SocialMedia'], row['Exercise']]])
                    
                    # Make predictions
                    test_marks, cluster = predict_student_data(features)
                    recommendation = get_recommendation(
                        row['StudyHours'], row['SleepHours'], row['SocialMedia']
                    )
                    
                    results.append({
                        'StudyHours': float(row['StudyHours']),
                        'SleepHours': float(row['SleepHours']),
                        'SocialMedia': float(row['SocialMedia']),
                        'Exercise': float(row['Exercise']),
                        'PredictedTestMarks': round(test_marks, 2),
                        'Cluster': cluster,
                        'Recommendation': recommendation
                    })
                
                # Create DataFrame for plotting
                df_results = pd.DataFrame(results)
                
                # Generate visualizations
                cluster_graph = plot_clusters(df_results)
                histogram_graph = plot_histogram(df_results)
                recommendations_graph = plot_recommendations(df_results)
                
                # Convert results for session storage
                clean_results = [{k: convert_numpy(v) for k, v in row.items()} 
                                for row in results]
                
                # Store in session
                request.session['csv_results'] = clean_results
                request.session['graphs'] = {
                    'cluster': cluster_graph,
                    'histogram': histogram_graph,
                    'recommendations': recommendations_graph,
                    'student_count': len(results)
                }
                
                # Update context
                context.update({
                    'csv_results': clean_results,
                    'cluster_graph': cluster_graph,
                    'plot_histogram_graph': histogram_graph,
                    'plot_recommendations_graph': recommendations_graph,
                    'has_csv_data': True,
                    'student_count': len(results)
                })
                
            except pd.errors.EmptyDataError:
                context['error'] = "The CSV file is empty"
            except pd.errors.ParserError:
                context['error'] = "Error parsing CSV file"
            except Exception as e:
                context['error'] = f"Error processing CSV: {str(e)}"
    
    return render(request, 'recommender/index.html', context)

def download_pdf(request):
    """Generate and download PDF report for CSV results."""
    # Get data from session
    csv_results = request.session.get('csv_results', [])
    graphs = request.session.get('graphs', {})
    
    if not csv_results:
        return HttpResponse(
            "<h3>No CSV data found.</h3><p>Please upload a CSV file first.</p>",
            content_type="text/html"
        )
    
    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#3498db'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    # Title
    elements.append(Paragraph("StudyTrack AI - Analysis Report", title_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#3498db')))
    elements.append(Spacer(1, 20))
    
    # Summary Section
    summary_text = f"""
    <b>Report Summary</b><br/>
    Total Students Analyzed: {len(csv_results)}<br/>
    Date Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}<br/>
    Average Predicted Score: {np.mean([r['PredictedTestMarks'] for r in csv_results]):.1f}
    """
    elements.append(Paragraph(summary_text, styles["Normal"]))
    elements.append(Spacer(1, 30))
    
    # Results Table
    elements.append(Paragraph("Student Performance Analysis", heading_style))
    
    # Prepare table data
    table_data = [['Study Hours', 'Sleep Hours', 'Social Media', 'Exercise', 
                   'Predicted Marks', 'Cluster', 'Recommendation']]
    
    for row in csv_results[:50]:  # Limit to first 50 for readability
        table_data.append([
            str(row['StudyHours']),
            str(row['SleepHours']),
            str(row['SocialMedia']),
            str(row['Exercise']),
            f"{row['PredictedTestMarks']:.1f}",
            f"Cluster {row['Cluster']}",
            row['Recommendation'][:30] + "..." if len(row['Recommendation']) > 30 else row['Recommendation']
        ])
    
    if len(csv_results) > 50:
        table_data.append(['...', '...', '...', '...', '...', '...', 
                          f'... and {len(csv_results)-50} more students'])
    
    # Create table
    table = Table(table_data, colWidths=[0.8*inch, 0.8*inch, 1*inch, 0.8*inch, 1*inch, 0.8*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 30))
    
    # Visualizations Section
    elements.append(Paragraph("Data Visualizations", heading_style))
    
    # Helper function to add images
    def add_image_to_pdf(base64_string, title, width=6*inch):
        if base64_string:
            elements.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
            elements.append(Spacer(1, 10))
            try:
                img_data = base64.b64decode(base64_string)
                img = Image(BytesIO(img_data), width=width, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 20))
            except Exception as e:
                elements.append(Paragraph(f"Error loading {title}: {str(e)}", styles["Normal"]))
    
    # Add all graphs
    add_image_to_pdf(graphs.get('cluster'), "1. Student Clusters Distribution")
    add_image_to_pdf(graphs.get('histogram'), "2. Predicted Marks Distribution")
    add_image_to_pdf(graphs.get('recommendations'), "3. Recommendations Analysis")
    
    # Recommendations Summary
    elements.append(Paragraph("Key Recommendations", heading_style))
    
    # Count recommendation types
    rec_counts = {}
    for row in csv_results:
        rec = row['Recommendation']
        rec_counts[rec] = rec_counts.get(rec, 0) + 1
    
    # Create recommendations list
    for rec, count in list(rec_counts.items())[:5]:  # Top 5 recommendations
        elements.append(Paragraph(
            f"• {rec} ({count} students)", 
            ParagraphStyle('CustomBullet', parent=styles["Normal"], leftIndent=20)
        ))
    
    elements.append(Spacer(1, 30))
    
    # Footer
    footer_text = """
    <i>Report generated by StudyTrack AI<br/>
    This analysis is based on machine learning models and should be used as guidance only.<br/>
    For personalized advice, consult with academic advisors.</i>
    """
    elements.append(Paragraph(footer_text, ParagraphStyle(
        'Footer', parent=styles["Normal"], fontSize=8, textColor=colors.grey
    )))
    
    # Build PDF
    doc.build(elements)
    
    # Prepare response
    buffer.seek(0)
    response = FileResponse(
        buffer, 
        as_attachment=True, 
        filename=f"StudyTrack_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
        content_type='application/pdf'
    )
    
    return response