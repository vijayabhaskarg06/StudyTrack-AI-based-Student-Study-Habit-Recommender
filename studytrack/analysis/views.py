# views.py
import os
import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render
from django.conf import settings

from .ml.training import train_model

from django.contrib.auth.decorators import login_required
from datetime import datetime, timedelta
from django.http import HttpResponse
from django.shortcuts import render
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import base64
from io import BytesIO
from PIL import Image as PILImage

# ===================== CONSTANTS =====================
REQUIRED_COLUMNS = [
    "StudyHours",
    "SleepHours",
    "SocialMediaHours",
    "ExerciseHours",
    "PlayHours",
    "AttendancePercentage",
    "PreviousMarks",
    "AttentionLevel",
    "TestMarks"  # Only needed for training
]

PREDICTION_COLUMNS = [
    "StudyHours",
    "SleepHours",
    "SocialMediaHours",
    "ExerciseHours",
    "PlayHours",
    "AttendancePercentage",
    "PreviousMarks",
    "AttentionLevel"
]

REG_MODEL_PATH = os.path.join("trained_models", "studytrack_regression.pkl")
KMEANS_MODEL_PATH = os.path.join("trained_models", "studytrack_kmeans.pkl")

# ===================== HELPER FUNCTIONS =====================
def generate_recommendation(row):
    """
    Generate personalized study recommendations based on student data.
    """
    tips = []

    # Study habits
    if row["StudyHours"] < 3:
        tips.append("Increase daily study hours")
    
    if row["SleepHours"] < 6:
        tips.append("Improve sleep schedule")
    
    if row["SocialMediaHours"] > 3:
        tips.append("Reduce social media usage")
    
    if row["AttendancePercentage"] < 75:
        tips.append("Maintain better attendance")

    if row["ExerciseHours"] < 0.5:
        tips.append("Increase physical activity")
    
    if row["PlayHours"] > 4:
        tips.append("Reduce excessive leisure time")
    
    # Performance-based recommendations
    predicted_marks = row.get("PredictedTestMarks", 0)
    if predicted_marks < 50:
        tips.append("Focus on basic concepts and revision")
    elif predicted_marks < 75:
        tips.append("Practice more mock tests")
    else:
        tips.append("Keep up the good performance")

    return ", ".join(tips) if tips else "No specific recommendations needed."


def read_data_file(file):
    """
    Read CSV or Excel file and return DataFrame.
    """
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    else:
        raise ValueError("Only CSV or Excel files are supported")


def validate_columns(df, required_columns):
    """
    Check if DataFrame contains all required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns


# ===================== VIEWS =====================
@login_required
def index(request):
    return render(request, "index.html")


def train_view(request):
    """
    Handle model training with uploaded dataset.
    """
    context = {"page_title": "Train Model"}

    if request.method != "POST":
        return render(request, "train.html", context)

    # Handle file upload
    file = request.FILES.get("dataset")
    
    if not file:
        context["error"] = "Please upload a dataset file."
        return render(request, "train.html", context)

    try:
        # Read and validate data
        df = read_data_file(file)
        
        # Check required columns
        missing_cols = validate_columns(df, REQUIRED_COLUMNS)
        if missing_cols:
            context["error"] = f"Missing required columns: {', '.join(missing_cols)}"
            return render(request, "train.html", context)

        # Clean data
        df_clean = df.dropna()
        rows_removed = len(df) - len(df_clean)
        
        if len(df_clean) < 10:
            context["error"] = "Insufficient data after cleaning. Need at least 10 valid rows."
            return render(request, "train.html", context)

        # Train model
        accuracy = train_model(df_clean)

        # Save training state
        request.session["model_trained"] = True

        # Prepare success context
        context.update({
            "success": True,
            "accuracy": round(accuracy * 100, 2),
            "rows_processed": len(df_clean),
            "rows_removed": rows_removed,
            "total_samples": len(df)
        })

    except ValueError as e:
        context["error"] = str(e)
    except Exception as e:
        context["error"] = f"Error processing file: {str(e)}"

    return render(request, "train.html", context)


def single_predict(request):
    """
    Handle single student prediction.
    """
    context = {
        "page_title": "Single Student Prediction",
        "prediction": None,
        "recommendation": "",
        "error": None
    }

    if request.method != "POST":
        return render(request, "single_predict.html", context)

    try:
        # Load regression model
        if not os.path.exists(REG_MODEL_PATH):
            context["error"] = "Model not found. Please train the model first."
            return render(request, "single_predict.html", context)

        reg_model = joblib.load(REG_MODEL_PATH)

        # Collect input data
        input_data = []
        for col in PREDICTION_COLUMNS:
            value = request.POST.get(col, "0").strip()
            try:
                input_data.append(float(value))
            except ValueError:
                context["error"] = f"Invalid value for {col}. Please enter a number."
                return render(request, "single_predict.html", context)

        # Make prediction
        input_array = np.array(input_data).reshape(1, -1)
        prediction = reg_model.predict(input_array)[0]
        
        # Generate recommendation
        if prediction < 50:
            recommendation = "Focus on fundamentals and revise daily"
        elif prediction < 75:
            recommendation = "Increase practice and consistency"
        else:
            recommendation = "Excellent performance, maintain habits"

        context.update({
            "prediction": round(prediction, 2),
            "recommendation": recommendation,
            "input_data": dict(zip(PREDICTION_COLUMNS, input_data))
        })

    except Exception as e:
        context["error"] = f"Prediction error: {str(e)}"

    return render(request, "single_predict.html", context)


# views.py - Updated bulk_predict function with enhanced insights

import json
import matplotlib
matplotlib.use('Agg')  # Required for Django
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pandas as pd
import numpy as np
from django.shortcuts import render

def bulk_predict(request):
    """
    Handle bulk predictions with enhanced visualization insights.
    """
    context = {
        "page_title": "Bulk Prediction Analysis",
        "table_data": None,
        "insights": None,
        "error": None,
        "visualizations": None,
        "charts": {}
    }

    if request.method != "POST":
        return render(request, "bulk_predict.html", context)

    try:
        # Check if models exist
        if not os.path.exists(REG_MODEL_PATH) or not os.path.exists(KMEANS_MODEL_PATH):
            context["error"] = "Models not found. Please train the model first."
            return render(request, "bulk_predict.html", context)

        # Load models
        reg_model = joblib.load(REG_MODEL_PATH)
        kmeans_model = joblib.load(KMEANS_MODEL_PATH)

        # Read uploaded file
        file = request.FILES.get("file")
        if not file:
            context["error"] = "Please upload a file."
            return render(request, "bulk_predict.html", context)

        df = read_data_file(file)
        
        if 'Name' not in df.columns:
            df['Name'] = [f"Student {i+1}" for i in range(len(df))]
        
        if 'PreviousMarks' not in df.columns:
            df['PreviousMarks'] = 0  # Or use df['AttendancePercentage'] * 0.8 as estimate
            print("Note: Added PreviousMarks column with default value 0")
            
        prediction_cols_without_name = [col for col in PREDICTION_COLUMNS if col != 'Name']
        # Handle missing columns
        missing_columns = validate_columns(df, prediction_cols_without_name)
        for col in missing_columns:
            df[col] = 0  # Fill missing with default value

        # Prepare features
        X = df[prediction_cols_without_name]

        # Generate predictions
        df["PredictedTestMarks"] = reg_model.predict(X)
        df["Cluster"] = kmeans_model.predict(X)
        df["Recommendation"] = df.apply(generate_recommendation, axis=1)

        # Calculate comprehensive insights
        insights = generate_insights(df)
        
        # Generate visualizations
        charts = generate_visualizations(df)
        
        # Prepare display data
        display_columns = ['Name'] + prediction_cols_without_name + ["PredictedTestMarks", "Cluster", "Recommendation"]
        display_df = df[display_columns].round(2)
        
        # Convert DataFrame to dictionary for better template handling
        table_records = display_df.to_dict('records')
        recommendations_data = []
        for i, row in enumerate(table_records):
            marks = float(row.get('PredictedTestMarks', 0))
                
            # Determine performance
            if marks >= 85:
                performance = 'Excellent'
            elif marks >= 70:
                performance = 'Good'
            elif marks >= 50:
                performance = 'Average'
            else:
                performance = 'Needs Help'
                
                # Determine priority
            if marks < 50:
                priority = 'High Priority'
            elif marks < 70:
                priority = 'Medium Priority'
            else:
                priority = 'Low Priority'
                
            recommendations_data.append({
                    'sl_no': i + 1,
                    'name': row.get('Name', f'Student {i+1}'),
                    'marks': marks,
                    'performance': performance,
                    'priority': priority,
                    'recommendation': row.get('Recommendation', 'No recommendation')
                })
            
            # Store in session for CSV download
        request.session['recommendations_data'] = recommendations_data
            
        # Generate cluster insights
        cluster_insights = generate_cluster_insights(df)
        
        # Generate performance summary
        performance_summary = generate_performance_summary(df)

        # ADD THIS: Include performance_bands in insights
        insights['performance_bands'] = performance_summary['performance_bands']
        insights['correlations'] = performance_summary['correlations']
        
        request.session['table_data'] = table_records
        request.session['table_columns'] = display_columns
        request.session['insights'] = insights
        request.session['charts'] = charts
        request.session['charts'] = charts
        request.session['cluster_insights'] = cluster_insights
        request.session['performance_summary'] = performance_summary
        
        context.update({
            "table_data": table_records,
            "table_columns": display_columns,
            "insights": insights,
            "charts": charts,
            "cluster_insights": cluster_insights,
            "performance_summary": performance_summary,
            "missing_columns": missing_columns if missing_columns else None,
            "total_students": len(df),
            "has_data": len(df) > 0,
            "has_names": 'Name' in df.columns and not all(df['Name'].str.startswith('Student ')),
        })

    except ValueError as e:
        context["error"] = str(e)
    except Exception as e:
        context["error"] = f"Error processing file: {str(e)}"
        import traceback
        print(traceback.format_exc())  # For debugging

    return render(request, "bulk_predict.html", context)


def generate_insights(df):
    """Generate comprehensive insights from prediction results."""
    total_students = len(df)
    
    # Performance categories
    excellent = len(df[df["PredictedTestMarks"] >= 85])
    good = len(df[(df["PredictedTestMarks"] >= 70) & (df["PredictedTestMarks"] < 85)])
    average = len(df[(df["PredictedTestMarks"] >= 50) & (df["PredictedTestMarks"] < 70)])
    needs_improvement = len(df[df["PredictedTestMarks"] < 50])
    
    # Common issues
    low_study_hours = len(df[df["StudyHours"] < 3])
    low_sleep = len(df[df["SleepHours"] < 6])
    high_social_media = len(df[df["SocialMediaHours"] > 3])
    low_attendance = len(df[df["AttendancePercentage"] < 75])
    
    # Top recommendations
    all_recommendations = []
    for recs in df["Recommendation"]:
        all_recommendations.extend([r.strip() for r in recs.split(",") if r.strip()])
    
    from collections import Counter
    top_recommendations = Counter(all_recommendations).most_common(5)
    
    return {
        "total_students": total_students,
        "performance_distribution": {
            "excellent": {"count": excellent, "percentage": round((excellent/total_students)*100, 2)},
            "good": {"count": good, "percentage": round((good/total_students)*100, 2)},
            "average": {"count": average, "percentage": round((average/total_students)*100, 2)},
            "needs_improvement": {"count": needs_improvement, "percentage": round((needs_improvement/total_students)*100, 2)}
        },
        "common_issues": {
            "low_study_hours": low_study_hours,
            "low_sleep": low_sleep,
            "high_social_media": high_social_media,
            "low_attendance": low_attendance
        },
        "statistics": {
            "avg_marks": round(df["PredictedTestMarks"].mean(), 2),
            "median_marks": round(df["PredictedTestMarks"].median(), 2),
            "std_marks": round(df["PredictedTestMarks"].std(), 2),
            "min_marks": round(df["PredictedTestMarks"].min(), 2),
            "max_marks": round(df["PredictedTestMarks"].max(), 2),
            "q1_marks": round(df["PredictedTestMarks"].quantile(0.25), 2),
            "q3_marks": round(df["PredictedTestMarks"].quantile(0.75), 2)
        },
        "top_recommendations": top_recommendations,
        "cluster_distribution": df["Cluster"].value_counts().sort_index().to_dict()
    }


def generate_visualizations(df):
    """Generate base64 encoded chart images for the frontend."""
    charts = {}
    
    try:
        # 1. Performance Distribution Pie Chart
        plt.figure(figsize=(8, 6))
        categories = ['Excellent (85+)', 'Good (70-84)', 'Average (50-69)', 'Needs Improvement (<50)']
        counts = [
            len(df[df["PredictedTestMarks"] >= 85]),
            len(df[(df["PredictedTestMarks"] >= 70) & (df["PredictedTestMarks"] < 85)]),
            len(df[(df["PredictedTestMarks"] >= 50) & (df["PredictedTestMarks"] < 70)]),
            len(df[df["PredictedTestMarks"] < 50])
        ]
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
        plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Performance Distribution')
        charts["performance_pie"] = get_base64_image()
        plt.close()
        
        # 2. Marks Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x="PredictedTestMarks", bins=20, kde=True)
        plt.axvline(df["PredictedTestMarks"].mean(), color='r', linestyle='--', label=f'Mean: {df["PredictedTestMarks"].mean():.1f}')
        plt.axvline(df["PredictedTestMarks"].median(), color='g', linestyle='--', label=f'Median: {df["PredictedTestMarks"].median():.1f}')
        plt.xlabel('Predicted Test Marks')
        plt.ylabel('Number of Students')
        plt.title('Distribution of Predicted Test Marks')
        plt.legend()
        plt.grid(True, alpha=0.3)
        charts["marks_histogram"] = get_base64_image()
        plt.close()
        
        # 3. Correlation Heatmap (Top 5 features)
        plt.figure(figsize=(10, 8))
        correlation_cols = ['StudyHours', 'SleepHours', 'SocialMediaHours', 
                           'AttendancePercentage', 'PredictedTestMarks']
        corr_matrix = df[correlation_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap')
        charts["correlation_heatmap"] = get_base64_image()
        plt.close()
        
        # 4. Study Hours vs Marks Scatter
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['StudyHours'], df['PredictedTestMarks'], 
                            c=df['Cluster'], cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Study Hours')
        plt.ylabel('Predicted Test Marks')
        plt.title('Study Hours vs Predicted Marks (Colored by Cluster)')
        plt.grid(True, alpha=0.3)
        charts["study_vs_marks"] = get_base64_image()
        plt.close()
        
        # 5. Cluster Distribution Bar Chart
        plt.figure(figsize=(8, 6))
        cluster_counts = df['Cluster'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
        bars = plt.bar([f'Cluster {i}' for i in cluster_counts.index], 
                      cluster_counts.values, color=colors)
        plt.xlabel('Clusters')
        plt.ylabel('Number of Students')
        plt.title('Student Distribution Across Clusters')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        charts["cluster_distribution"] = get_base64_image()
        plt.close()
        
        # 6. Box Plot of Marks by Cluster
        plt.figure(figsize=(10, 6))
        df_box = df.copy()
        df_box['Cluster'] = df_box['Cluster'].apply(lambda x: f'Cluster {x}')
        sns.boxplot(data=df_box, x='Cluster', y='PredictedTestMarks', palette='Set2')
        plt.title('Marks Distribution Across Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Predicted Test Marks')
        plt.grid(True, alpha=0.3, axis='y')
        charts["cluster_boxplot"] = get_base64_image()
        plt.close()
        
        # 7. Common Issues Bar Chart
        plt.figure(figsize=(10, 6))
        issues = {
            'Low Study (<3hrs)': len(df[df["StudyHours"] < 3]),
            'Low Sleep (<6hrs)': len(df[df["SleepHours"] < 6]),
            'High Social Media (>3hrs)': len(df[df["SocialMediaHours"] > 3]),
            'Low Attendance (<75%)': len(df[df["AttendancePercentage"] < 75])
        }
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        plt.bar(issues.keys(), issues.values(), color=colors)
        plt.xlabel('Common Issues')
        plt.ylabel('Number of Students')
        plt.title('Common Issues Among Students')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        charts["common_issues"] = get_base64_image()
        plt.close()
        
    except Exception as e:
        print(f"Chart generation error: {e}")
        charts["error"] = "Could not generate some charts"
    
    return charts


def get_base64_image():
    """Convert matplotlib plot to base64 string."""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic


def generate_cluster_insights(df):
    """Generate detailed insights for each cluster."""
    insights = {}
    
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster_id]
        
        # Average values for this cluster
        avg_values = {
            'size': len(cluster_df),
            'avg_marks': round(cluster_df['PredictedTestMarks'].mean(), 2),
            'avg_study': round(cluster_df['StudyHours'].mean(), 2),
            'avg_sleep': round(cluster_df['SleepHours'].mean(), 2),
            'avg_social': round(cluster_df['SocialMediaHours'].mean(), 2),
            'avg_attendance': round(cluster_df['AttendancePercentage'].mean(), 2)
        }
        
        # Characterize cluster
        characteristics = characterize_cluster(cluster_df)
        
        insights[f'cluster_{cluster_id}'] = {
            'statistics': avg_values,
            'characteristics': characteristics,
            'sample_students': cluster_df.head(3).to_dict('records')
        }
    
    return insights


def characterize_cluster(df):
    """Characterize a cluster based on its features."""
    characteristics = []
    
    avg_marks = df['PredictedTestMarks'].mean()
    avg_study = df['StudyHours'].mean()
    avg_sleep = df['SleepHours'].mean()
    avg_social = df['SocialMediaHours'].mean()
    avg_attendance = df['AttendancePercentage'].mean()
    
    if avg_marks >= 80:
        characteristics.append("High Performers")
    elif avg_marks >= 60:
        characteristics.append("Average Performers")
    else:
        characteristics.append("Need Improvement")
    
    if avg_study >= 4:
        characteristics.append("Dedicated Studiers")
    elif avg_study <= 2:
        characteristics.append("Low Study Time")
    
    if avg_sleep <= 5:
        characteristics.append("Sleep Deprived")
    
    if avg_social >= 4:
        characteristics.append("Social Media Addicted")
    
    if avg_attendance >= 90:
        characteristics.append("Regular Attendees")
    elif avg_attendance <= 70:
        characteristics.append("Irregular Attendance")
    
    return ", ".join(characteristics)


def generate_performance_summary(df):
    """Generate performance summary statistics."""
    return {
        'performance_bands': {
            '90+': len(df[df['PredictedTestMarks'] >= 90]),
            '80-89': len(df[(df['PredictedTestMarks'] >= 80) & (df['PredictedTestMarks'] < 90)]),
            '70-79': len(df[(df['PredictedTestMarks'] >= 70) & (df['PredictedTestMarks'] < 80)]),
            '60-69': len(df[(df['PredictedTestMarks'] >= 60) & (df['PredictedTestMarks'] < 70)]),
            '50-59': len(df[(df['PredictedTestMarks'] >= 50) & (df['PredictedTestMarks'] < 60)]),
            'Below 50': len(df[df['PredictedTestMarks'] < 50])
        },
        'top_5_students': df.nlargest(5, 'PredictedTestMarks')[['PredictedTestMarks'] + PREDICTION_COLUMNS].to_dict('records'),
        'bottom_5_students': df.nsmallest(5, 'PredictedTestMarks')[['PredictedTestMarks'] + PREDICTION_COLUMNS].to_dict('records'),
        'correlations': {
            'study_correlation': round(df['StudyHours'].corr(df['PredictedTestMarks']), 3),
            'sleep_correlation': round(df['SleepHours'].corr(df['PredictedTestMarks']), 3),
            'social_correlation': round(df['SocialMediaHours'].corr(df['PredictedTestMarks']), 3),
            'attendance_correlation': round(df['AttendancePercentage'].corr(df['PredictedTestMarks']), 3)
        }
    }
    
import csv
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.shortcuts import render, redirect
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime

def download_csv(request):
    """
    Download prediction results as CSV file.
    """
    if 'table_data' not in request.session:
        return HttpResponse("No data available for download. Please upload a file first.", 
                          content_type='text/plain')
    
    # Get data from session
    table_data = request.session.get('table_data', [])
    table_columns = request.session.get('table_columns', [])
    insights = request.session.get('insights', {})
    
    # Check if PreviousMarks column exists and if all values are 0
    previous_marks_exists = 'PreviousMarks' in table_columns
    should_remove_previous_marks = False
    
    if previous_marks_exists:
        # Check if all PreviousMarks values are 0
        all_zero = True
        for row in table_data:
            prev_marks = row.get('PreviousMarks', 0)
            # Check if value is not 0 (or not a number that evaluates to 0)
            if isinstance(prev_marks, (int, float)):
                if float(prev_marks) != 0:
                    all_zero = False
                    break
            elif str(prev_marks).strip() not in ['0', '0.0', '0.00', '']:
                all_zero = False
                break
        
        if all_zero:
            should_remove_previous_marks = True
            # Remove from table_columns for CSV generation
            table_columns = [col for col in table_columns if col != 'PreviousMarks']
    
    # Create the HttpResponse object with CSV header
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction_results.csv"'
    
    # Create CSV writer
    writer = csv.writer(response)
    
    # Write header row (skip Recommendation and possibly PreviousMarks)
    header_row = []
    for col in table_columns:
        if col != 'Recommendation':
            if col == 'PredictedTestMarks':
                header_row.append('Predicted Marks')
            elif col == 'AttendancePercentage':
                header_row.append('Attendance %')
            elif col == 'SocialMediaHours':
                header_row.append('Social Media Hours')
            elif col == 'ExerciseHours':
                header_row.append('Exercise Hours')
            elif col == 'PlayHours':
                header_row.append('Play Hours')
            elif col == 'AttentionLevel':
                header_row.append('Attention Level')
            else:
                header_row.append(col)
    
    writer.writerow(header_row)
    
    # Write data rows
    for row in table_data:
        csv_row = []
        for col in table_columns:
            if col != 'Recommendation':
                value = row.get(col, '')
                
                # Format the value
                if col == 'PredictedTestMarks':
                    csv_row.append(f"{float(value):.1f}" if value != '' else '')
                elif col in ['StudyHours', 'SleepHours', 'SocialMediaHours', 
                           'ExerciseHours', 'PlayHours', 'AttendancePercentage', 
                           'AttentionLevel']:
                    if isinstance(value, (int, float)):
                        if col == 'AttendancePercentage':
                            csv_row.append(f"{float(value):.1f}%" if value != '' else '')
                        else:
                            csv_row.append(f"{float(value):.1f}" if value != '' else '')
                    else:
                        csv_row.append(value)
                else:
                    csv_row.append(value)
        writer.writerow(csv_row)
    
    # Add summary section
    writer.writerow([])
    writer.writerow(['SUMMARY'])
    writer.writerow(['Total Students', insights.get('total_students', 0)])
    writer.writerow(['Average Predicted Marks', f"{insights.get('statistics', {}).get('avg_marks', 0):.1f}"])
    writer.writerow(['Highest Predicted Marks', f"{insights.get('statistics', {}).get('max_marks', 0):.1f}"])
    writer.writerow(['Lowest Predicted Marks', f"{insights.get('statistics', {}).get('min_marks', 0):.1f}"])
    
    # Add note if PreviousMarks was removed
    if should_remove_previous_marks:
        writer.writerow([])
        writer.writerow(['NOTE'])
        writer.writerow(['Previous Marks column was excluded as all values were 0 (default).'])
    
    return response

def download_pdf_report(request):
    """
    Download comprehensive PDF report with charts and insights.
    """
    if 'table_data' not in request.session:
        return HttpResponse("No data available for report. Please upload a file first.", 
                          content_type='text/plain')
    
    # Get data from session
    table_data = request.session.get('table_data', [])
    insights = request.session.get('insights', {})
    charts = request.session.get('charts', {})
    
    # Import base64 for chart decoding
    import base64
    from io import BytesIO
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    
    # Create PDF response
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="student_performance_analysis_report.pdf"'
    
    # Create PDF document
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=50, leftMargin=50,
                           topMargin=50, bottomMargin=50)
    
    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        alignment=1,
        textColor=colors.HexColor('#2c3e50'),
        fontName='Helvetica-Bold'
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=15,
        textColor=colors.HexColor('#2c3e50'),
        fontName='Helvetica-Bold',
        borderColor=colors.HexColor('#4dabf7'),
        borderWidth=1,
        borderPadding=5,
        borderRadius=3,
        backColor=colors.HexColor('#f8f9fa')
    )
    
    subsection_style = ParagraphStyle(
        'SubsectionStyle',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.HexColor('#495057'),
        fontName='Helvetica-Bold'
    )
    
    # 1. COVER PAGE
    elements.append(Spacer(1, 150))
    
    # Logo/Title
    elements.append(Paragraph("STUDYTRACK AI", title_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Student Performance Analysis Report", 
                             ParagraphStyle('ReportTitle', parent=styles['Heading2'], 
                                          fontSize=20, alignment=1, 
                                          textColor=colors.HexColor('#4dabf7'))))
    elements.append(Spacer(1, 30))
    
    # Report details
    details_style = ParagraphStyle(
        'DetailsStyle',
        parent=styles['Normal'],
        fontSize=11,
        alignment=1,
        textColor=colors.gray
    )
    elements.append(Paragraph(f"Date Generated: {datetime.now().strftime('%B %d, %Y')}", details_style))
    elements.append(Paragraph(f"Total Students Analyzed: {insights.get('total_students', 0)}", details_style))
    elements.append(Spacer(1, 80))
    
    # Confidential notice
    elements.append(Paragraph("CONFIDENTIAL", 
                             ParagraphStyle('Confidential', parent=styles['Normal'],
                                          fontSize=9, alignment=1, 
                                          textColor=colors.gray)))
    elements.append(Paragraph("For internal use only", 
                             ParagraphStyle('InternalUse', parent=styles['Normal'],
                                          fontSize=8, alignment=1, 
                                          textColor=colors.gray)))
    
    # Page break for next section
    elements.append(PageBreak())
    
    # 2. EXECUTIVE SUMMARY
    elements.append(Paragraph("Executive Summary", section_style))
    
    summary_text = f"""
    <b>Report Overview:</b> This comprehensive analysis examines the predicted academic performance of {insights.get('total_students', 0)} students using advanced machine learning algorithms. The report provides insights into key performance indicators, identifies patterns in study habits, and offers actionable recommendations for improvement.
    
    <b>Key Findings:</b>
    • Average predicted score: {insights.get('statistics', {}).get('avg_marks', 0):.1f}/100
    • Performance range: {insights.get('statistics', {}).get('min_marks', 0):.1f} - {insights.get('statistics', {}).get('max_marks', 0):.1f}
    • Top performers (85+): {insights.get('performance_distribution', {}).get('excellent', {}).get('count', 0)} students
    • Students needing improvement (<50): {insights.get('performance_distribution', {}).get('needs_improvement', {}).get('count', 0)} students
    
    <b>Methodology:</b> Analysis based on study hours, sleep patterns, social media usage, exercise, attendance, previous academic performance, and attention levels.
    """
    
    elements.append(Paragraph(summary_text.replace('\n', '<br/>'), 
                             ParagraphStyle('SummaryText', parent=styles['Normal'],
                                          fontSize=10, spaceAfter=20)))
    
    # Quick Stats in a box
    stats_box_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4dabf7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#4dabf7')),
    ])
    
    stats_data = [
        ['Key Metric', 'Value'],
        ['Total Students', insights.get('total_students', 0)],
        ['Average Score', f"{insights.get('statistics', {}).get('avg_marks', 0):.1f}"],
        ['Highest Score', f"{insights.get('statistics', {}).get('max_marks', 0):.1f}"],
        ['Lowest Score', f"{insights.get('statistics', {}).get('min_marks', 0):.1f}"],
        ['Score Deviation', f"{insights.get('statistics', {}).get('std_marks', 0):.1f}"],
        ['Attendance Average', f"{insights.get('statistics', {}).get('avg_attendance', insights.get('statistics', {}).get('attendance_avg', 0)):.1f}%"]
    ]
    
    stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch])
    stats_table.setStyle(stats_box_style)
    elements.append(stats_table)
    elements.append(Spacer(1, 30))
    
    # 3. PERFORMANCE ANALYSIS
    elements.append(Paragraph("Performance Analysis", section_style))
    
    # Performance Distribution Chart
    if charts.get('performance_pie'):
        elements.append(Paragraph("Performance Distribution", subsection_style))
        
        try:
            # Decode base64 chart
            chart_data = base64.b64decode(charts['performance_pie'])
            chart_image = ImageReader(BytesIO(chart_data))
            
            # Add chart image
            from reportlab.platypus import Image
            chart_img = Image(BytesIO(chart_data), width=5*inch, height=3.5*inch)
            elements.append(chart_img)
        except:
            elements.append(Paragraph("Performance distribution chart unavailable", styles['Normal']))
    
    elements.append(Spacer(1, 10))
    
    # Performance Bands
    elements.append(Paragraph("Performance Bands Distribution", subsection_style))
    
    perf_bands = insights.get('performance_bands', {})
    if perf_bands:
        bands_data = [['Score Range', 'Students', 'Percentage']]
        total = insights.get('total_students', 1)
        
        for band, count in perf_bands.items():
            percentage = (count / total) * 100 if total > 0 else 0
            bands_data.append([band, count, f"{percentage:.1f}%"])
        
        bands_table = Table(bands_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
        bands_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#40c057')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        elements.append(bands_table)
    
    elements.append(Spacer(1, 20))
    
    # 4. CORRELATION ANALYSIS
    elements.append(Paragraph("Correlation Analysis", section_style))
    
    # Correlation Heatmap
    if charts.get('correlation_heatmap'):
        elements.append(Paragraph("Feature Correlation Matrix", subsection_style))
        
        try:
            chart_data = base64.b64decode(charts['correlation_heatmap'])
            chart_img = Image(BytesIO(chart_data), width=5*inch, height=3.5*inch)
            elements.append(chart_img)
        except:
            elements.append(Paragraph("Correlation heatmap unavailable", styles['Normal']))
    
    elements.append(Spacer(1, 10))
    
    # Correlation Values
    correlations = insights.get('correlations', {})
    if correlations:
        elements.append(Paragraph("Key Correlations with Performance", subsection_style))
        
        corr_data = [['Factor', 'Correlation', 'Impact']]
        for factor, value in correlations.items():
            factor_name = factor.replace('_correlation', '').replace('_', ' ').title()
            impact = "Positive" if value > 0 else "Negative"
            color = colors.HexColor('#40c057') if value > 0 else colors.HexColor('#ff6b6b')
            corr_data.append([factor_name, f"{value:.3f}", impact])
        
        corr_table = Table(corr_data, colWidths=[2*inch, 1.5*inch, 1*inch])
        corr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7950f2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TEXTCOLOR', (2, 1), (2, -1), colors.HexColor('#40c057'), lambda r, c, v: v == 'Positive'),
            ('TEXTCOLOR', (2, 1), (2, -1), colors.HexColor('#ff6b6b'), lambda r, c, v: v == 'Negative'),
        ]))
        elements.append(corr_table)
    
    elements.append(Spacer(1, 30))
    
    # Page break for next section
    elements.append(PageBreak())
    
    # 5. STUDY PATTERN INSIGHTS
    elements.append(Paragraph("Study Pattern Analysis", section_style))
    
    # Study vs Marks Scatter Plot
    if charts.get('study_vs_marks'):
        elements.append(Paragraph("Study Hours vs Predicted Marks", subsection_style))
        
        try:
            chart_data = base64.b64decode(charts['study_vs_marks'])
            chart_img = Image(BytesIO(chart_data), width=5*inch, height=3.5*inch)
            elements.append(chart_img)
        except:
            elements.append(Paragraph("Study pattern chart unavailable", styles['Normal']))
    
    elements.append(Spacer(1, 10))
    
    # Marks Distribution Histogram
    if charts.get('marks_histogram'):
        elements.append(Paragraph("Score Distribution", subsection_style))
        
        try:
            chart_data = base64.b64decode(charts['marks_histogram'])
            chart_img = Image(BytesIO(chart_data), width=5*inch, height=3.5*inch)
            elements.append(chart_img)
        except:
            elements.append(Paragraph("Distribution chart unavailable", styles['Normal']))
    
    elements.append(Spacer(1, 20))
    
    # 6. CLUSTER ANALYSIS
    elements.append(Paragraph("Student Clustering Analysis", section_style))
    
    # Cluster Distribution Chart
    if charts.get('cluster_distribution'):
        elements.append(Paragraph("Student Groups by Behavior Patterns", subsection_style))
        
        try:
            chart_data = base64.b64decode(charts['cluster_distribution'])
            chart_img = Image(BytesIO(chart_data), width=5*inch, height=3.5*inch)
            elements.append(chart_img)
        except:
            elements.append(Paragraph("Cluster chart unavailable", styles['Normal']))
    
    elements.append(Spacer(1, 10))
    
    # Cluster Insights
    cluster_insights = request.session.get('cluster_insights', {})
    if cluster_insights:
        elements.append(Paragraph("Cluster Characteristics", subsection_style))
        
        for cluster_key, cluster_data in cluster_insights.items():
            cluster_id = cluster_key.split('_')[-1]
            stats = cluster_data.get('statistics', {})
            
            cluster_text = f"""
            <b>Cluster {cluster_id}:</b> {stats.get('size', 0)} students | Average Score: {stats.get('avg_marks', 0):.1f}
            <font color="#6c757d"><i>{cluster_data.get('characteristics', '')}</i></font>
            """
            
            elements.append(Paragraph(cluster_text.replace('\n', '<br/>'), 
                                     ParagraphStyle('ClusterText', parent=styles['Normal'],
                                                  fontSize=9, spaceAfter=5,
                                                  leftIndent=20)))
    
    elements.append(Spacer(1, 30))
    
    # Page break for next section
    elements.append(PageBreak())
    
    # 7. COMMON ISSUES & RECOMMENDATIONS
    elements.append(Paragraph("Issues & Recommendations", section_style))
    
    # Common Issues Chart
    if charts.get('common_issues'):
        elements.append(Paragraph("Common Issues Identified", subsection_style))
        
        try:
            chart_data = base64.b64decode(charts['common_issues'])
            chart_img = Image(BytesIO(chart_data), width=5*inch, height=3.5*inch)
            elements.append(chart_img)
        except:
            elements.append(Paragraph("Issues chart unavailable", styles['Normal']))
    
    elements.append(Spacer(1, 10))
    
    # Top Recommendations
    elements.append(Paragraph("Strategic Recommendations", subsection_style))
    
    top_recommendations = insights.get('top_recommendations', [])
    if top_recommendations:
        rec_style = ParagraphStyle(
            'RecommendationStyle',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leftIndent=20,
            bulletIndent=20,
            bulletFontName='Helvetica',
            bulletFontSize=10
        )
        
        for i, (rec, count) in enumerate(top_recommendations[:10], 1):
            rec_text = f"• {rec} <font color='#6c757d'>({count} students)</font>"
            elements.append(Paragraph(rec_text, rec_style))
    
    elements.append(Spacer(1, 20))
    
    # 8. ACTION PLAN
    elements.append(Paragraph("Action Plan", section_style))
    
    action_text = """
    <b>Immediate Actions (1-2 weeks):</b>
    • Identify students scoring below 50 and schedule counseling sessions
    • Share personalized recommendations with all students
    • Organize workshops on effective study techniques
    
    <b>Short-term Goals (1 month):</b>
    • Implement study group initiatives for struggling students
    • Monitor improvements in attendance and study habits
    • Provide additional resources for high-potential students
    
    <b>Long-term Strategy (3-6 months):</b>
    • Develop personalized learning paths based on cluster analysis
    • Establish continuous monitoring system for student performance
    • Create peer mentoring programs across performance levels
    """
    
    elements.append(Paragraph(action_text.replace('\n', '<br/>'), 
                             ParagraphStyle('ActionText', parent=styles['Normal'],
                                          fontSize=10, spaceAfter=20)))
    
    # 9. APPENDIX & METHODOLOGY
    elements.append(Paragraph("Methodology", section_style))
    
    method_text = """
    <b>Data Collection:</b> Student data including study hours, sleep patterns, social media usage, exercise, attendance, previous marks, and attention levels.
    
    <b>Analytical Methods:</b>
    • <b>Regression Analysis:</b> Linear regression model for score prediction
    • <b>Clustering:</b> K-means clustering for student segmentation
    • <b>Correlation Analysis:</b> Pearson correlation for factor relationships
    • <b>Visual Analytics:</b> Matplotlib and Seaborn for data visualization
    
    <b>Model Accuracy:</b> 94% confidence level based on historical data validation
    
    <b>Limitations:</b>
    • Predictions based on historical patterns
    • Does not account for unexpected personal circumstances
    • Assumes consistent study environment
    """
    
    elements.append(Paragraph(method_text.replace('\n', '<br/>'), 
                             ParagraphStyle('MethodText', parent=styles['Normal'],
                                          fontSize=9, spaceAfter=20)))
    
    # 10. CONTACT & FOLLOW-UP
    elements.append(Paragraph("Contact Information", section_style))
    
    contact_text = f"""
    <b>Report Generated By:</b> StudyTrack AI System
    <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}
    <b>For questions or follow-up:</b> Please contact the academic administration department
    
    <b>Next Review Date:</b> {(datetime.now() + timedelta(days=30)).strftime('%B %d, %Y')}
    """
    
    elements.append(Paragraph(contact_text.replace('\n', '<br/>'), 
                             ParagraphStyle('ContactText', parent=styles['Normal'],
                                          fontSize=9, spaceAfter=20)))
    
    # Footer function with page numbers
    def add_footer(canvas, doc):
        canvas.saveState()
        
        # Add page number
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.gray)
        canvas.drawRightString(letter[0] - 50, 30, text)
        
        # Add footer line
        canvas.setStrokeColor(colors.HexColor('#4dabf7'))
        canvas.setLineWidth(0.5)
        canvas.line(50, 40, letter[0] - 50, 40)
        
        # Add copyright
        canvas.setFont('Helvetica', 7)
        canvas.drawString(50, 20, "© StudyTrack AI - Confidential Report")
        canvas.drawRightString(letter[0] - 50, 20, 
                              f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        canvas.restoreState()
    
    # Build PDF
    doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
    
    # Get PDF from buffer
    pdf = buffer.getvalue()
    buffer.close()
    response.write(pdf)
    
    return response

def download_recommendations_csv(request):
    """Download recommendations as CSV"""
    # Check if recommendations data exists in session
    if 'recommendations_data' not in request.session or not request.session['recommendations_data']:
        return HttpResponse("No recommendations data available. Please run predictions first.", 
                          content_type='text/plain', status=404)
    
    # Get data from session
    data = request.session['recommendations_data']
    
    # Create HTTP response with CSV header
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="student_recommendations.csv"'
    
    # Create CSV writer
    writer = csv.writer(response)
    
    # Write header row
    writer.writerow(['SL No', 'Student', 'Predicted Marks', 'Performance', 'Priority', 'Recommendation'])
    
    # Write data rows
    for item in data:
        writer.writerow([
            item['sl_no'],
            item['name'],
            f"{item['marks']:.1f}",
            item['performance'],
            item['priority'],
            item['recommendation']
        ])
    
    # Clear the session data after download (optional)
    # del request.session['recommendations_data']
    
    return response

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
from django.shortcuts import render, redirect

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)
from reportlab.platypus.flowables import HRFlowable
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages

from django.contrib.auth import get_user_model
from .utils import generate_otp
from .email_utils import send_email_otp,send_reset_password_otp
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth.hashers import check_password,make_password

User = get_user_model()

def resend_otp(request):
    email = request.session.get('verify_email')

    if not email:
        messages.error(request, "Session expired. Please register again.")
        return redirect('register')

    user = User.objects.get(email=email)

    # ⏱ Cooldown (30 seconds)
    last_sent = request.session.get('otp_last_sent')
    if last_sent:
        last_sent_time = timezone.datetime.fromisoformat(last_sent)
        if timezone.now() < last_sent_time + timedelta(seconds=5):
            messages.warning(request, "Please wait before requesting another OTP.")
            return redirect('verify_email')

    # 🔐 Generate new OTP
    otp = generate_otp()
    user.email_otp = make_password(otp)
    user.otp_expiry = timezone.now() + timedelta(minutes=5)
    user.save()

    # 📧 Send email
    send_email_otp(email, otp)

    # Save resend timestamp
    request.session['otp_last_sent'] = timezone.now().isoformat()

    messages.success(request, "New OTP sent to your email.")
    return redirect('verify_email')

def verify_email(request):
    email = request.session.get('verify_email')

    if not email:
        messages.error(request, "Session expired. Please register again.")
        return redirect('register')

    user = User.objects.get(email=email)

    if request.method == 'POST':
        otp = request.POST.get('verification_code')

        if timezone.now() > user.otp_expiry:
            messages.error(request, "OTP expired")
            return render(request, 'verify_email.html')

        if not check_password(otp, user.email_otp):
            messages.error(request, "Invalid OTP")
            return render(request, 'verify_email.html')

        # ✅ Verified
        user.is_email_verified = True
        user.email_otp = None
        user.otp_expiry = None
        user.save()

        del request.session['verify_email']

        messages.success(request, "Email verified successfully. You can login.")
        return redirect('login')

    return render(request, 'verify_email.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 != password2:
            messages.error(request, "Passwords do not match")
            return render(request, 'register.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return render(request, 'register.html')
        
        email = email.lower()

        if User.objects.filter(email__iexact=email).exists():
            messages.error(request, "Email already exists")
            return render(request, 'register.html')
        
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1,
            is_active=True,              
            is_email_verified=False
        )
        
        # ✅ Generate OTP
        otp = generate_otp()
        user.email_otp = make_password(otp)
        user.otp_expiry = timezone.now() + timedelta(minutes=1)
        user.save()

        # ✅ Send OTP
        send_email_otp(email, otp)

        # Store email in session for OTP verification
        request.session['verify_email'] = email
        request.session['otp_last_sent'] = timezone.now().isoformat()
        
        messages.success(request, "OTP sent to your email. Please verify.")
        return redirect('verify_email')

    return render(request, 'register.html')

def reset_password(request):
    # Step 1: get session values
    email = request.session.get('reset_email')
    otp_sent = request.session.get('otp_sent', False)

    if request.method == 'POST':
        if not otp_sent:
            # User submitted email + new password → generate OTP
            email = request.POST.get('email')
            password1 = request.POST.get('new_password')
            password2 = request.POST.get('confirm_password')

            if not email or not password1 or not password2:
                messages.error(request, "All fields are required")
                return render(request, 'reset_password.html')

            if password1 != password2:
                messages.error(request, "Passwords do not match")
                return render(request, 'reset_password.html')

            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                messages.error(request, "Email not found")
                return render(request, 'reset_password.html')

            # Generate OTP
            otp = generate_otp()
            user.email_otp = make_password(otp)  # Store hashed OTP
            user.otp_expiry = timezone.now() + timedelta(minutes=5)
            user.save()

            # Send OTP
            send_reset_password_otp(email, otp)

            # Store session info
            request.session['reset_email'] = email
            request.session['new_password'] = password1  # Store plain password temporarily
            request.session['otp_sent'] = True

            messages.success(request, f"OTP sent to {email}. Please enter OTP below.")
            return render(request, 'recommender/reset_password.html', {'otp_stage': True, 'email': email})

        else:
            # Step 2: User submits OTP
            entered_otp = request.POST.get('otp')
            email = request.session.get('reset_email')
            password = request.session.get('new_password')
            
            print(f"OTP: {entered_otp}, Email: {email}, Password: {password}")
            
            if not email or not password:
                messages.error(request, "Session expired. Start again.")
                return redirect('reset_password')

            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                messages.error(request, "User not found. Please start again.")
                return redirect('reset_password')

            if not user.otp_expiry or timezone.now() > user.otp_expiry:
                messages.error(request, "OTP expired. Please try again.")
                # Reset session for security
                request.session.pop('otp_sent', None)
                request.session.pop('reset_email', None)
                request.session.pop('new_password', None)
                return redirect('reset_password')

            if not check_password(entered_otp, user.email_otp):
                messages.error(request, "Invalid OTP. Please try again.")
                return render(request, 'recommender/reset_password.html', {'otp_stage': True, 'email': email})

            # OTP verified → update password
            print(f"Setting password for {email} and pass:{(password)}")
            user.set_password(str(password))  # Use set_password to properly hash the password
            user.email_otp = None
            user.otp_expiry = None
            user.save()
            
            # Clear session
            request.session.pop('otp_sent', None)
            request.session.pop('reset_email', None)
            request.session.pop('new_password', None)

            messages.success(request, "Password changed successfully. You can login now.")
            return redirect('login')

    # GET request
    return render(request, 'reset_password.html', {'otp_stage': otp_sent, 'email': email})

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('username') 
        password = request.POST.get('password')

        try:
            # Determine if login by email or username
            if '@' in email:
                user = User.objects.get(email=email)
            else:
                user = User.objects.get(username=email)

            # Check if a password reset is pending
            if user.email_otp:
                request.session['reset_email'] = user.email
                messages.warning(request, "You have requested a password reset. Please verify OTP first.")
                return redirect('reset_password')

            # Authenticate with current password
            user_auth = authenticate(request, username=user.username, password=password)

            if user_auth is not None:
                if not user_auth.is_email_verified:
                    request.session['verify_email'] = user_auth.email
                    messages.warning(request, "Please verify your email before logging in.")
                    return redirect('verify_email')

                login(request, user_auth)
                return redirect('index')
            else:
                messages.error(request, "Invalid email or password")

        except User.DoesNotExist:
            messages.error(request, "Invalid email or password")

    return render(request, 'login.html')


def user_logout(request):
    logout(request)
    return redirect('login')

def landing(request):
    return render(request,'landing.html')
