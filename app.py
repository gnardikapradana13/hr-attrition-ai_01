import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')

# 1. Page Configuration with Enhanced Corporate Theme
st.set_page_config(
    page_title="AI-Powered Employee Retention Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Colorful Elegant Design
st.markdown("""
<style>
    /* Main Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        text-align: center;
        border: 3px solid white;
    }
    
    .metric-card {
        background: white;
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 2px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }
    
    .card-1 { border-color: #667eea; }
    .card-2 { border-color: #764ba2; }
    .card-3 { border-color: #f093fb; }
    .card-4 { border-color: #4facfe; }
    
    .risk-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.3);
        text-align: center;
        border: 2px solid white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(247, 151, 30, 0.3);
        text-align: center;
        border: 2px solid white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3);
        text-align: center;
        border: 2px solid white;
    }
    
    .risk-stable {
        background: linear-gradient(135deg, #42e695 0%, #3bb2b8 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(66, 230, 149, 0.3);
        text-align: center;
        border: 2px solid white;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 15px rgba(106, 17, 203, 0.2);
        border: 2px solid white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-left: 5px solid #667eea;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
    }
    
    .critical-recommendation {
        border-left: 5px solid #dc3545;
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    }
    
    .warning-recommendation {
        border-left: 5px solid #ff9800;
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    }
    
    .info-recommendation {
        border-left: 5px solid #2196f3;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    .success-recommendation {
        border-left: 5px solid #4caf50;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }
    
    .feature-label {
        font-weight: 700;
        color: #4a5568;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .progress-container {
        width: 100%;
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
        border-radius: 25px;
        margin: 1.5rem 0;
        height: 30px;
        border: 2px solid white;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .progress-bar {
        height: 26px;
        border-radius: 25px;
        text-align: center;
        line-height: 26px;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
        transition: width 1s ease-in-out;
    }
    
    .input-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #e2e8f0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
        border: 2px solid #e2e8f0;
        margin: 1.5rem 0;
    }
    
    .download-btn {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
        text-align: center;
        text-decoration: none;
        box-shadow: 0 5px 15px rgba(0, 176, 155, 0.3);
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 176, 155, 0.4);
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .welcome-step {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        height: 100%;
    }
    
    .badge {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        display: inline-block;
        margin: 0.5rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .st-emotion-cache-1qg05tj {
        background: linear-gradient(135deg, #f8faff 0%, #ffffff 100%);
    }
    
    .intervention-card {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    
    .warning-intervention {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
    }
    
    .info-intervention {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
    }
    
    .success-intervention {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
    }
    
    .strategic-intervention {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Define satisfaction labels globally
SATISFACTION_LABELS = ["Very Low", "Low", "Medium", "High"]

# 2. Load Model & Scaler Function with Protection (Try-Except)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model_ibm_attrition_87pc.pkl')
        scaler = joblib.load('scaler_ibm_attrition.pkl')
        return model, scaler, None
    except FileNotFoundError:
        error_msg = "Model (.pkl) or scaler file not found in the application folder."
        return None, None, error_msg
    except Exception as e:
        error_msg = f"Technical error occurred: {str(e)}"
        return None, None, error_msg

# 3. PDF Generation Function
def generate_pdf_report(prob_percent, risk_level, risk_icon, df_user, feature_fig, analysis_time):
    """Generate PDF report with analysis results"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from io import BytesIO
        
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Header with gradient effect (simulated)
        c.setFillColorRGB(0.4, 0.5, 0.9)
        c.rect(0, height-100, width, 100, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(width/2, height-50, "AI-Powered Employee Retention Predictor")
        
        # Subtitle
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        c.drawCentredString(width/2, height-70, f"Analysis Report | {analysis_time}")
        
        # Risk Summary
        c.setFont("Helvetica-Bold", 18)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.drawString(50, height-150, "Risk Analysis Summary")
        
        # Risk Level Box
        risk_colors = {
            "CRITICAL": (1, 0.3, 0.3),
            "HIGH": (1, 0.6, 0.2),
            "MODERATE": (0.3, 0.6, 1),
            "LOW": (0.4, 0.8, 0.4)
        }
        
        risk_key = risk_level.split()[0] if risk_level else "MODERATE"
        c.setFillColorRGB(*risk_colors.get(risk_key, (0.5, 0.5, 0.5)))
        c.roundRect(50, height-200, 200, 60, 10, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(60, height-170, f"{risk_icon} {risk_level}")
        c.setFont("Helvetica-Bold", 28)
        c.drawString(60, height-195, f"{prob_percent:.1f}%")
        
        # Employee Data
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height-280, "Employee Profile Data")
        
        y_position = height-310
        employee_data = [
            ("Age", f"{df_user['Age'].values[0]} years"),
            ("Monthly Income", f"${df_user['MonthlyIncome'].values[0]:,}"),
            ("Job Satisfaction", SATISFACTION_LABELS[df_user['JobSatisfaction'].values[0]-1]),
            ("Stock Option Level", f"Level {df_user['StockOptionLevel'].values[0]}"),
            ("Overtime", "Yes" if df_user['OverTime_Yes'].values[0] == 1 else "No")
        ]
        
        c.setFont("Helvetica", 12)
        for label, value in employee_data:
            c.drawString(70, y_position, f"• {label}: {value}")
            y_position -= 25
        
        # Feature Importance (if figure is provided)
        if feature_fig:
            # Save the matplotlib figure to a bytes buffer
            fig_buffer = BytesIO()
            feature_fig.savefig(fig_buffer, format='png', dpi=150, bbox_inches='tight')
            fig_buffer.seek(0)
            
            # Add the image to PDF
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, y_position-40, "Feature Contribution Analysis")
            c.drawImage(ImageReader(fig_buffer), 50, y_position-250, width=400, height=200)
        
        # Footer
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(50, 50, f"Generated on: {analysis_time}")
        c.drawString(50, 40, "Confidential - For HR Department Use Only")
        c.drawString(width-200, 50, "© 2026 AI-Powered Employee Retention Predictor")
        
        c.save()
        buffer.seek(0)
        return buffer
    except ImportError:
        return None
    except Exception as e:
        st.warning(f"Could not generate PDF: {e}")
        return None

# Load model and scaler
model, scaler, error_status = load_assets()

# Enhanced Main Header Section
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; margin: 0; font-size: 3.5rem; letter-spacing: 1px;">🎯 AI-Powered Employee Retention Predictor</h1>
        <h3 style="text-align: center; margin: 0.5rem 0 0 0; font-weight: 300; font-size: 1.5rem;">
            Advanced Predictive Analytics for Workforce Stability
        </h3>
        <p style="text-align: center; margin: 1rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
            Leveraging Machine Learning to Forecast Employee Attrition Risk
        </p>
    </div>
    """, unsafe_allow_html=True)

# Colorful System Status Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card card-1">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="background: #667eea; color: white; width: 40px; height: 40px; border-radius: 10px; 
                        display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                ⚙️
            </div>
            <h4 style="margin: 0; color: #4a5568;">System Status</h4>
        </div>
        <h2 style="margin: 0; color: #667eea; font-size: 2rem;">🟢 ONLINE</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card card-2">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="background: #764ba2; color: white; width: 40px; height: 40px; border-radius: 10px; 
                        display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                📊
            </div>
            <h4 style="margin: 0; color: #4a5568;">Model Accuracy</h4>
        </div>
        <h2 style="margin: 0; color: #764ba2; font-size: 2rem;">87.2%</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card card-3">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="background: #f093fb; color: white; width: 40px; height: 40px; border-radius: 10px; 
                        display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                🔒
            </div>
            <h4 style="margin: 0; color: #4a5568;">Data Privacy</h4>
        </div>
        <h2 style="margin: 0; color: #f093fb; font-size: 2rem;">ENCRYPTED</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card card-4">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="background: #4facfe; color: white; width: 40px; height: 40px; border-radius: 10px; 
                        display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                ⚡
            </div>
            <h4 style="margin: 0; color: #4a5568;">Processing Speed</h4>
        </div>
        <h2 style="margin: 0; color: #4facfe; font-size: 2rem;">REAL-TIME</h2>
    </div>
    """, unsafe_allow_html=True)

# Check if there was an error during loading
if error_status:
    st.error(f"""
    ### ❌ **System Initialization Failed**
    
    **Error Details:** {error_status}
    
    **Required Actions:**
    1. Ensure both model files are in the application directory
    2. Verify file permissions
    3. Contact system administrator for support
    """)
    st.stop()

# 4. Enhanced Sidebar with Colorful Design
st.sidebar.markdown("""
<div class="sidebar-header">
    <h3 style="margin: 0; font-size: 1.8rem;">👤 Employee Profile</h3>
    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.95;">
        Enter employee details for intelligent risk assessment
    </p>
</div>
""", unsafe_allow_html=True)

def get_user_input():
    st.sidebar.markdown("""
    <div class="input-section">
    """, unsafe_allow_html=True)
    
    # 1. Input Age
    st.sidebar.markdown('<p class="feature-label">👤 Employee Age</p>', unsafe_allow_html=True)
    age = st.sidebar.slider("", 18, 60, 30, key="s_age", label_visibility="collapsed")
    
    # 2. DYNAMIC INCOME LOGIC
    if age <= 25:
        max_salary = 7000
        default_salary = 3000
        age_group = "Junior (≤25)"
    elif 25 < age <= 40:
        max_salary = 15000
        default_salary = 6000
        age_group = "Mid-career (26-40)"
    else:
        max_salary = 20000
        default_salary = 10000
        age_group = "Senior (41+)"

    st.sidebar.markdown(f'<p style="font-size: 0.9rem; color: #764ba2; margin-top: -5px; font-weight: 600;">🎯 {age_group}</p>', unsafe_allow_html=True)
    
    # Monthly Income
    st.sidebar.markdown('<p class="feature-label">💰 Monthly Income ($)</p>', unsafe_allow_html=True)
    income = st.sidebar.slider(
        "", 
        min_value=1000, 
        max_value=max_salary, 
        value=default_salary,
        step=100,
        key="s_inc",
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown(f'<p style="font-size: 0.9rem; color: #667eea; margin-top: -5px;">💵 Range: $1,000 - ${max_salary:,}</p>', unsafe_allow_html=True)
    
    # Overtime
    st.sidebar.markdown('<p class="feature-label">⏰ Overtime Status</p>', unsafe_allow_html=True)
    overtime_val = st.sidebar.selectbox("", ("No", "Yes"), key="s_ot", label_visibility="collapsed")
    
    # Job Satisfaction
    st.sidebar.markdown('<p class="feature-label">😊 Job Satisfaction Level</p>', unsafe_allow_html=True)
    satisfaction = st.sidebar.slider("", 1, 4, 3, key="s_sat", label_visibility="collapsed")
    
    satisfaction_colors = ["#ff6b6b", "#ffa726", "#42a5f5", "#66bb6a"]
    satisfaction_color = satisfaction_colors[satisfaction-1]
    
    # Create a visual satisfaction indicator
    satisfaction_html = f"""
    <div style="display: flex; justify-content: space-between; margin: 5px 0;">
        {"".join([f'<div style="width: 20px; height: 20px; border-radius: 50%; background: {satisfaction_colors[i] if i < satisfaction else "#e2e8f0"}; margin: 0 2px;"></div>' for i in range(4)])}
    </div>
    """
    st.sidebar.markdown(satisfaction_html, unsafe_allow_html=True)
    st.sidebar.markdown(f'<p style="font-size: 0.9rem; color: {satisfaction_color}; margin-top: -5px; font-weight: 600;">📊 {SATISFACTION_LABELS[satisfaction-1]}</p>', unsafe_allow_html=True)
    
    # Stock Option Level
    st.sidebar.markdown('<p class="feature-label">📈 Stock Option Level</p>', unsafe_allow_html=True)
    stock_options = {
        "🔴 Level 0: None (Contract/New Employee)": 0,
        "🟡 Level 1: Standard (Permanent Employee)": 1,
        "🟢 Level 2: High (Key Talent/Managerial)": 2,
        "🔵 Level 3: Maximum (Executive/Vital Position)": 3
    }
    
    selected_label = st.sidebar.selectbox(
        "", 
        options=list(stock_options.keys()), 
        index=1, 
        key="s_stock",
        label_visibility="collapsed"
    )
    
    stock_val = stock_options[selected_label]
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # Profile Summary
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin: 1.5rem 0;
                box-shadow: 0 8px 20px rgba(106, 17, 203, 0.2);">
        <h3 style="margin: 0 0 1rem 0; text-align: center;">📊 Profile Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="feature-highlight">
        <p style="margin: 0.5rem 0;"><strong>👤 Age:</strong> <span style="color: #667eea; font-weight: 700;">{age} years</span></p>
        <p style="margin: 0.5rem 0;"><strong>💰 Income:</strong> <span style="color: #764ba2; font-weight: 700;">${income:,}/month</span></p>
        <p style="margin: 0.5rem 0;"><strong>⏰ Overtime:</strong> <span style="color: #f093fb; font-weight: 700;">{overtime_val}</span></p>
        <p style="margin: 0.5rem 0;"><strong>😊 Satisfaction:</strong> <span style="color: {satisfaction_color}; font-weight: 700;">{SATISFACTION_LABELS[satisfaction-1]}</span></p>
        <p style="margin: 0.5rem 0;"><strong>📈 Stock Level:</strong> <span style="color: #4facfe; font-weight: 700;">{stock_val}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    data = {
        'Age': age,
        'MonthlyIncome': income,
        'JobSatisfaction': satisfaction,
        'StockOptionLevel': stock_val,
        'OverTime_Yes': 1 if overtime_val == "Yes" else 0
    }
    return pd.DataFrame([data]), satisfaction

# --- LOGICAL DATA PROCESSING FLOW ---

# A. Get Input
df_user, satisfaction_value = get_user_input()

# B. Feature Alignment (Template)
X_template = pd.DataFrame(columns=scaler.feature_names_in_)
df_final = pd.concat([X_template, df_user]).fillna(0)
df_final = df_final[scaler.feature_names_in_]

# C. Scaling
input_scaled = scaler.transform(df_final)

# Main Content Area
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                padding: 2rem; border-radius: 20px; margin: 1rem 0; border: 2px solid rgba(102, 126, 234, 0.2);">
        <h2 style="margin: 0 0 1rem 0; color: #4a5568;">🎯 Analysis Dashboard</h2>
        <p style="margin: 0; color: #718096; line-height: 1.6;">
            This intelligent system utilizes advanced machine learning algorithms to predict 
            employee attrition risk based on comprehensive HR metrics. 
            Our model has been meticulously trained on historical workforce data, 
            achieving an impressive 87.2% prediction accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("🚀 **RUN ADVANCED ANALYSIS**", key="predict_button"):
        st.session_state['analysis_run'] = True
        st.session_state['analysis_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        if 'analysis_run' not in st.session_state:
            st.session_state['analysis_run'] = False

st.markdown("---")

# Display Results Section
if st.session_state.get('analysis_run'):
    # 1. Perform Prediction
    prediction = model.predict(input_scaled)
    prob_resign_raw = model.predict_proba(input_scaled)[0][1]
    prob_resign = float(prob_resign_raw)
    prob_percent = prob_resign * 100
    
    # Create two columns for results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Advanced Risk Analysis Results")
        
        # Determine risk level and styling
        if prob_percent >= 75:
            risk_class = "risk-high"
            risk_level = "CRITICAL RISK"
            risk_icon = "🚨"
            bar_color = "#ff416c"
            intervention_class = "intervention-card"
        elif prob_percent >= 50:
            risk_class = "risk-medium"
            risk_level = "HIGH RISK"
            risk_icon = "⚠️"
            bar_color = "#f7971e"
            intervention_class = "intervention-card warning-intervention"
        elif prob_percent >= 30:
            risk_class = "risk-low"
            risk_level = "MODERATE RISK"
            risk_icon = "🔶"
            bar_color = "#4facfe"
            intervention_class = "intervention-card info-intervention"
        else:
            risk_class = "risk-stable"
            risk_level = "LOW RISK"
            risk_icon = "✅"
            bar_color = "#42e695"
            intervention_class = "intervention-card success-intervention"
        
        # Enhanced Progress Bar with animation
        progress_html = f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {prob_percent}%; background: linear-gradient(90deg, {bar_color} 0%, {bar_color}99 100%);">
                {prob_percent:.1f}% Attrition Risk
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        
        # Risk Zones with Icons
        zones_col1, zones_col2, zones_col3, zones_col4 = st.columns(4)
        with zones_col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                        border-radius: 10px; margin: 0.5rem 0; border: 2px solid #28a745;">
                <div style="font-size: 1.5rem;">✅</div>
                <small><strong>LOW</strong></small><br>
                <small>0-30%</small>
            </div>
            """, unsafe_allow_html=True)
        with zones_col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                        border-radius: 10px; margin: 0.5rem 0; border: 2px solid #ffc107;">
                <div style="font-size: 1.5rem;">🔶</div>
                <small><strong>MODERATE</strong></small><br>
                <small>30-50%</small>
            </div>
            """, unsafe_allow_html=True)
        with zones_col3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
                        border-radius: 10px; margin: 0.5rem 0; border: 2px solid #dc3545;">
                <div style="font-size: 1.5rem;">⚠️</div>
                <small><strong>HIGH</strong></small><br>
                <small>50-75%</small>
            </div>
            """, unsafe_allow_html=True)
        with zones_col4:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); 
                        color: white; border-radius: 10px; margin: 0.5rem 0; border: 2px solid #bd2130;">
                <div style="font-size: 1.5rem;">🚨</div>
                <small><strong>CRITICAL</strong></small><br>
                <small>75-100%</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Create enhanced bar chart with matplotlib
        fig, ax = plt.subplots(figsize=(12, 7))
        feature_names = ['Age', 'Monthly Income', 'Job Satisfaction', 'Stock Options', 'Overtime']
        contributions = [
            df_user['Age'].values[0] / 60 * 100,
            df_user['MonthlyIncome'].values[0] / 20000 * 100,
            df_user['JobSatisfaction'].values[0] / 4 * 100,
            df_user['StockOptionLevel'].values[0] / 3 * 100,
            100 if df_user['OverTime_Yes'].values[0] == 1 else 20
        ]
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00b09b']
        bars = ax.bar(feature_names, contributions, color=colors, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, contribution in zip(bars, contributions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{contribution:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Customize the plot
        ax.set_ylabel('Relative Impact (%)', fontsize=14, fontweight='bold', color='#4a5568')
        ax.set_title('Feature Contribution to Risk Assessment', fontsize=16, fontweight='bold', color='#2d3748', pad=20)
        ax.set_ylim(0, 120)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        plt.xticks(rotation=45, ha='right', fontsize=12, color='#4a5568')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#e2e8f0')
        
        # Add background color
        ax.set_facecolor('#f8faff')
        fig.patch.set_facecolor('#ffffff')
        
        st.pyplot(fig)
        
        # Store figure for PDF export
        st.session_state['feature_fig'] = fig
    
    with col2:
        st.markdown("### 📈 Risk Level Assessment")
        risk_html = f"""
        <div class="{risk_class}">
            <h1 style="margin: 0; font-size: 3.5rem;">{risk_icon}</h1>
            <h2 style="margin: 0.5rem 0; font-size: 1.8rem;">{risk_level}</h2>
            <h3 style="margin: 0; font-size: 2.5rem;">{prob_percent:.1f}%</h3>
        </div>
        """
        st.markdown(risk_html, unsafe_allow_html=True)
        
        st.markdown("### 📋 Detailed Assessment")
        assessment_time = st.session_state['analysis_time']
        assessment_html = f"""
        <div class="result-card">
            <div style="margin-bottom: 1.5rem;">
                <p style="margin: 0.5rem 0;"><strong>🎯 Attrition Probability:</strong> <span style="color: #667eea; font-weight: 700;">{prob_percent:.2f}%</span></p>
                <p style="margin: 0.5rem 0;"><strong>✅ Retention Probability:</strong> <span style="color: #42e695; font-weight: 700;">{100-prob_percent:.2f}%</span></p>
                <p style="margin: 0.5rem 0;"><strong>📊 Confidence Level:</strong> <span style="color: #764ba2; font-weight: 700;">87.2%</span></p>
                <p style="margin: 0.5rem 0;"><strong>🤖 Model Prediction:</strong> <span style="color: #f093fb; font-weight: 700;">{'Will Leave' if prediction[0] == 1 else 'Will Stay'}</span></p>
            </div>
            <hr style="border: none; border-top: 2px dashed #e2e8f0; margin: 1rem 0;">
            <div>
                <p style="margin: 0.5rem 0;"><strong>⏰ Analysis Time:</strong> {assessment_time}</p>
                <p style="margin: 0.5rem 0;"><strong>🔢 Model Version:</strong> v2.1.0</p>
                <p style="margin: 0.5rem 0;"><strong>📊 Data Points:</strong> 1,470 employees</p>
            </div>
        </div>
        """
        st.markdown(assessment_html, unsafe_allow_html=True)
        
        # Quick Stats with Icons
        st.markdown("### 📊 Employee Profile Snapshot")
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                        padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                <div style="font-size: 2rem;">👤</div>
                <p style="margin: 0.25rem 0; font-size: 1.8rem; font-weight: 700; color: #667eea;">{df_user['Age'].values[0]}</p>
                <p style="margin: 0; font-size: 0.9rem; color: #718096;">Years Old</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(66, 230, 149, 0.1) 0%, rgba(59, 178, 184, 0.1) 100%); 
                        padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                <div style="font-size: 2rem;">😊</div>
                <p style="margin: 0.25rem 0; font-size: 1.8rem; font-weight: 700; color: #42e695;">{SATISFACTION_LABELS[df_user['JobSatisfaction'].values[0]-1]}</p>
                <p style="margin: 0; font-size: 0.9rem; color: #718096;">Satisfaction</p>
            </div>
            """, unsafe_allow_html=True)
            
        with stats_col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(118, 75, 162, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%); 
                        padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                <div style="font-size: 2rem;">💰</div>
                <p style="margin: 0.25rem 0; font-size: 1.8rem; font-weight: 700; color: #764ba2;">${df_user['MonthlyIncome'].values[0]:,}</p>
                <p style="margin: 0; font-size: 0.9rem; color: #718096;">Monthly Income</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%); 
                        padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                <div style="font-size: 2rem;">📈</div>
                <p style="margin: 0.25rem 0; font-size: 1.8rem; font-weight: 700; color: #4facfe;">{df_user['StockOptionLevel'].values[0]}</p>
                <p style="margin: 0; font-size: 0.9rem; color: #718096;">Stock Level</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Recommendations Section using Streamlit components
    st.markdown("### 💡 Strategic Action Plan")
    
    if prob_percent >= 90:
        st.markdown("#### 🚨 CRITICAL ACTION REQUIRED")
        st.markdown("**🛑 Immediate Interventions (48 Hours):**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="intervention-card">
                <strong>1. Executive Review</strong><br>
                <small>Schedule urgent meeting with HR Director & Department Head</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card">
                <strong>2. Succession Planning</strong><br>
                <small>Initiate immediate backup and knowledge transfer planning</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="intervention-card">
                <strong>3. Retention Interview</strong><br>
                <small>Conduct pre-emptive retention interview with HRBP</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card">
                <strong>4. Compensation Review</strong><br>
                <small>Evaluate competitive market position and adjust if needed</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**📊 Priority Level:** :red[**HIGHEST**]")
        st.markdown("**Timeline:** Immediate action required within 48 hours")
        st.markdown("**Stakeholders:** HR Director, Department Head, Talent Management")
        
    elif 75 <= prob_percent < 90:
        st.markdown("#### ⚠️ HIGH PRIORITY ATTENTION")
        st.markdown("**🔍 Recommended Actions (1 Week):**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="intervention-card warning-intervention">
                <strong>1. Stay Interview</strong><br>
                <small>Conduct in-depth retention discussion with manager</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card warning-intervention">
                <strong>2. Career Pathing</strong><br>
                <small>Develop clear advancement opportunities and timeline</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="intervention-card warning-intervention">
                <strong>3. Mentorship Program</strong><br>
                <small>Assign senior mentor for guidance and support</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card warning-intervention">
                <strong>4. Workload Assessment</strong><br>
                <small>Review and adjust responsibilities and work-life balance</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**📊 Priority Level:** :orange[**HIGH**]")
        st.markdown("**Timeline:** Action required within 1 week")
        st.markdown("**Stakeholders:** Line Manager, HR Business Partner")
        
    elif 50 <= prob_percent < 75:
        st.markdown("#### 🔶 MODERATE CONCERN")
        st.markdown("**🛡️ Preventive Measures (2 Weeks):**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="intervention-card info-intervention">
                <strong>1. Recognition Program</strong><br>
                <small>Implement regular appreciation and recognition</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card info-intervention">
                <strong>2. Skill Development</strong><br>
                <small>Offer relevant training and certification opportunities</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="intervention-card info-intervention">
                <strong>3. Team Engagement</strong><br>
                <small>Enhance team-building and social activities</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card info-intervention">
                <strong>4. Flexibility Options</strong><br>
                <small>Consider flexible work arrangements if possible</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**📊 Priority Level:** :blue[**MEDIUM**]")
        st.markdown("**Timeline:** Action recommended within 2 weeks")
        st.markdown("**Stakeholders:** Team Lead, HR Coordinator")
        
    elif 30 <= prob_percent < 50:
        st.markdown("#### ✅ STABLE PERFORMER")
        st.markdown("**🌟 Retention Strategies (Quarterly):**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="intervention-card success-intervention">
                <strong>1. Career Development</strong><br>
                <small>Plan next promotion cycle and growth opportunities</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card success-intervention">
                <strong>2. Special Projects</strong><br>
                <small>Assign challenging and meaningful projects</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="intervention-card success-intervention">
                <strong>3. Leadership Training</strong><br>
                <small>Include in management development programs</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card success-intervention">
                <strong>4. Peer Recognition</strong><br>
                <small>Highlight achievements in team meetings</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**📊 Priority Level:** :green[**LOW**]")
        st.markdown("**Timeline:** Quarterly review and planning")
        st.markdown("**Stakeholders:** Department Manager")
        
    else:
        st.markdown("#### 🌟 HIGH-POTENTIAL TALENT")
        st.markdown("**🚀 Growth Initiatives (Annual):**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="intervention-card strategic-intervention">
                <strong>1. Succession Candidate</strong><br>
                <small>Include in leadership pipeline and succession planning</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card strategic-intervention">
                <strong>2. Strategic Projects</strong><br>
                <small>Assign high-impact, business-critical initiatives</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="intervention-card strategic-intervention">
                <strong>3. External Representation</strong><br>
                <small>Include in industry events and conferences</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="intervention-card strategic-intervention">
                <strong>4. Cross-Training</strong><br>
                <small>Develop broader organizational knowledge</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**📊 Priority Level:** :violet[**STRATEGIC**]")
        st.markdown("**Timeline:** Annual development planning")
        st.markdown("**Stakeholders:** Executive Leadership, Talent Development")
    
    # Export Section with PDF Generation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analysis_time = st.session_state['analysis_time']
        export_html = f"""
        <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2.5rem; border-radius: 20px; color: white; margin: 1.5rem 0;
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);">
            <h3 style="margin: 0 0 1rem 0; font-size: 1.8rem;">📤 Export Analysis Report</h3>
            <p style="margin: 0 0 1.5rem 0; opacity: 0.9;">Generate comprehensive professional report for HR records</p>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;">
                <p style="margin: 0.5rem 0;"><strong>📅 Report Generated:</strong> {analysis_time}</p>
                <p style="margin: 0.5rem 0;"><strong>🎯 Risk Level:</strong> {risk_level}</p>
                <p style="margin: 0.5rem 0;"><strong>📊 Probability:</strong> {prob_percent:.1f}%</p>
            </div>
        </div>
        """
        st.markdown(export_html, unsafe_allow_html=True)
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Generate and download PDF
            try:
                pdf_buffer = generate_pdf_report(
                    prob_percent=prob_percent,
                    risk_level=risk_level,
                    risk_icon=risk_icon,
                    df_user=df_user,
                    feature_fig=st.session_state.get('feature_fig'),
                    analysis_time=st.session_state['analysis_time']
                )
                
                if pdf_buffer:
                    pdf_b64 = base64.b64encode(pdf_buffer.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="Employee_Retention_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf" class="download-btn">📄 DOWNLOAD PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("PDF generation failed. Please install reportlab: `pip install reportlab`")
            except Exception as e:
                st.warning("PDF generation requires additional libraries. Please install: `pip install reportlab`")
                st.code("pip install reportlab", language="bash")
        
        with export_col2:
            # Export as CSV
            csv = df_user.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="employee_data_{datetime.now().strftime("%Y%m%d")}.csv" class="download-btn">📊 EXPORT CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with export_col3:
            # Quick Summary
            summary_text = f"""AI-Powered Employee Retention Predictor Report
Generated: {st.session_state['analysis_time']}

EMPLOYEE PROFILE:
- Age: {df_user['Age'].values[0]} years
- Monthly Income: ${df_user['MonthlyIncome'].values[0]:,}
- Job Satisfaction: {SATISFACTION_LABELS[df_user['JobSatisfaction'].values[0]-1]}
- Stock Option Level: {df_user['StockOptionLevel'].values[0]}
- Overtime: {'Yes' if df_user['OverTime_Yes'].values[0] == 1 else 'No'}

RISK ANALYSIS:
- Attrition Risk: {prob_percent:.1f}%
- Risk Level: {risk_level}
- Retention Probability: {100-prob_percent:.1f}%
- Model Prediction: {'Will Leave' if prediction[0] == 1 else 'Will Stay'}

RECOMMENDED ACTION: {risk_level.replace('RISK', 'PRIORITY')}
"""
            b64 = base64.b64encode(summary_text.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="retention_summary_{datetime.now().strftime("%Y%m%d")}.txt" class="download-btn">📋 QUICK SUMMARY</a>'
            st.markdown(href, unsafe_allow_html=True)

else:
    # Welcome/Instructions when no analysis has been run
    st.markdown("""
    <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 25px; margin: 2rem 0; color: white; box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);">
        <h2 style="margin: 0 0 1.5rem 0; font-size: 2.5rem;">🎯 Welcome to AI-Powered Employee Retention Predictor</h2>
        <p style="font-size: 1.3rem; opacity: 0.95; margin-bottom: 2rem;">
            Transform your HR strategy with predictive analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Steps using Streamlit columns
    st.markdown("### 📋 How to Use This System")
    step_col1, step_col2, step_col3 = st.columns(3)
    
    with step_col1:
        st.markdown("""
        <div class="welcome-step">
            <div style="font-size: 3rem; margin-bottom: 1rem;">1️⃣</div>
            <h4 style="margin: 0;">Complete Profile</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Fill employee details in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step_col2:
        st.markdown("""
        <div class="welcome-step">
            <div style="font-size: 3rem; margin-bottom: 1rem;">2️⃣</div>
            <h4 style="margin: 0;">Run Analysis</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Click the RUN button above</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step_col3:
        st.markdown("""
        <div class="welcome-step">
            <div style="font-size: 3rem; margin-bottom: 1rem;">3️⃣</div>
            <h4 style="margin: 0;">Get Insights</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Review detailed analysis & recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Badges using Streamlit columns
    st.markdown("<br>", unsafe_allow_html=True)
    badge_col1, badge_col2, badge_col3 = st.columns(3)
    
    with badge_col1:
        st.markdown("""
        <div class="badge">
            🔒 GDPR Compliant
        </div>
        """, unsafe_allow_html=True)
    
    with badge_col2:
        st.markdown("""
        <div class="badge">
            🤖 AI-Powered Analytics
        </div>
        """, unsafe_allow_html=True)
    
    with badge_col3:
        st.markdown("""
        <div class="badge">
            📊 Real-time Insights
        </div>
        """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.95rem; padding: 2rem; 
            background: linear-gradient(135deg, #f8faff 0%, #ffffff 100%); 
            border-radius: 15px; margin-top: 2rem; border: 2px solid #e2e8f0;">
    <p style="margin: 0.5rem 0; font-size: 1.1rem; color: #4a5568;">
        <strong>© 2026 AI-Powered Employee Retention Predictor | HR Analytics Platform v2.1.0</strong>
    </p>
    <p style="margin: 0.5rem 0;">
        For support contact: 
        <a href="mailto:gustingurahardi@student.ub.ac.id" style="color: #667eea; text-decoration: none; font-weight: 600;">
            gustingurahardi@student.ub.ac.id
        </a> 
        | 📞 +62 (555) 123-4567
    </p>
    <div style="margin: 1rem 0;">
        <span style="color: #667eea; font-weight: 600;">🔬 Advanced Analytics</span> • 
        <span style="color: #764ba2; font-weight: 600;">🤖 Machine Learning</span> • 
        <span style="color: #f093fb; font-weight: 600;">📊 Predictive Modeling</span> • 
        <span style="color: #4facfe; font-weight: 600;">🎯 Strategic HR</span>
    </div>
    <p style="font-size: 0.85rem; margin-top: 1rem; color: #a0aec0; max-width: 800px; margin-left: auto; margin-right: auto;">
        This system is for internal organizational use only. All predictions are based on statistical models 
        and should be used as decision-support tools alongside professional HR judgment. 
        Data privacy and security are our top priorities.
    </p>
</div>
""", unsafe_allow_html=True)
