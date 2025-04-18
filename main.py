from flask import Flask, render_template, request, redirect, url_for, flash, session,jsonify
from flask_mysqldb import MySQL
import re
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import xgboost as xgb  # Import xgboost explicitly
import json

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'crop_yield_prediction'

# Secret key for session
app.config['SECRET_KEY'] = 'crop_yield_prediction'

# Initialize MySQL
mysql = MySQL(app)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

# Constants
AREAS = ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
         'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi',
         'Cameroon', 'Canada', 'Central African Republic', 'Chile', 'Colombia', 'Croatia', 'Denmark',
         'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Eritrea', 'Estonia', 'Finland', 'France', 'Germany',
         'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'India', 'Indonesia',
         'Iraq', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon', 'Lesotho', 'Libya',
         'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Montenegro',
         'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway',
         'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia',
         'Senegal', 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland',
         'Tajikistan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom', 'Uruguay', 'Zambia',
         'Zimbabwe']

CROPS = ['Cassava', 'Maize', 'Plantains and others', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Sweet potatoes',
         'Wheat', 'Yams']


def predict_single_sample(area, item, year, rainfall, pesticides, avg_temp):
    # Check if models exist
    if not os.path.exists('models/bigru_model.h5'):
        print("Error: Models not found. Please run the training script first.")
        return None, 0

    # Load models and preprocessing objects
    model_files = [
        'models/bigru_model.h5',
        'models/hybrid_xgb_model.joblib',
        'models/scaler_X.joblib',
        'models/scaler_y.joblib'
    ]

    for file in model_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Error: Missing model file {file}. Ensure all models are trained and saved.")

    bigru_model = load_model('models/bigru_model.h5')
    hybrid_xgb = joblib.load('models/hybrid_xgb_model.joblib')
    scaler_X = joblib.load('models/scaler_X.joblib')
    scaler_y = joblib.load('models/scaler_y.joblib')

    # Load encoders
    encoders = {}
    for col in ['Area', 'Item']:
        encoders[col] = joblib.load(f'models/encoder_{col}.joblib')

    if area not in encoders['Area'].classes_:
        print(f"Error: '{area}' not found in training data.")
        return None, 0  # Return early to prevent undefined variables

    if item not in encoders['Item'].classes_:
        print(f"Error: '{item}' not found in training data.")
        return None, 0  # Return early to prevent undefined variables

    # Now, encode the categorical variables safely
    area_encoded = encoders['Area'].transform([area])[0]
    item_encoded = encoders['Item'].transform([item])[0]

    # Create a single sample in array format
    sample = np.array([[area_encoded, item_encoded, year, rainfall, pesticides, avg_temp]])

    print("Before Scaling:", sample)
    sample_scaled = scaler_X.transform(sample)  # Keep only one transformation
    print("After Scaling:", sample_scaled)

    # For Bi-GRU, we need historical data to form a sequence
    # Here we'll duplicate the sample to create a sequence of required length
    time_steps = 3
    sample_seq = np.expand_dims(sample_scaled, axis=0)  # Reshape for Bi-GRU

    # Get Bi-GRU prediction
    # Fix Bi-GRU input shape
    time_steps = 3  # Set time steps as expected by your model
    sample_seq = np.tile(sample_scaled, (time_steps, 1))  # Duplicate the sample 3 times
    sample_seq = np.reshape(sample_seq, (1, time_steps, 6))  # Reshape for Bi-GRU

    # Predict using Bi-GRU
    bigru_pred = bigru_model.predict(sample_seq)

    # Combine with tabular data for hybrid model
    hybrid_features = np.column_stack((bigru_pred, sample_scaled))

    # Get hybrid model prediction
    y_pred_hybrid = hybrid_xgb.predict(hybrid_features)
    print("Bi-GRU Prediction Shape:", bigru_pred.shape)
    print("Hybrid Features Shape:", hybrid_features.shape)

    # Calculate confidence score (using a simpler approach)
    # Option 1: Using feature importance-based approach
    try:
        # Create DMatrix properly
        dmatrix = xgb.DMatrix(np.array(hybrid_features, dtype=np.float32))

        confidence = hybrid_xgb.get_booster().predict(dmatrix, pred_contribs=True)
        confidence_score = min(0.95, max(0.99, np.mean(np.abs(confidence)) / 10))
    except Exception as e:
        print(f"Error calculating confidence with XGBoost API: {e}")
        # Fallback method: Use prediction value as a basis for confidence
        prediction_range = [0, 100000]  # Approximate range of possible yield values
        normalized_pred = (y_pred_hybrid[0] - prediction_range[0]) / (prediction_range[1] - prediction_range[0])
        confidence_score = min(0.95,
                               max(0.65, 0.80 + normalized_pred * 0.15))  # Base confidence of 0.80 with adjustment

    if year < 1990 or year > 2100:  # Adjust the range as per your dataset
        print(f"Warning: Year {year} is out of range. Adjusting to default (2025).")
        year = 2025  # Set a default year or handle it accordingly

    # Inverse transform prediction to original scale
    prediction = float(scaler_y.inverse_transform(y_pred_hybrid.reshape(-1, 1)).flatten()[0])
    confidence_score = float(confidence_score)

    return prediction, confidence_score


def get_suggestions(area, item, predicted_yield, rainfall, pesticides, avg_temp):
    """Generate suggestions to improve or maintain crop yield"""
    suggestions = []

    # Basic suggestions based on crop type
    crop_suggestions = {
        'Wheat': ['Ensure proper spacing between plants (15-20 cm)', 'Apply nitrogen fertilizer during tillering stage',
                  'Consider drought-resistant varieties for low rainfall areas'],
        'Rice, paddy': ['Maintain proper water level in paddy fields',
                        'Consider SRI (System of Rice Intensification) method',
                        'Apply organic matter before transplanting'],
        'Maize': ['Plant in rows 75 cm apart with 20-25 cm between plants', 'Apply fertilizer in split doses',
                  'Control fall armyworm with appropriate measures'],
        'Potatoes': ['Ensure proper hilling to prevent greening',
                     'Control for late blight disease with appropriate fungicides',
                     'Use certified seed potatoes for better yield'],
        'Cassava': ['Plant at the beginning of rainy season', 'Harvest at optimal maturity (8-12 months)',
                    'Control cassava mosaic disease through resistant varieties'],
        'Soybeans': ['Inoculate seeds with Rhizobium bacteria', 'Control for pod-sucking insects',
                     'Maintain adequate soil pH (6.0-6.5)'],
        'Sweet potatoes': ['Use vine cuttings for planting', 'Maintain adequate soil moisture',
                           'Rotate crops to prevent soil-borne diseases'],
        'Sorghum': ['Plant when soil temperature reaches 15°C', 'Control for sorghum midge and head smut',
                    'Consider bird-resistant varieties in affected areas'],
        'Yams': ['Use minisett technique for propagation', 'Provide support for climbing vines',
                 'Control nematodes through crop rotation'],
        'Plantains and others': ['Apply mulch to conserve moisture', 'Ensure proper drainage to prevent waterlogging',
                                 'Control black Sigatoka disease with appropriate measures']
    }

    # Add crop-specific suggestions (up to 3)
    if item in crop_suggestions:
        suggestions.extend(crop_suggestions[item][:3])

    # Region-specific suggestions
    tropical_regions = ['India', 'Brazil', 'Indonesia', 'Thailand', 'Malaysia', 'Bangladesh']
    temperate_regions = ['France', 'Germany', 'United Kingdom', 'Canada', 'Ukraine']
    arid_regions = ['Egypt', 'Saudi Arabia', 'Algeria', 'Libya', 'Sudan']

    if area in tropical_regions:
        suggestions.append('Consider intercropping with nitrogen-fixing legumes to improve soil fertility')
    elif area in temperate_regions:
        suggestions.append('Monitor winter temperatures for potential frost damage to early plantings')
    elif area in arid_regions:
        suggestions.append('Implement drip irrigation to conserve water in this arid region')

    # Rainfall suggestions
    if rainfall < 500:
        suggestions.append('Increase irrigation frequency as rainfall is below optimal level for most crops')
    elif rainfall > 1500:
        suggestions.append('Ensure proper drainage to prevent waterlogging and root diseases')

    # Temperature suggestions
    if avg_temp < 15:
        suggestions.append('Consider cold-resistant varieties for better yield in cooler temperatures')
    elif avg_temp > 30:
        suggestions.append('Implement shade structures or mulching to reduce heat stress during peak temperatures')

    # Pesticide suggestions
    if pesticides < 10000:
        suggestions.append('Monitor pest populations closely as pesticide usage is relatively low')
    elif pesticides > 100000:
        suggestions.append('Consider integrated pest management to reduce chemical dependence and environmental impact')

    return suggestions[:5]  # Limit to top 5 suggestions for better UI
    # Return top 5 suggestions for better UI


def create_confidence_graph(confidence):
    """Create an enhanced confidence meter graph"""
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 2.5))

    # Define color regions
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    # Create a background with color regions
    for i, color in enumerate(colors):
        start = i / 3
        ax.barh(0, 1 / 3, left=start, color=color, alpha=0.3)

    # Create main confidence bar
    ax.barh(0, confidence, color='#2980b9', height=0.5, alpha=0.8)

    # Add confidence value
    ax.text(confidence, 0, f'{confidence:.2f}', va='center', ha='center',
            fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=0.3',
                                                        fc='#2980b9', ec='none'))

    # Add labels for confidence regions
    ax.text(1 / 6, -0.5, 'Low', ha='center', va='center', fontsize=9)
    ax.text(3 / 6, -0.5, 'Medium', ha='center', va='center', fontsize=9)
    ax.text(5 / 6, -0.5, 'High', ha='center', va='center', fontsize=9)

    # Set limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add a title
    ax.set_title('Prediction Confidence Level', fontsize=12, pad=10)

    # Add benchmark indicators
    for x in [1 / 3, 2 / 3]:
        ax.axvline(x, color='white', linestyle='-', alpha=0.5, lw=2)

    # Convert plot to base64 string for embedding in HTML
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_str

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = request.form.get('role', 'farmer')

        # Form validation
        error = None

        if not name or not email or not password:
            error = "All fields are required"
        elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            error = "Invalid email address"
        elif password != confirm_password:
            error = "Passwords do not match"
        elif len(password) < 8:
            error = "Password must be at least 8 characters long"

        if error:
            flash(error, 'danger')
            return render_template('register.html')

        # Check if email already exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        if user:
            cur.close()
            flash('Email already registered', 'danger')
            return render_template('register.html')

        # Hash password
        hashed_password = generate_password_hash(password)

        # Insert new user
        cur.execute(
            "INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, %s)",
            (name, email, hashed_password, role)
        )
        mysql.connection.commit()
        cur.close()

        flash('Registration successful, please log in', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        email = request.form['email']
        password = request.form['password']

        # Validate form
        if not email or not password:
            flash('Please enter email and password', 'danger')
            return render_template('login.html')

        # Check if user exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        if user and check_password_hash(user[3], password):
            # Create session
            session['logged_in'] = True
            session['user_id'] = user[0]
            session['name'] = user[1]
            session['email'] = user[2]
            session['role'] = user[4]

            flash(f'Welcome back, {user[1]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')

        cur.close()

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', areas=AREAS, crops=CROPS)


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form
    area = data.get('area')
    item = data.get('item')
    year = int(data.get('year'))  # Keep only one conversion

    rainfall = float(data.get('rainfall'))
    pesticides = float(data.get('pesticides'))
    avg_temp = float(data.get('avg_temp'))
    year = int(data.get('year'))  # Ensure this is correctly retrieved

    # Validate year range
    if year < 1990 or year > 2100:
        flash("Invalid year selected. Please choose a year between 1990 and 2100.", "danger")
        return redirect(url_for('dashboard'))

    # Make prediction
    predicted_yield, confidence = predict_single_sample(area, item, year, rainfall, pesticides, avg_temp)

    # Get suggestions
    suggestions = get_suggestions(area, item, predicted_yield, rainfall, pesticides, avg_temp)

    # Create confidence graph
    confidence_graph = create_confidence_graph(confidence)

    # Prepare results
    result = {
        'yield': round(predicted_yield, 2),
        'yield_tonnes': round(predicted_yield / 100, 2),
        'tonnes_acres':round(predicted_yield * 0.0000892, 5),
        'confidence': round(confidence, 2),
        'confidence_graph': confidence_graph,
        'suggestions': suggestions,
        'input_data': {
            'area': area,
            'item': item,
            'year': year,
            'rainfall': rainfall,
            'pesticides': pesticides,
            'avg_temp': avg_temp
        }
    }

    return jsonify(result)

@app.route('/profile')
def profile():
    if 'logged_in' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))

    # Get user details
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (session['user_id'],))
    user = cur.fetchone()
    cur.close()

    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('login'))

    # Create user object to pass to template
    user_data = {
        'id': user[0],
        'name': user[1],
        'email': user[2],
        'role': user[4],
        'created_at': user[5]
    }

    return render_template('profile.html', user=user_data)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)