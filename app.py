#!/usr/bin/env python3

import os
import pickle
import json
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import re
import random

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with restrictions for production
if os.environ.get('FLASK_ENV') == 'production':
    CORS(app, origins=['https://your-domain.com'])  # Restrict in production
else:
    CORS(app)  # Allow all origins in development

# Require secret key from environment
app.secret_key = os.environ.get('SESSION_SECRET')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
wildfire_models = {}

def load_models():
    """Load all wildfire prediction models"""
    models_dir = 'models'
    model_files = {
        'ultimate_risk': 'ultimate_wildfire_risk_model_1758094894286.pkl',
        'risk': 'wildfire_risk_model_1758094894333.pkl', 
        'cause': 'wildfire_cause_model_1758094894332.pkl'
    }
    
    global wildfire_models
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(models_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                wildfire_models[model_name] = pickle.load(f)
            logger.info(f"Successfully loaded {model_name} model from {filename}")
        except Exception as e:
            logger.error(f"Failed to load {model_name} model: {str(e)}")
            wildfire_models[model_name] = None

def get_weather_data(lat, lon):
    """Get weather data for given coordinates using Open-Meteo free API"""
    try:
        # Open-Meteo API - free, no API key needed
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation', 'surface_pressure', 'weather_code'],
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'precipitation_unit': 'inch',
            'timezone': 'auto'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        
        # Map weather codes to descriptions
        weather_descriptions = {
            0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
            45: 'Fog', 48: 'Depositing rime fog', 51: 'Light drizzle', 53: 'Moderate drizzle', 
            55: 'Dense drizzle', 61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
            71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow', 80: 'Slight rain showers',
            81: 'Moderate rain showers', 82: 'Violent rain showers', 95: 'Thunderstorm',
            96: 'Thunderstorm with hail', 99: 'Thunderstorm with heavy hail'
        }
        
        weather_code = current.get('weather_code', 0)
        description = weather_descriptions.get(weather_code, 'Unknown')
        
        return {
            'temperature': round(current.get('temperature_2m', 75.0), 1),
            'humidity': int(current.get('relative_humidity_2m', 45)),
            'wind_speed': round(current.get('wind_speed_10m', 10.0), 1),
            'precipitation': round(current.get('precipitation', 0.0), 2),
            'pressure': round(current.get('surface_pressure', 1013.25), 2),
            'description': description,
            'weather_code': weather_code
        }
    except requests.RequestException as e:
        logger.error(f"Error fetching weather data from Open-Meteo API: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing weather data: {str(e)}")
        return None

def calculate_weather_based_risk(weather):
    """Calculate wildfire risk based on weather conditions when models aren't available"""
    if not weather:
        return 0.5  # Default moderate risk
    
    risk_score = 0.0
    
    # Temperature factor (higher temp = higher risk)
    temp = weather.get('temperature', 75)
    if temp > 90:
        risk_score += 0.3
    elif temp > 80:
        risk_score += 0.2
    elif temp > 70:
        risk_score += 0.1
    
    # Humidity factor (lower humidity = higher risk)
    humidity = weather.get('humidity', 50)
    if humidity < 20:
        risk_score += 0.3
    elif humidity < 30:
        risk_score += 0.2
    elif humidity < 40:
        risk_score += 0.1
    
    # Wind speed factor (higher wind = higher risk)
    wind = weather.get('wind_speed', 10)
    if wind > 25:
        risk_score += 0.2
    elif wind > 15:
        risk_score += 0.1
    
    # Precipitation factor (less rain = higher risk)
    precipitation = weather.get('precipitation', 0)
    if precipitation < 0.01:
        risk_score += 0.2
    elif precipitation < 0.1:
        risk_score += 0.1
    
    # Weather code factor (clear/hot conditions = higher risk)
    weather_code = weather.get('weather_code', 0)
    if weather_code in [0, 1]:  # Clear or mainly clear
        risk_score += 0.1
    
    # Normalize to 0-1 range
    risk_score = min(risk_score, 1.0)
    return risk_score

def simple_gpt2_response(question):
    """Enhanced wildfire assistant with intelligent pattern matching"""
    # More comprehensive response patterns with context awareness
    patterns = {
        # Prevention and safety
        'prevent': {
            'keywords': ['prevent', 'stop', 'avoid', 'protect'],
            'responses': [
                'Key wildfire prevention steps: 1) Create defensible space around your property by clearing vegetation 30+ feet from structures, 2) Use fire-resistant landscaping and building materials, 3) Maintain equipment and vehicles properly, 4) Follow local fire restrictions and burn bans, 5) Never leave fires unattended.',
                'To prevent wildfires: Keep your property clear of dead vegetation, maintain a safe distance between trees and your home, use spark arresters on chimneys, and always fully extinguish campfires. During high-risk periods, avoid activities that could create sparks.',
                'Wildfire prevention is everyone\'s responsibility. Clear brush and flammable materials from around your home, keep your roof and gutters clean, have multiple evacuation routes planned, and stay informed about local fire conditions and restrictions.'
            ]
        },
        'safety': {
            'keywords': ['safe', 'evacuate', 'escape', 'emergency'],
            'responses': [
                'Wildfire safety essentials: 1) Sign up for local emergency alerts, 2) Have a "go bag" ready with important documents and supplies, 3) Know multiple evacuation routes, 4) If trapped, find a large cleared area away from flammable materials, 5) Call 911 immediately.',
                'During a wildfire emergency: Leave immediately when ordered to evacuate, never drive through heavy smoke, keep car windows closed and use recirculated air, have a battery-powered radio for updates, and go to designated evacuation centers.',
                'Stay safe during wildfire season by monitoring weather conditions, keeping vehicles fueled and ready, having an emergency kit prepared, and practicing your family evacuation plan. Never ignore evacuation orders.'
            ]
        },
        'risk': {
            'keywords': ['risk', 'danger', 'likelihood', 'probability'],
            'responses': [
                'Wildfire risk varies based on several key factors: weather conditions (temperature, humidity, wind speed), vegetation density and moisture content, topography, recent precipitation, and human activities. Areas with dry vegetation and hot, windy conditions face the highest risk.',
                'Your wildfire risk depends on location, weather patterns, fuel availability (dry vegetation), and local fire history. High-risk conditions include: temperatures above 90°F, humidity below 30%, wind speeds over 25 mph, and little recent rainfall.',
                'Risk assessment considers the "fire triangle": fuel (vegetation), weather (heat, wind, low humidity), and ignition sources. Mountain areas, urban-wildland interfaces, and regions with dense dry vegetation typically face higher wildfire risks.'
            ]
        },
        'causes': {
            'keywords': ['cause', 'start', 'ignition', 'source'],
            'responses': [
                'Leading wildfire causes: 1) Human activities (85% of wildfires) including campfires, equipment use, arson, and power lines, 2) Lightning strikes (15% of wildfires), 3) Vehicle accidents and dragging chains, 4) Cigarettes and debris burning, 5) Equipment malfunctions.',
                'Most wildfires are human-caused: unattended campfires, power line failures, vehicle exhaust systems, cigarettes, fireworks, and equipment sparks. Natural causes like lightning are less common but can trigger large fires in remote areas.',
                'Wildfire ignition sources include electrical equipment, hot vehicle parts, dragging chains, welding, lawn mowers on dry grass, and improperly disposed smoking materials. Even small sparks can start major fires in dry conditions.'
            ]
        },
        'season': {
            'keywords': ['season', 'when', 'time', 'month'],
            'responses': [
                'Wildfire season varies by region: Western US (May-October), with peak danger June-September. California has two seasons: traditional (June-November) and "fire weather" (October-April). Climate change is extending fire seasons worldwide.',
                'Fire season timing depends on climate patterns. Typically: Spring - increasing temperatures dry vegetation, Summer - peak fire conditions with hot, dry weather, Fall - continued risk with Santa Ana/Diablo winds, Winter - lower risk but still possible.',
                'Fire season is becoming longer and more intense due to climate change. What used to be 3-4 month seasons now often extend 6+ months. Stay vigilant year-round, especially during drought conditions and heat waves.'
            ]
        },
        'weather': {
            'keywords': ['weather', 'wind', 'temperature', 'humidity', 'rain'],
            'responses': [
                'Critical wildfire weather factors: Low humidity (below 30%), high temperatures (above 85°F), strong winds (over 15 mph), and lack of precipitation. These conditions dry out vegetation and help fires spread rapidly.',
                'Weather drives wildfire behavior: Wind spreads fires quickly and unpredictably, low humidity dries vegetation creating more fuel, high temperatures increase fire intensity, and lack of rain leaves landscapes vulnerable.',
                'Monitor fire weather conditions: Red Flag Warnings indicate critical fire weather, Santa Ana and Diablo winds in California create extreme danger, and drought conditions increase fire risk for extended periods.'
            ]
        }
    }
    
    question_lower = question.lower()
    
    # Find the best matching pattern
    best_match = None
    max_matches = 0
    
    for category, pattern_data in patterns.items():
        matches = sum(1 for keyword in pattern_data['keywords'] if keyword in question_lower)
        if matches > max_matches:
            max_matches = matches
            best_match = pattern_data
    
    if best_match and max_matches > 0:
        # Select response based on question complexity/length
        responses = best_match['responses']
        if len(question) > 50:  # Longer questions get more detailed responses
            return responses[-1] if len(responses) > 2 else responses[0]
        elif len(question) > 20:  # Medium questions get medium response
            return responses[1] if len(responses) > 1 else responses[0]
        else:  # Short questions get concise response
            return responses[0]
    
    # Enhanced fallback responses
    fallbacks = [
        "I specialize in wildfire information. I can help with prevention strategies, safety protocols, risk assessment factors, common ignition causes, and seasonal fire patterns. What specific aspect interests you?",
        "As your wildfire assistant, I can provide guidance on fire safety, evacuation planning, risk reduction, weather impacts, and prevention measures. How can I help keep you informed and prepared?",
        "I'm here to help with wildfire-related questions about safety, prevention, risk factors, causes, and seasonal patterns. What would you like to learn about fire protection and preparedness?"
    ]
    
    # Rotate through fallback responses for variety
    import time
    fallback_index = int(time.time()) % len(fallbacks)
    return fallbacks[fallback_index]

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_wildfire_risk():
    """API endpoint for wildfire risk prediction"""
    try:
        data = request.get_json()
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))
        
        # Get weather data
        weather = get_weather_data(lat, lon)
        
        # Create feature array for prediction
        # Note: You'll need to adjust these features based on your model's expected inputs
        features = np.array([[
            lat,
            lon, 
            weather['temperature'] if weather else 75.0,
            weather['humidity'] if weather else 45.0,
            weather['wind_speed'] if weather else 10.0,
            weather['precipitation'] if weather else 0.1
        ]])
        
        # Check if any models are available
        available_models = [name for name, model in wildfire_models.items() if model is not None]
        
        if not available_models:
            logger.warning("No machine learning models available - using weather-based risk estimation")
            # Provide weather-based risk estimation when models aren't available
            weather_risk = calculate_weather_based_risk(weather)
            predictions = {
                'ultimate_risk': weather_risk,
                'risk': weather_risk,
                'cause': weather_risk * 0.8  # Slightly lower for cause prediction
            }
            warnings = ["Machine learning models are currently unavailable. Risk assessment based on weather conditions only."]
        else:
            # Make predictions with available models
            predictions = {}
            warnings = []
            
            for model_name, model in wildfire_models.items():
                if model is not None:
                    try:
                        # Attempt prediction - may need to adjust based on your model's expected features
                        pred = model.predict(features)[0]
                        predictions[model_name] = float(pred)
                    except Exception as e:
                        logger.error(f"Error predicting with {model_name}: {str(e)}")
                        # Use weather-based fallback for this specific model
                        weather_risk = calculate_weather_based_risk(weather)
                        predictions[model_name] = weather_risk
                        warnings.append(f"{model_name.replace('_', ' ').title()} model unavailable - using weather-based estimation")
                else:
                    # Model failed to load
                    weather_risk = calculate_weather_based_risk(weather)
                    predictions[model_name] = weather_risk
                    warnings.append(f"{model_name.replace('_', ' ').title()} model failed to load - using weather-based estimation")
        
        return jsonify({
            'predictions': predictions,
            'weather': weather,
            'location': {'latitude': lat, 'longitude': lon},
            'warnings': warnings if 'warnings' in locals() else [],
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'status': 'error'}), 500

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """API endpoint for weather data"""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if lat is None or lon is None:
        return jsonify({'error': 'Latitude and longitude required'}), 400
    
    weather = get_weather_data(lat, lon)
    if weather:
        return jsonify(weather)
    else:
        return jsonify({'error': 'Weather data unavailable'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for GPT-2 chat functionality"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        response = simple_gpt2_response(question)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Chat failed', 'status': 'error'}), 500

@app.route('/api/status')
def status():
    """API endpoint to check model and system status"""
    model_status = {}
    for model_name, model in wildfire_models.items():
        model_status[model_name] = 'loaded' if model is not None else 'failed'
    
    return jsonify({
        'models': model_status,
        'status': 'running'
    })

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the app
    # Get debug mode from environment, default to False for production security
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1']
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)