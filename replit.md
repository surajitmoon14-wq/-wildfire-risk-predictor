# Overview

The USA Wildfire Risk Predictor is a web-based application that provides real-time wildfire risk assessment for any location in the United States. Users can click on an interactive map to get predictions about wildfire risk levels, potential causes, and comprehensive risk analysis using machine learning models. The application combines geospatial data, weather information, and predictive analytics to deliver actionable insights for wildfire preparedness and prevention.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Vanilla JavaScript with HTML5 and CSS3
- **UI Framework**: Bootstrap 5.1.3 for responsive design and components
- **Mapping**: Leaflet.js 1.9.4 for interactive map functionality using OpenStreetMap tiles
- **Architecture Pattern**: Single-page application with event-driven interactions
- **Map Interaction**: Click-based coordinate selection with marker placement and removal

## Backend Architecture
- **Framework**: Flask (Python) web framework with CORS support
- **API Design**: RESTful endpoints for prediction services
- **Model Loading**: Pickle-based machine learning model serialization and loading
- **Environment Configuration**: Environment-based settings for development vs production
- **Error Handling**: Comprehensive logging with Python's logging module

## Machine Learning Models
- **Model Storage**: Three separate pickle files for different prediction types:
  - Ultimate risk assessment model
  - General wildfire risk model  
  - Wildfire cause prediction model
- **Model Management**: Global model loading strategy with error handling
- **Data Processing**: Pandas and NumPy for data manipulation and numerical computations

## Security Architecture
- **CORS Policy**: Environment-specific CORS configuration (restrictive for production, permissive for development)
- **Session Management**: Environment-based secret key configuration
- **Input Validation**: Server-side validation for API requests

# External Dependencies

## Mapping and Geospatial Services
- **OpenStreetMap**: Tile layer provider for map visualization
- **Leaflet.js**: Open-source mapping library for interactive map functionality

## Frontend Libraries
- **Bootstrap 5.1.3**: CSS framework for responsive UI components and styling
- **Leaflet 1.9.4**: JavaScript library for mobile-friendly interactive maps

## Python Libraries
- **Flask**: Web framework for API development and routing
- **Flask-CORS**: Cross-origin resource sharing extension
- **Pandas**: Data manipulation and analysis library
- **NumPy**: Numerical computing library for array operations
- **Requests**: HTTP library for external API calls
- **Pickle**: Python serialization for model persistence

## Development and Production
- **Environment Variables**: Configuration management for deployment settings
- **Session Security**: Environment-based secret key management for secure sessions