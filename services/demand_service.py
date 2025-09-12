"""
Demand prediction service using machine learning
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from concurrent.futures import ThreadPoolExecutor
import pickle
import os

from app.models.request_models import DemandPredictionRequest
from app.models.response_models import DemandPredictionResponse, DemandPrediction
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DemandPredictionService:
    """ML-based demand prediction service"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.city_encoder = LabelEncoder()
        self.is_trained = False
        self.model_accuracy = None
        self.model_path = "demand_model.pkl"
        self.encoder_path = "city_encoder.pkl"
        
        # Try to load pre-trained model
        asyncio.create_task(self._load_or_train_model())
    
    async def _load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
                await self._load_model()
                logger.info("Loaded pre-trained demand prediction model")
            else:
                await self._train_model()
                logger.info("Trained new demand prediction model")
        except Exception as e:
            logger.error(f"Model loading/training failed: {e}")
            await self._train_model()  # Fallback to training
    
    async def _load_model(self):
        """Load pre-trained model from disk"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            def load_models():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.encoder_path, 'rb') as f:
                    self.city_encoder = pickle.load(f)
                self.is_trained = True
                return True
            
            await loop.run_in_executor(executor, load_models)
    
    async def _save_model(self):
        """Save trained model to disk"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            def save_models():
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                with open(self.encoder_path, 'wb') as f:
                    pickle.dump(self.city_encoder, f)
                return True
            
            await loop.run_in_executor(executor, save_models)
    
    async def _train_model(self):
        """Train the demand prediction model"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, self._train_model_sync)
            
        if result:
            await self._save_model()
            self.is_trained = True
            logger.info(f"Model training completed. Accuracy: {self.model_accuracy:.3f}")
    
    def _train_model_sync(self) -> bool:
        """Synchronous model training"""
        try:
            # Generate synthetic training data
            training_data = self._generate_synthetic_data(2000)
            
            # Prepare features
            training_data['city_encoded'] = self.city_encoder.fit_transform(training_data['city'])
            
            features = ['city_encoded', 'month', 'day_of_week', 'is_weekend', 'rainfall']
            X = training_data[features]
            y = training_data['demand_kg']
            
            # Train model
            self.model.fit(X, y)
            
            # Calculate accuracy
            y_pred = self.model.predict(X)
            self.model_accuracy = r2_score(y, y_pred)
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic historical demand data"""
        cities = ['Colombo', 'Kandy', 'Galle', 'Jaffna', 'Anuradhapura', 
                 'Negombo', 'Kurunegala', 'Ratnapura', 'Batticaloa', 'Trincomalee']
        
        data = []
        start_date = datetime(2023, 1, 1)
        
        for i in range(n_samples):
            date = start_date + timedelta(days=np.random.randint(0, 365))
            city = np.random.choice(cities)
            
            # Seasonal factors
            month = date.month
            seasonal_factor = 1.0
            if month in [4, 5, 6]:  # New Year season
                seasonal_factor = 1.3
            elif month in [12, 1]:  # Christmas/New Year
                seasonal_factor = 1.2
            
            # Weekend factor
            is_weekend = 1 if date.weekday() >= 5 else 0
            weekend_factor = 1.1 if is_weekend else 1.0
            
            # Weather simulation
            rainfall = max(0, np.random.normal(5, 10))
            weather_factor = 0.8 if rainfall > 10 else 1.0
            
            # Base demand by city
            city_base_demand = {
                'Colombo': 500, 'Kandy': 300, 'Galle': 250, 'Jaffna': 200,
                'Anuradhapura': 180, 'Negombo': 400, 'Kurunegala': 220,
                'Ratnapura': 150, 'Batticaloa': 160, 'Trincomalee': 170
            }
            
            base_demand = city_base_demand.get(city, 200)
            final_demand = base_demand * seasonal_factor * weekend_factor * weather_factor
            final_demand += np.random.normal(0, 20)  # Add noise
            final_demand = max(0, final_demand)
            
            data.append({
                'date': date,
                'city': city,
                'month': month,
                'day_of_week': date.weekday(),
                'is_weekend': is_weekend,
                'rainfall': rainfall,
                'demand_kg': final_demand
            })
        
        return pd.DataFrame(data)
    
    async def predict_demand(self, request: DemandPredictionRequest) -> DemandPredictionResponse:
        """Main demand prediction endpoint"""
        try:
            start_time = time.time()
            
            if not self.is_trained:
                await self._train_model()
            
            if not self.is_trained:
                return DemandPredictionResponse(
                    success=False,
                    message="Model training failed",
                    predictions=[],
                    prediction_date=request.prediction_date,
                    model_accuracy_score=None,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    timestamp=datetime.utcnow()
                )
            
            # Make predictions for each city
            predictions = []
            
            for city in request.cities:
                demand = await self._predict_city_demand(
                    city, request.prediction_date, request.rainfall
                )
                
                # Determine confidence level based on model accuracy
                confidence = "high" if self.model_accuracy > 0.8 else "medium" if self.model_accuracy > 0.6 else "low"
                
                predictions.append(DemandPrediction(
                    city=city,
                    predicted_demand_kg=round(demand, 1),
                    confidence_level=confidence
                ))
            
            processing_time = time.time() - start_time
            
            return DemandPredictionResponse(
                success=True,
                message="Demand prediction completed successfully",
                predictions=predictions,
                prediction_date=request.prediction_date,
                model_accuracy_score=round(self.model_accuracy, 3) if self.model_accuracy else None,
                processing_time_seconds=round(processing_time, 2),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Demand prediction failed: {str(e)}")
            return DemandPredictionResponse(
                success=False,
                message=f"Prediction failed: {str(e)}",
                predictions=[],
                prediction_date=request.prediction_date,
                model_accuracy_score=None,
                processing_time_seconds=round(time.time() - start_time, 2),
                timestamp=datetime.utcnow()
            )
    
    async def _predict_city_demand(self, city: str, date: datetime, rainfall: float) -> float:
        """Predict demand for a specific city"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, self._predict_city_demand_sync, city, date, rainfall
            )
        
        return result
    
    def _predict_city_demand_sync(self, city: str, date: datetime, rainfall: float) -> float:
        """Synchronous city demand prediction"""
        try:
            # Encode city
            try:
                city_encoded = self.city_encoder.transform([city])[0]
            except ValueError:
                # If city not in training data, use average encoding
                city_encoded = len(self.city_encoder.classes_) // 2
            
            # Prepare features
            features = np.array([[
                city_encoded,
                date.month,
                date.weekday(),
                1 if date.weekday() >= 5 else 0,
                rainfall
            ]])
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            return max(0, prediction)
            
        except Exception as e:
            logger.error(f"City demand prediction failed for {city}: {e}")
            # Fallback to simple heuristic
            base_demands = {
                'Colombo': 500, 'Kandy': 300, 'Galle': 250, 'Jaffna': 200,
                'Anuradhapura': 180, 'Negombo': 400, 'Kurunegala': 220,
                'Ratnapura': 150, 'Batticaloa': 160, 'Trincomalee': 170
            }
            return base_demands.get(city, 200)
    
    async def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "is_trained": self.is_trained,
            "accuracy": self.model_accuracy,
            "features": ['city', 'month', 'day_of_week', 'is_weekend', 'rainfall'],
            "cities_supported": list(self.city_encoder.classes_) if self.is_trained else [],
            "model_type": "Random Forest Regressor"
        }