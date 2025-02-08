import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import json

class ReadTimeModel:
    def __init__(self):
        self.model = None
        self.genre_encoder = LabelEncoder()
        self.feature_names = ['pages', 'reading_level', 'genre', 'available_time', 'reading_speed']
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic data for training the model."""
        np.random.seed(42)
        
        # Generate features
        pages = np.random.randint(50, 1000, n_samples)
        reading_level = np.random.randint(1, 4, n_samples)
        genres = np.random.choice(['fiction', 'non-fiction', 'textbook', 'technical'], n_samples)
        available_time = np.random.randint(15, 240, n_samples)
        reading_speed = np.random.randint(1, 4, n_samples)
        
        # Create base reading time (hours) with some realistic assumptions
        base_time = pages * (1/25)  # Assume average 25 pages per hour base rate
        
        # Apply modifiers based on features
        level_modifier = 1 + (0.2 * (reading_level - 2))  # ±20% based on level
        genre_modifier = np.where(genres == 'fiction', 0.8,
                        np.where(genres == 'non-fiction', 1.0,
                        np.where(genres == 'textbook', 1.3, 1.5)))  # Technical is 1.5
        speed_modifier = 1 + (0.3 * (2 - reading_speed))  # ±30% based on speed
        
        # Calculate total reading hours with some random noise
        total_hours = (base_time * level_modifier * genre_modifier * speed_modifier * 
                      np.random.normal(1, 0.1, n_samples))
        
        # Calculate recommended days based on available_time
        days_to_complete = np.ceil((total_hours * 60) / available_time)
        
        # Generate confidence scores (higher for more common combinations)
        confidence = np.random.uniform(0.7, 0.95, n_samples)
        confidence = np.where((reading_level == 2) & (reading_speed == 2), confidence + 0.1, confidence)
        confidence = np.clip(confidence, 0, 1)
        
        # Create DataFrame
        data = pd.DataFrame({
            'pages': pages,
            'reading_level': reading_level,
            'genre': genres,
            'available_time': available_time,
            'reading_speed': reading_speed,
            'total_hours': total_hours,
            'days_to_complete': days_to_complete,
            'confidence': confidence
        })
        
        return data
    
    def train(self, data):
        """Train the model on the provided data."""
        # Encode categorical variables
        X = data[self.feature_names].copy()
        X['genre'] = self.genre_encoder.fit_transform(X['genre'])
        
        # Prepare target variables
        y_hours = data['total_hours']
        y_days = data['days_to_complete']
        y_confidence = data['confidence']
        
        # Train models for each target
        self.model_hours = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_days = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_confidence = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model_hours.fit(X, y_hours)
        self.model_days.fit(X, y_days)
        self.model_confidence.fit(X, y_confidence)
        
    def predict(self, features):
        """Make predictions for new data."""
        X = pd.DataFrame([features])
        X['genre'] = self.genre_encoder.transform([X['genre'].iloc[0]])
        
        hours = self.model_hours.predict(X)[0]
        days = self.model_days.predict(X)[0]
        confidence = self.model_confidence.predict(X)[0]
        
        return {
            'total_hours': round(hours, 1),
            'days_to_complete': int(days),
            'confidence': round(confidence, 2)
        }
    
    def get_reading_tips(self, genre):
        """Generate reading tips based on genre."""
        tips = {
            'fiction': [
                "Find a quiet, comfortable spot to immerse yourself in the story",
                "Take notes on characters and plot points",
                "Try to read at least one chapter in each sitting",
                "Visualize the scenes as you read"
            ],
            'non-fiction': [
                "Start with the table of contents to understand the structure",
                "Take notes on key concepts and ideas",
                "Use sticky notes or highlights for important passages",
                "Summarize each chapter after reading"
            ],
            'textbook': [
                "Review chapter summaries before starting",
                "Use active recall techniques while reading",
                "Create mind maps for complex topics",
                "Take breaks every 45-60 minutes"
            ],
            'technical': [
                "Start with a quick skim to understand the structure",
                "Practice examples as you encounter them",
                "Keep reference materials handy",
                "Take detailed notes and create cheat sheets"
            ]
        }
        return tips.get(genre, [])

    def save_model(self, filename='readtime_model.pkl'):
        """Save the trained model to a file."""
        model_data = {
            'model_hours': self.model_hours,
            'model_days': self.model_days,
            'model_confidence': self.model_confidence,
            'genre_encoder': self.genre_encoder
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filename='readtime_model.pkl'):
        """Load a trained model from a file."""
        instance = cls()
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        instance.model_hours = model_data['model_hours']
        instance.model_days = model_data['model_days']
        instance.model_confidence = model_data['model_confidence']
        instance.genre_encoder = model_data['genre_encoder']
        return instance

# Training example
if __name__ == "__main__":
    # Create and train the model
    model = ReadTimeModel()
    data = model.generate_synthetic_data(n_samples=1000)
    model.train(data)
    
    # Save the model
    model.save_model()
    
    # Example prediction
    test_features = {
        'pages': 300,
        'reading_level': 2,
        'genre': 'fiction',
        'available_time': 60,
        'reading_speed': 2
    }
    
    prediction = model.predict(test_features)
    tips = model.get_reading_tips(test_features['genre'])
    
    print("Prediction:", json.dumps(prediction, indent=2))
    print("\nReading Tips:", json.dumps(tips, indent=2))