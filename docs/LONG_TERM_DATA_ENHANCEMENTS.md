# Long-Term Data Enhancements (3-12 Months)

Advanced features requiring significant infrastructure, external data sources, or machine learning sophistication.

---

## 1. Neural Network Architecture for Sequence Modeling

**Impact**: Very High  
**Difficulty**: High  
**Timeline**: 2-3 months

### Why It Matters
- Current XGBoost model treats each race independently
- Horse form is a *sequence* - past races predict future
- LSTM/Transformer can capture temporal patterns

### Architecture: Horse Form Sequence Model

```python
#!/usr/bin/env python3
"""models/sequence_model.py
LSTM-based model for horse racing prediction using form sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path


class HorseFormSequenceDataset(Dataset):
    """
    Dataset that creates sequences of past races for each horse entry.
    
    For each (horse, race) pair, creates:
    - Sequence of last N races (features from each)
    - Target: did horse win this race?
    """
    
    def __init__(self, df, seq_length=10, feature_cols=None):
        self.seq_length = seq_length
        self.feature_cols = feature_cols or [
            'pos_clean', 'class_num', 'dist_f', 'going_numeric',
            'field_size', 'or_numeric', 'btn_lengths', 'days_gap',
            'jockey_win_rate', 'trainer_win_rate'
        ]
        
        # Group by horse and create sequences
        self.samples = self._create_sequences(df)
    
    def _create_sequences(self, df):
        """Create (sequence, race_features, target) samples."""
        samples = []
        
        df = df.sort_values(['horse', 'date']).reset_index(drop=True)
        
        for horse in df['horse'].unique():
            horse_data = df[df['horse'] == horse].copy()
            
            for i in range(len(horse_data)):
                # Get past races (up to seq_length)
                start_idx = max(0, i - self.seq_length)
                past_races = horse_data.iloc[start_idx:i]
                
                if len(past_races) < 2:
                    continue  # Need some history
                
                # Current race features
                current_race = horse_data.iloc[i]
                target = int(current_race['pos_clean'] == 1)
                
                # Pad sequence if needed
                seq_features = past_races[self.feature_cols].values
                if len(seq_features) < self.seq_length:
                    padding = np.zeros((self.seq_length - len(seq_features), len(self.feature_cols)))
                    seq_features = np.vstack([padding, seq_features])
                
                # Current race static features (for attention)
                race_features = current_race[[
                    'class_num', 'dist_f', 'going_numeric', 'field_size', 'is_turf'
                ]].values
                
                samples.append({
                    'sequence': torch.FloatTensor(seq_features),
                    'race_features': torch.FloatTensor(race_features),
                    'target': torch.FloatTensor([target]),
                    'horse': horse,
                    'date': current_race['date']
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class HorseLSTMModel(nn.Module):
    """
    LSTM-based model for predicting race outcomes from form sequences.
    """
    
    def __init__(
        self, 
        seq_feature_dim=10, 
        race_feature_dim=5,
        hidden_dim=64, 
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=seq_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Race condition embedding
        self.race_embed = nn.Sequential(
            nn.Linear(race_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32)
        )
        
        # Attention layer (attend to relevant past races)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequence, race_features):
        batch_size = sequence.size(0)
        
        # Process sequence through LSTM
        lstm_out, (hidden, cell) = self.lstm(sequence)
        # lstm_out: [batch, seq_len, hidden_dim]
        
        # Self-attention over sequence
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Take last hidden state after attention
        seq_repr = attn_out[:, -1, :]  # [batch, hidden_dim]
        
        # Process race features
        race_repr = self.race_embed(race_features)  # [batch, 32]
        
        # Combine
        combined = torch.cat([seq_repr, race_repr], dim=1)
        
        # Classify
        output = self.classifier(combined)
        
        return output


class HorseTransformerModel(nn.Module):
    """
    Transformer-based model for horse racing prediction.
    Better at capturing long-range dependencies in form.
    """
    
    def __init__(
        self,
        seq_feature_dim=10,
        race_feature_dim=5,
        d_model=64,
        nhead=4,
        num_layers=3,
        dropout=0.3
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(seq_feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 20, d_model))  # Max 20 past races
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Race condition embedding
        self.race_embed = nn.Sequential(
            nn.Linear(race_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, sequence, race_features):
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        
        # Project input
        x = self.input_proj(sequence)  # [batch, seq, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer
        x = self.transformer(x)
        
        # Take CLS token output
        cls_output = x[:, 0, :]  # [batch, d_model]
        
        # Process race features
        race_repr = self.race_embed(race_features)
        
        # Combine and classify
        combined = torch.cat([cls_output, race_repr], dim=1)
        output = self.classifier(combined)
        
        return output


def train_sequence_model(
    train_loader, 
    val_loader, 
    model,
    epochs=50,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the sequence model."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    criterion = nn.BCELoss()
    
    best_auc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            sequence = batch['sequence'].to(device)
            race_features = batch['race_features'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(sequence, race_features)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequence = batch['sequence'].to(device)
                race_features = batch['race_features'].to(device)
                target = batch['target'].to(device)
                
                output = model(sequence, race_features)
                val_preds.extend(output.cpu().numpy().flatten())
                val_targets.extend(target.cpu().numpy().flatten())
        
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_targets, val_preds)
        
        scheduler.step(val_auc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'models/best_sequence_model.pt')
    
    return best_auc


if __name__ == '__main__':
    # Load data
    df = pd.read_parquet('data/processed/race_scores.parquet')
    
    # Split by date
    train_df = df[df['date'] < '2024-01-01']
    val_df = df[(df['date'] >= '2024-01-01') & (df['date'] < '2024-07-01')]
    test_df = df[df['date'] >= '2024-07-01']
    
    # Create datasets
    train_dataset = HorseFormSequenceDataset(train_df, seq_length=10)
    val_dataset = HorseFormSequenceDataset(val_df, seq_length=10)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Train LSTM model
    lstm_model = HorseLSTMModel()
    best_auc = train_sequence_model(train_loader, val_loader, lstm_model)
    print(f"Best LSTM AUC: {best_auc:.4f}")
    
    # Train Transformer model
    transformer_model = HorseTransformerModel()
    best_auc = train_sequence_model(train_loader, val_loader, transformer_model)
    print(f"Best Transformer AUC: {best_auc:.4f}")
```

---

## 2. Race-Level Contrastive Learning

**Impact**: High  
**Difficulty**: Very High  
**Timeline**: 2-3 months

### Why It Matters
- Standard models predict each horse independently
- Race outcome depends on relative abilities
- Learn embeddings that capture "better than" relationships

### Implementation: Contrastive Horse Embeddings

```python
#!/usr/bin/env python3
"""models/contrastive_model.py
Contrastive learning for horse racing: learn embeddings where
winners are closer to each other than to losers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class RaceContrastiveDataset(Dataset):
    """
    Create contrastive pairs from races.
    
    Positive pair: (winner features, 2nd place features)
    Negative pairs: (winner features, 4th+ place features)
    
    Goal: Learn embeddings where winner is closer to 2nd than to losers.
    """
    
    def __init__(self, df, feature_cols):
        self.feature_cols = feature_cols
        self.triplets = self._create_triplets(df)
    
    def _create_triplets(self, df):
        """Create (anchor, positive, negative) triplets."""
        triplets = []
        
        # Group by race
        for (date, course, off), race_df in df.groupby(['date', 'course', 'off']):
            race_df = race_df.sort_values('pos_clean')
            
            if len(race_df) < 4:
                continue
            
            # Anchor = winner
            anchor = race_df.iloc[0][self.feature_cols].values
            
            # Positive = 2nd or 3rd place (competitive finisher)
            positive = race_df.iloc[1][self.feature_cols].values
            
            # Negative = 4th+ place (not competitive)
            for i in range(3, len(race_df)):
                negative = race_df.iloc[i][self.feature_cols].values
                
                triplets.append({
                    'anchor': torch.FloatTensor(anchor),
                    'positive': torch.FloatTensor(positive),
                    'negative': torch.FloatTensor(negative)
                })
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]


class ContrastiveHorseEncoder(nn.Module):
    """
    Encoder that maps horse features to embedding space.
    """
    
    def __init__(self, input_dim, embed_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim)
        )
        
        # L2 normalize embeddings
        self.normalize = True
    
    def forward(self, x):
        embed = self.encoder(x)
        if self.normalize:
            embed = F.normalize(embed, p=2, dim=1)
        return embed


class TripletLoss(nn.Module):
    """
    Triplet margin loss for contrastive learning.
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # Euclidean distances
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def train_contrastive_encoder(
    train_loader,
    encoder,
    epochs=100,
    lr=1e-3,
    device='cpu'
):
    """Train contrastive encoder."""
    
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = TripletLoss(margin=1.0)
    
    for epoch in range(epochs):
        encoder.train()
        total_loss = 0
        
        for batch in train_loader:
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)
            
            optimizer.zero_grad()
            
            # Encode all three
            anchor_embed = encoder(anchor)
            pos_embed = encoder(positive)
            neg_embed = encoder(negative)
            
            loss = criterion(anchor_embed, pos_embed, neg_embed)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
    
    return encoder


def use_contrastive_embeddings_for_prediction(encoder, race_df, feature_cols):
    """
    Use trained encoder to predict race outcomes.
    
    Strategy: Horse with highest cosine similarity to "ideal winner" wins.
    """
    encoder.eval()
    
    with torch.no_grad():
        features = torch.FloatTensor(race_df[feature_cols].values)
        embeddings = encoder(features).numpy()
    
    # Calculate "winner-likeness" as distance from origin
    # (normalized embeddings with high magnitude = more winner-like)
    winner_scores = np.linalg.norm(embeddings, axis=1)
    
    # Or use: average embedding of past winners as reference
    # winner_scores = cosine_similarity(embeddings, winner_reference_embedding)
    
    race_df['contrastive_score'] = winner_scores
    race_df['contrastive_rank'] = race_df['contrastive_score'].rank(ascending=False)
    
    return race_df
```

---

## 3. External Data Integration: Weather & Track Conditions

**Impact**: Medium-High  
**Difficulty**: Medium  
**Timeline**: 1-2 months

### Why It Matters
- Real-time going changes affect outcomes
- Temperature affects horses differently
- Wind direction matters at certain courses

### Implementation

```python
#!/usr/bin/env python3
"""scripts/fetch_weather_data.py
Fetch weather data for race locations.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Course coordinates (major UK courses)
COURSE_COORDS = {
    'Ascot': (51.4096, -0.6782),
    'Newmarket': (52.2413, 0.3865),
    'York': (53.9246, -1.0540),
    'Doncaster': (53.5168, -1.1100),
    'Cheltenham': (51.9131, -2.0677),
    'Epsom': (51.3191, -0.2602),
    'Goodwood': (50.8986, -0.7540),
    'Sandown': (51.3650, -0.3580),
    'Newbury': (51.4006, -1.3079),
    'Kempton': (51.4187, -0.4105),
    'Haydock': (53.4754, -2.6343),
    'Chester': (53.1827, -2.9075),
    'Lingfield': (51.1669, -0.0187),
    'Wolverhampton': (52.5922, -2.0927)
}


def fetch_weather_openweather(lat, lon, date, api_key):
    """
    Fetch historical weather from OpenWeatherMap.
    
    Note: Historical data requires paid plan. Free tier = current only.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            'temp_c': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind'].get('deg', 0),
            'rain_1h': data.get('rain', {}).get('1h', 0),
            'condition': data['weather'][0]['main']
        }
    return None


def fetch_weather_visualcrossing(lat, lon, date_str, api_key):
    """
    Fetch historical weather from Visual Crossing (better for historical).
    Free tier: 1000 calls/day.
    """
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date_str}"
    params = {
        'key': api_key,
        'unitGroup': 'metric',
        'include': 'hours'
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        day = data['days'][0]
        return {
            'temp_max': day['tempmax'],
            'temp_min': day['tempmin'],
            'temp_avg': day['temp'],
            'precip': day.get('precip', 0),
            'humidity': day['humidity'],
            'wind_speed': day['windspeed'],
            'wind_dir': day.get('winddir', 0),
            'conditions': day['conditions']
        }
    return None


def engineer_weather_features(df, weather_df):
    """
    Add weather features to race data.
    """
    print("\nEngineering weather features...")
    
    df = df.merge(
        weather_df[['course', 'date', 'temp_avg', 'precip', 'humidity', 'wind_speed']],
        on=['course', 'date'],
        how='left'
    )
    
    # Fill missing with course/month averages
    for col in ['temp_avg', 'precip', 'humidity', 'wind_speed']:
        df[col] = df.groupby(['course', df['date'].dt.month])[col].transform(
            lambda x: x.fillna(x.mean())
        )
    
    # Derived features
    # Rain in last 24h affects going
    df['rain_24h'] = df['precip']
    df['is_wet'] = (df['precip'] > 5).astype(int)
    
    # Temperature bins (some horses prefer hot/cold)
    df['temp_bin'] = pd.cut(
        df['temp_avg'],
        bins=[-10, 5, 12, 18, 40],
        labels=['cold', 'cool', 'mild', 'warm']
    )
    
    # Wind strength (affects some courses more)
    df['is_windy'] = (df['wind_speed'] > 25).astype(int)  # >25 km/h
    
    print(f"  Weather features: temp_avg, precip, humidity, wind_speed, rain_24h, is_wet")
    
    return df


# Track condition monitoring
def fetch_going_updates(date_str):
    """
    Fetch going updates from Racing API or scrape from BHA.
    Going can change during the day.
    """
    # This would connect to Racing API /going-updates endpoint
    # or scrape from AtTheRaces/Racing Post
    pass


def calculate_going_change_impact(official_going, current_going):
    """
    Calculate impact of going change from declared to race-time.
    """
    GOING_VALUES = {
        'Firm': 1, 'Good to Firm': 2, 'Good': 3,
        'Good to Soft': 4, 'Soft': 5, 'Heavy': 6
    }
    
    official_val = GOING_VALUES.get(official_going, 3)
    current_val = GOING_VALUES.get(current_going, 3)
    
    change = current_val - official_val
    
    return {
        'going_softened': 1 if change > 0 else 0,
        'going_firmed': 1 if change < 0 else 0,
        'going_change_magnitude': abs(change)
    }
```

---

## 4. Computer Vision: Race Replay Analysis

**Impact**: Very High (unique edge)  
**Difficulty**: Very High  
**Timeline**: 6-12 months

### Why It Matters
- Visual tells that humans miss
- Trouble in running (interference, bad luck)
- Running style analysis from actual footage
- Completely unique data source

### Conceptual Architecture

```python
#!/usr/bin/env python3
"""models/video_analysis.py
Analyze race replays to extract features.

This is a conceptual framework - full implementation requires:
- Video download pipeline
- Object detection for horses/jockeys
- Tracking algorithms
- Action recognition models
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import r3d_18, R3D_18_Weights
import cv2
import numpy as np
from pathlib import Path


class RaceReplayAnalyzer:
    """
    Analyze race replay videos to extract features.
    """
    
    def __init__(self):
        # Pre-trained 3D CNN for video understanding
        self.video_model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.video_model.eval()
        
        # Object detection for horses
        # Would use YOLO or similar fine-tuned on horse racing
        self.horse_detector = None  # Placeholder
        
        # Horse re-identification model
        # Fine-tuned on silks/colors to track specific horses
        self.horse_reid = None  # Placeholder
    
    def extract_frames(self, video_path, fps=5):
        """Extract frames from video at specified fps."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(original_fps / fps)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_trouble_in_running(self, frames, horse_id):
        """
        Detect if horse experienced trouble (interference, blocked).
        
        Returns:
            dict: Trouble indicators
        """
        # This would use:
        # 1. Horse tracking to follow specific horse
        # 2. Trajectory analysis to detect sudden direction changes
        # 3. Proximity detection for interference
        
        trouble_indicators = {
            'was_blocked': False,
            'suffered_interference': False,
            'had_wide_run': False,
            'slow_start': False,
            'trouble_severity': 0  # 0-10 scale
        }
        
        return trouble_indicators
    
    def extract_running_style_from_video(self, frames, horse_id):
        """
        Extract running style features from video.
        More accurate than in-running positions.
        """
        running_style = {
            'position_1f': None,  # Position at 1 furlong
            'position_2f': None,
            'position_half': None,  # Halfway
            'position_final': None,
            'ground_covered': None,  # Total distance (wide = more)
            'finishing_speed': None,  # Acceleration in final furlong
            'stride_efficiency': None  # Placeholder for gait analysis
        }
        
        return running_style
    
    def calculate_visual_form(self, video_path, horse_id):
        """
        Main function: extract all visual features.
        """
        frames = self.extract_frames(video_path)
        
        trouble = self.detect_trouble_in_running(frames, horse_id)
        running = self.extract_running_style_from_video(frames, horse_id)
        
        # Combine into feature vector
        visual_features = {
            **trouble,
            **running,
            'had_unlucky_run': trouble['was_blocked'] or trouble['suffered_interference'],
            'visual_form_score': self._calculate_visual_score(trouble, running)
        }
        
        return visual_features
    
    def _calculate_visual_score(self, trouble, running):
        """
        Calculate overall visual form score.
        Adjusts raw finishing position for trouble/luck.
        """
        base_score = 50
        
        # Trouble adjustments
        if trouble.get('was_blocked'):
            base_score += 10  # Deserved to finish better
        if trouble.get('had_wide_run'):
            base_score += 5  # Covered extra ground
        if trouble.get('slow_start'):
            base_score += 3  # Lost ground at start
        
        return base_score


# Simplified: analyze race photos instead of video
class RacePhotoAnalyzer:
    """
    Analyze finish line photos for more accessible visual analysis.
    """
    
    def __init__(self):
        # Use pre-trained ResNet for image features
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone.eval()
    
    def analyze_finish_photo(self, image_path):
        """
        Analyze finish line photo.
        """
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.backbone(image_tensor).squeeze()
        
        return features.numpy()
```

---

## 5. Betting Market Microstructure

**Impact**: Very High  
**Difficulty**: High  
**Timeline**: 3-6 months

### Why It Matters
- Smart money moves markets
- Order flow reveals information
- Can detect when sharps are betting

### Implementation

```python
#!/usr/bin/env python3
"""scripts/market_microstructure.py
Analyze betting market microstructure from exchange data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


class MarketAnalyzer:
    """
    Analyze Betfair exchange market data.
    """
    
    def __init__(self):
        self.snapshots = []
        self.trades = []
    
    def add_snapshot(self, timestamp, market_data):
        """
        Add market snapshot.
        
        market_data: dict of {selection_id: {back_price, lay_price, back_size, lay_size}}
        """
        self.snapshots.append({
            'timestamp': timestamp,
            'data': market_data
        })
    
    def calculate_vwap(self, selection_id, time_window_minutes=60):
        """
        Calculate Volume Weighted Average Price.
        Better represents "true" price than last traded.
        """
        recent_trades = [
            t for t in self.trades
            if t['selection_id'] == selection_id
            and (datetime.now() - t['timestamp']).seconds < time_window_minutes * 60
        ]
        
        if not recent_trades:
            return None
        
        total_value = sum(t['price'] * t['size'] for t in recent_trades)
        total_volume = sum(t['size'] for t in recent_trades)
        
        return total_value / total_volume if total_volume > 0 else None
    
    def detect_steam_move(self, selection_id, threshold_pct=10):
        """
        Detect if selection is being "steamed" (heavily backed).
        
        Returns:
            dict: Steam detection results
        """
        if len(self.snapshots) < 2:
            return {'is_steamer': False, 'steam_strength': 0}
        
        first_snapshot = self.snapshots[0]['data'].get(selection_id, {})
        last_snapshot = self.snapshots[-1]['data'].get(selection_id, {})
        
        if not first_snapshot or not last_snapshot:
            return {'is_steamer': False, 'steam_strength': 0}
        
        opening_price = first_snapshot.get('back_price', 0)
        current_price = last_snapshot.get('back_price', 0)
        
        if opening_price <= 0:
            return {'is_steamer': False, 'steam_strength': 0}
        
        pct_change = (current_price - opening_price) / opening_price * 100
        
        # Negative pct = price shortened (backed)
        is_steamer = pct_change < -threshold_pct
        steam_strength = abs(pct_change) if is_steamer else 0
        
        return {
            'is_steamer': is_steamer,
            'steam_strength': steam_strength,
            'price_move': pct_change,
            'volume_ratio': self._calculate_volume_ratio(selection_id)
        }
    
    def _calculate_volume_ratio(self, selection_id):
        """
        Calculate ratio of back volume to lay volume.
        High ratio = more bettors backing (bullish).
        """
        back_volume = sum(
            s['data'].get(selection_id, {}).get('back_size', 0)
            for s in self.snapshots
        )
        lay_volume = sum(
            s['data'].get(selection_id, {}).get('lay_size', 0)
            for s in self.snapshots
        )
        
        if lay_volume == 0:
            return float('inf')
        
        return back_volume / lay_volume
    
    def identify_sharp_money(self, selection_id):
        """
        Attempt to identify if professional bettors ("sharps") are betting.
        
        Indicators:
        - Large single bets that move the market
        - Bets placed at inefficient times (late night)
        - Counter-public sentiment moves
        """
        indicators = {
            'large_bet_detected': False,
            'off_peak_activity': False,
            'counter_trend_move': False,
            'sharp_score': 0
        }
        
        # This would analyze order flow if available
        # Betfair API provides matched bet history
        
        return indicators
    
    def calculate_market_efficiency(self, market_data):
        """
        Calculate how efficient/accurate the market is.
        Overround = sum of implied probabilities - 1.
        Lower = more efficient.
        """
        back_prices = [
            data.get('back_price', 100)
            for data in market_data.values()
        ]
        
        implied_probs = [1 / p for p in back_prices if p > 0]
        overround = sum(implied_probs) - 1
        
        return {
            'overround': overround,
            'market_efficiency': 1 - overround  # 0-1, higher = better
        }


def engineer_market_microstructure_features(df, market_analyzer):
    """
    Add market microstructure features to predictions.
    """
    print("\nEngineering market microstructure features...")
    
    features = []
    
    for _, row in df.iterrows():
        selection_id = row.get('selection_id')
        
        # Steam detection
        steam = market_analyzer.detect_steam_move(selection_id)
        
        # Sharp money detection
        sharp = market_analyzer.identify_sharp_money(selection_id)
        
        features.append({
            'is_steamer': steam['is_steamer'],
            'steam_strength': steam['steam_strength'],
            'price_move': steam['price_move'],
            'volume_ratio': steam['volume_ratio'],
            'sharp_score': sharp['sharp_score']
        })
    
    features_df = pd.DataFrame(features)
    
    for col in features_df.columns:
        df[col] = features_df[col]
    
    return df
```

---

## 6. Ensemble Model Architecture

**Impact**: High  
**Difficulty**: Medium-High  
**Timeline**: 2-3 months

### Why It Matters
- Different models capture different patterns
- Combining predictions reduces variance
- Can weight models by race type

### Implementation

```python
#!/usr/bin/env python3
"""models/ensemble.py
Ensemble multiple models for robust predictions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import pickle
from pathlib import Path


class RacingEnsemble:
    """
    Ensemble of multiple models with learned weights.
    """
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.calibrators = {}
        
    def add_model(self, name, model, model_type='sklearn'):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model identifier
            model: Trained model object
            model_type: 'sklearn', 'xgboost', 'pytorch', 'manual'
        """
        self.models[name] = {
            'model': model,
            'type': model_type
        }
    
    def get_prediction(self, model_name, X):
        """Get prediction from a single model."""
        model_info = self.models[model_name]
        model = model_info['model']
        model_type = model_info['type']
        
        if model_type in ['sklearn', 'xgboost']:
            return model.predict_proba(X)[:, 1]
        elif model_type == 'pytorch':
            import torch
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                return model(X_tensor).numpy().flatten()
        elif model_type == 'manual':
            # Manual model returns probabilities directly
            return model.predict(X)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def calibrate_predictions(self, predictions, y_true, model_name):
        """
        Calibrate model predictions using isotonic regression.
        """
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(predictions, y_true)
        
        self.calibrators[model_name] = calibrator
        
        return calibrator.transform(predictions)
    
    def train_meta_model(self, X, y, model_names=None):
        """
        Train meta-model to learn optimal ensemble weights.
        
        Args:
            X: Feature matrix
            y: True outcomes
            model_names: Which models to include (default: all)
        """
        model_names = model_names or list(self.models.keys())
        
        # Get predictions from each model
        meta_features = []
        for name in model_names:
            preds = self.get_prediction(name, X)
            meta_features.append(preds)
        
        meta_X = np.column_stack(meta_features)
        
        # Train stacking meta-model
        self.meta_model = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000
        )
        self.meta_model.fit(meta_X, y)
        
        # Store model order
        self.meta_model_order = model_names
        
        print(f"Meta-model weights: {dict(zip(model_names, self.meta_model.coef_[0]))}")
    
    def predict(self, X, method='meta'):
        """
        Get ensemble prediction.
        
        Args:
            X: Feature matrix
            method: 'meta' (learned weights), 'average', 'max'
        
        Returns:
            Array of win probabilities
        """
        # Get predictions from each model
        predictions = {}
        for name in self.models.keys():
            preds = self.get_prediction(name, X)
            
            # Apply calibration if available
            if name in self.calibrators:
                preds = self.calibrators[name].transform(preds)
            
            predictions[name] = preds
        
        if method == 'meta' and self.meta_model is not None:
            # Use meta-model
            meta_X = np.column_stack([
                predictions[name] for name in self.meta_model_order
            ])
            return self.meta_model.predict_proba(meta_X)[:, 1]
        
        elif method == 'average':
            # Simple average
            return np.mean(list(predictions.values()), axis=0)
        
        elif method == 'max':
            # Take most confident prediction
            return np.max(list(predictions.values()), axis=0)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict_by_race_type(self, X, race_type_features):
        """
        Use different ensemble weights for different race types.
        """
        # This would have separate meta-models for:
        # - Sprints vs distance races
        # - Handicaps vs conditions races
        # - Maidens vs open races
        # - Flat vs jumps
        pass
    
    def save(self, path):
        """Save ensemble to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'meta_model': self.meta_model,
                'meta_model_order': getattr(self, 'meta_model_order', []),
                'calibrators': self.calibrators
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load ensemble from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        ensemble = cls()
        ensemble.models = data['models']
        ensemble.meta_model = data['meta_model']
        ensemble.meta_model_order = data.get('meta_model_order', [])
        ensemble.calibrators = data['calibrators']
        
        return ensemble


# Example: Build ensemble from existing models
def build_racing_ensemble():
    """
    Build ensemble from multiple trained models.
    """
    ensemble = RacingEnsemble()
    
    # Load XGBoost model (current production)
    with open('models/horse_win_predictor.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    ensemble.add_model('xgboost', xgb_model, 'sklearn')
    
    # Load LSTM sequence model (if trained)
    try:
        import torch
        lstm_model = HorseLSTMModel()
        lstm_model.load_state_dict(torch.load('models/best_sequence_model.pt'))
        lstm_model.eval()
        ensemble.add_model('lstm', lstm_model, 'pytorch')
    except:
        print("No LSTM model found, skipping")
    
    # Add rule-based model
    class RuleBasedModel:
        """Simple rules that sometimes beat ML."""
        
        def predict(self, X):
            # X is DataFrame with features
            probs = np.zeros(len(X))
            
            # Rule 1: First-time blinkers on exposed horse
            probs += 0.1 * (X['first_time_blinkers'] == 1)
            
            # Rule 2: Jockey-trainer combo with high win rate
            probs += 0.1 * (X['jockey_trainer_win_rate'] > 0.20)
            
            # Rule 3: Course winner returning
            probs += 0.1 * (X['cd_win_rate'] > 0.25)
            
            # Rule 4: Trainer in form
            probs += 0.05 * (X['trainer_win_rate_14d'] > 0.20)
            
            # Normalize
            return np.clip(probs + 0.05, 0.01, 0.99)
    
    ensemble.add_model('rules', RuleBasedModel(), 'manual')
    
    return ensemble
```

---

## 7. Reinforcement Learning for Bankroll Management

**Impact**: Medium-High  
**Difficulty**: Very High  
**Timeline**: 3-6 months

### Why It Matters
- Kelly criterion is optimal in theory, suboptimal in practice
- RL can learn bankroll strategies from simulation
- Adapt bet sizing to market conditions

### Conceptual Implementation

```python
#!/usr/bin/env python3
"""models/rl_bankroll.py
Reinforcement learning for bankroll management.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class BankrollEnvironment:
    """
    Simulated betting environment for RL training.
    """
    
    def __init__(self, historical_data, initial_bankroll=10000):
        self.data = historical_data
        self.initial_bankroll = initial_bankroll
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.bankroll = self.initial_bankroll
        self.current_race_idx = 0
        self.total_bets = 0
        self.total_wins = 0
        self.history = []
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state observation.
        
        State includes:
        - Current bankroll (normalized)
        - Recent win rate
        - Current race features
        - Betting opportunity quality
        """
        race = self._get_current_race()
        
        state = np.array([
            self.bankroll / self.initial_bankroll,  # Normalized bankroll
            self.total_wins / max(1, self.total_bets),  # Win rate
            len(self.history),  # Number of bets made
            race.get('model_edge', 0),  # Model's predicted edge
            race.get('odds', 3.0),  # Current odds
            race.get('race_score', 50) / 100,  # Race quality
            race.get('confidence', 0.5)  # Model confidence
        ])
        
        return state
    
    def _get_current_race(self):
        """Get current betting opportunity."""
        if self.current_race_idx >= len(self.data):
            return {}
        return self.data[self.current_race_idx]
    
    def step(self, action):
        """
        Take betting action and observe result.
        
        Args:
            action: 0 = don't bet, 1-10 = bet 1-10% of bankroll
        
        Returns:
            next_state, reward, done, info
        """
        race = self._get_current_race()
        
        if action == 0 or self.bankroll <= 0:
            # Skip this race
            bet_amount = 0
            outcome = 0
        else:
            # Bet (action * 1% of bankroll)
            bet_pct = action / 100
            bet_amount = self.bankroll * bet_pct
            
            # Simulate outcome
            won = race.get('result', 0) == 1  # Did our selection win?
            
            if won:
                outcome = bet_amount * (race.get('odds', 3.0) - 1)
                self.total_wins += 1
            else:
                outcome = -bet_amount
            
            self.total_bets += 1
            self.bankroll += outcome
            
            self.history.append({
                'bet': bet_amount,
                'outcome': outcome,
                'bankroll': self.bankroll
            })
        
        # Move to next race
        self.current_race_idx += 1
        done = self.current_race_idx >= len(self.data) or self.bankroll <= 0
        
        # Calculate reward
        # Reward = profit + survival bonus
        reward = outcome / self.initial_bankroll  # Normalize
        if self.bankroll > 0:
            reward += 0.001  # Small survival bonus
        if self.bankroll <= self.initial_bankroll * 0.5:
            reward -= 0.01  # Penalty for drawdown
        
        next_state = self._get_state()
        
        info = {
            'bankroll': self.bankroll,
            'bet_amount': bet_amount,
            'outcome': outcome,
            'roi': (self.bankroll - self.initial_bankroll) / self.initial_bankroll
        }
        
        return next_state, reward, done, info


class DQNAgent:
    """
    Deep Q-Network agent for bankroll management.
    """
    
    def __init__(self, state_dim=7, action_dim=11):
        # action_dim = 11: 0 (no bet) + 1-10% bet sizes
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Target network (for stable training)
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train on batch from replay memory."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_rl_bankroll_manager(env, agent, episodes=1000):
    """
    Train RL agent for bankroll management.
    """
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size=32)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent
```

---

## Implementation Roadmap

### Months 1-3: Neural Networks
- [ ] Implement LSTM form sequence model
- [ ] Train and validate vs XGBoost baseline
- [ ] A/B test on paper trades

### Months 3-6: External Data
- [ ] Weather data integration
- [ ] Betfair exchange data pipeline
- [ ] Market microstructure features

### Months 6-9: Ensemble & Optimization
- [ ] Build model ensemble
- [ ] Train meta-model for stacking
- [ ] Calibrate probability outputs

### Months 9-12: Advanced Features
- [ ] Race video analysis (if feasible)
- [ ] RL bankroll management
- [ ] Continuous model retraining pipeline

---

## Expected Impact

| Enhancement | Est. AUC Improvement | ROI Impact | Effort |
|-------------|---------------------|------------|--------|
| LSTM/Transformer | +0.02-0.04 | High | High |
| Contrastive Learning | +0.01-0.03 | Medium | Very High |
| Weather Data | +0.005-0.015 | Medium | Low |
| Market Microstructure | +0.02-0.04 | Very High | High |
| Video Analysis | +0.03-0.06 | Very High | Very High |
| Ensemble | +0.02-0.04 | High | Medium |
| RL Bankroll | N/A (risk reduction) | Medium | High |

**Total Long-Term Expected**: +0.08-0.20 AUC improvement (transformative for betting edge)

---

## Resource Requirements

### Computing
- GPU for neural network training (local or cloud)
- Storage for video data (if pursuing video analysis)
- Real-time data pipeline for market microstructure

### Data
- Betfair Exchange API subscription (~£50/month)
- Weather API (Visual Crossing free tier sufficient)
- Video storage/streaming (expensive, consider carefully)

### Time
- 10-20 hours/week for 12 months
- Significant research and experimentation

---

## Risk Considerations

### Model Risks
- Overfitting to historical data
- Concept drift as market evolves
- Black swan events (cancelled races, etc.)

### Operational Risks
- API rate limits and downtime
- Bet execution latency
- Bookmaker restrictions on winning accounts

### Financial Risks
- Even good models have variance
- Large drawdowns possible
- Never bet more than you can afford to lose

---

## Success Metrics

### Model Performance
- AUC > 0.70 on holdout data
- Calibrated probabilities (Brier score < 0.2)
- Stable performance across race types

### Betting Performance
- Positive ROI after 1000+ bets
- Sharpe ratio > 1.0
- Maximum drawdown < 30%

### Operational
- < 1 hour daily time commitment
- Automated data pipeline reliability > 99%
- Real-time odds capture within 1 second

---

## Final Notes

These long-term enhancements require significant investment but offer the potential for a genuine edge. The key is:

1. **Validate incrementally** - Each enhancement should show measurable improvement before proceeding
2. **Maintain discipline** - Don't let complexity introduce more bugs than improvements
3. **Focus on unique data** - Video analysis and market microstructure are hardest to replicate
4. **Keep it sustainable** - Build systems that require minimal daily intervention

Good luck on your horse racing prediction journey! 🏇
