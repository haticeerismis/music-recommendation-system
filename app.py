from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
import base64
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

app = Flask(__name__)

df = pd.read_csv('spotify_features_data_2023.csv')
track_info = df[['id']].dropna().reset_index(drop=True)

df = df.drop(columns=['id', 'uri', 'track_href', 'analysis_url', 'type'])
df = df.sample(n=10000, random_state=42).reset_index(drop=True)
track_info = track_info.sample(n=10000, random_state=42).reset_index(drop=True)

def mood_label(row):
    if row['valence'] > 0.7:
        return 'happy'
    elif row['valence'] < 0.4:
        return 'sad'
    elif row['energy'] > 0.7:
        return 'energetic'
    else:
        return 'calm'

df['mood'] = df.apply(mood_label, axis=1)

X_mood = df.drop(columns=['mood'])
y_mood = df['mood']
X_train_mood, X_test_mood, y_train_mood, y_test_mood = train_test_split(X_mood, y_mood, test_size=0.2, random_state=42)
scaler_mood = StandardScaler()
X_train_mood = scaler_mood.fit_transform(X_train_mood)
X_test_mood = scaler_mood.transform(X_test_mood)
mood_model = LogisticRegression(max_iter=1000)
mood_model.fit(X_train_mood, y_train_mood)

df['discoverable'] = ((df['danceability'] > 0.6) & (df['energy'] > 0.6)).astype(int)
X_discover = df[['danceability', 'energy']]
y_discover = df['discoverable']
X_train_dis, X_test_dis, y_train_dis, y_test_dis = train_test_split(X_discover, y_discover, test_size=0.2, random_state=42)
scaler_dis = StandardScaler()
X_train_dis = scaler_dis.fit_transform(X_train_dis)
X_test_dis = scaler_dis.transform(X_test_dis)
discover_model = RandomForestClassifier()
discover_model.fit(X_train_dis, y_train_dis)

X_knn = df.drop(columns=['mood', 'discoverable'])
scaler_knn = StandardScaler()
X_knn_scaled = scaler_knn.fit_transform(X_knn)
similarity_matrix = cosine_similarity(X_knn_scaled)

def get_spotify_token():
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64_auth}"}
    data = {"grant_type": "client_credentials"}
    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def fetch_track_names(track_ids, token):
    headers = {"Authorization": f"Bearer {token}"}
    track_data = []
    for tid in track_ids:
        url = f"https://api.spotify.com/v1/tracks/{tid}"
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            track_data.append({'id': tid, 'name': data['name']})
    return track_data

@app.route('/')
def index():
    token = get_spotify_token()
    ids = track_info['id'].dropna().unique()[:80]
    song_list = fetch_track_names(ids, token)
    example_id = song_list[0]['id']
    return render_template('index.html', example_id=example_id, song_list=song_list)

@app.route('/api/recommend')
def api_recommend():
    r_type = request.args.get('type')

    if r_type == 'mood':
        mood = request.args.get('value')
        result = df[df['mood'] == mood].sample(3)
        links = [{'id': track_info.iloc[idx]['id']} for idx in result.index]
        return jsonify({'songs': links})

    elif r_type == 'discover':
        result = df[df['discoverable'] == 1].sample(3)
        links = [{'id': track_info.iloc[idx]['id']} for idx in result.index]
        return jsonify({'songs': links})

    elif r_type == 'similar':
        song_id = request.args.get('id')
        try:
            song_idx = track_info[track_info['id'] == song_id].index[0]
            similarity_scores = similarity_matrix[song_idx]
            similar_indices = similarity_scores.argsort()[::-1][1:10]  # Kendisi hariç en benzerler
            chosen = np.random.choice(similar_indices, size=3, replace=False)
            links = [{'id': track_info.iloc[i]['id']} for i in chosen]
            return jsonify({'songs': links})
        except:
            return jsonify({'error': 'ID geçersiz veya şarkı bulunamadı'}), 400

    return jsonify({'error': 'Geçersiz tür'}), 400 

if __name__ == '__main__':
    app.run(debug=True)