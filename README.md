# 🎬 WatchWise — AI-Powered Movie Discovery Platform

<div align="center">

![WatchWise Banner](https://img.shields.io/badge/WatchWise-AI%20Movie%20Discovery-FFD700?style=for-the-badge&logo=filmstrip&logoColor=black)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-watchwise--ai.netlify.app-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://watchwise-ai.netlify.app)
[![Backend](https://img.shields.io/badge/API-watchwise--api.onrender.com-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://watchwise-api.onrender.com/docs)
[![GitHub](https://img.shields.io/badge/GitHub-ANIKET640--a11y-181717?style=for-the-badge&logo=github)](https://github.com/ANIKET640-a11y/WatchWise)

**WatchWise** is a full-stack AI-powered movie discovery platform that uses machine learning to deliver personalized movie recommendations in real time.

</div>

---

## 🚀 Live Demo

🌐 **Frontend:** [https://watchwise-ai.netlify.app](https://watchwise-ai.netlify.app)  
⚙️ **API Docs:** [https://watchwise-api.onrender.com/docs](https://watchwise-api.onrender.com/docs)

> ⚠️ The backend runs on Render's free tier and may take 30–60 seconds to wake up after inactivity.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 AI Recommendations | TF-IDF vectorization + cosine similarity engine |
| 🔥 Top 10 This Week | Real-time trending movies from TMDB |
| 🎭 Celebrity Profiles | Full filmography, biography & starmeter rankings |
| 📺 Streaming Availability | Netflix, Prime Video, Disney+, Apple TV+ |
| 🎬 Trailer Integration | YouTube trailers with modal player |
| 🎪 In Theaters & Coming Soon | Live cinema listings |
| 📰 Movie News Feed | Latest entertainment news via News API |
| 🔖 Personal Watchlist | Save & manage your movie list locally |
| 🏆 Awards Tracker | Oscar & film festival coverage |
| 🔍 Smart Search | Real-time movie search with dropdown results |
| 🎨 Genre Browsing | Discover movies by genre |

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** — High-performance Python REST API framework
- **Scikit-learn** — TF-IDF Vectorizer for ML recommendation engine
- **Pandas & NumPy** — Data processing and manipulation
- **TMDB API** — Real-time movie data, posters, trailers, cast
- **News API** — Live entertainment news feed
- **Uvicorn** — ASGI server

### Frontend
- **HTML5 / CSS3 / Vanilla JavaScript** — No frameworks, pure web
- **Responsive Design** — Mobile and desktop optimized

### ML Model
- **TF-IDF Vectorizer** — Converts movie metadata into feature vectors
- **Cosine Similarity** — Measures similarity between movies for recommendations
- **Pre-trained pickle files** — Fast inference at runtime (no retraining on each request)

### Deployment
- **Render** — Backend (FastAPI) cloud deployment
- **Netlify** — Frontend static site hosting with auto-deploy on push

---

## 📁 Project Structure

```
WatchWise/
├── main.py                 # FastAPI backend — all API routes
├── main_secondary.py       # Helper functions
├── watchwise.html          # Full frontend (single-page app)
├── index.html              # Netlify entry point
├── movies_metadata.csv     # Movie dataset
├── df.pkl                  # Preprocessed dataframe
├── tfidf.pkl               # Trained TF-IDF vectorizer
├── tfidf_matrix.pkl        # TF-IDF feature matrix
├── indices.pkl             # Movie title → index mapping
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11+
- TMDB API Key → [Get it here](https://www.themoviedb.org/settings/api)
- News API Key → [Get it here](https://newsapi.org/)

### 1. Clone the repository
```bash
git clone https://github.com/ANIKET640-a11y/WatchWise.git
cd WatchWise
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
TMDB_API_KEY=your_tmdb_api_key_here
NEWS_API_KEY=your_news_api_key_here
```

### 5. Run the backend
```bash
uvicorn main:app --reload
```

### 6. Open the frontend
Open `watchwise.html` in your browser or serve it locally:
```bash
python -m http.server 3000
```
Then visit `http://localhost:3000/watchwise.html`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/home` | Homepage data |
| GET | `/top10` | Top 10 trending movies |
| GET | `/featured` | Featured movies |
| GET | `/tmdb/search` | Search movies |
| GET | `/movie/id/{tmdb_id}` | Movie details |
| GET | `/recommend/tfidf` | AI recommendations |
| GET | `/recommend/genre` | Genre-based recommendations |
| GET | `/movie/{tmdb_id}/watch-providers` | Streaming availability |
| GET | `/people/trending` | Trending celebrities |
| GET | `/people/born-today` | Celebrities born today |
| GET | `/person/{person_id}` | Celebrity profile |

Full interactive API docs: [https://watchwise-api.onrender.com/docs](https://watchwise-api.onrender.com/docs)

---

## 🧠 How the Recommendation Engine Works

1. **Dataset** — Movie metadata (title, genres, keywords, cast, crew, overview) from `movies_metadata.csv`
2. **Preprocessing** — Text fields are combined and cleaned
3. **TF-IDF Vectorization** — Converts text into numerical feature vectors (captures word importance across documents)
4. **Cosine Similarity** — Measures the angle between two movie vectors; closer angle = more similar movies
5. **Inference** — Pre-trained matrices are loaded at startup via pickle for fast real-time recommendations

---

## 🚀 Deployment

### Backend (Render)
- Runtime: Python 3.11
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend (Netlify)
- Connected to GitHub for auto-deploy on every push to `main`
- No build step required (pure HTML/CSS/JS)

---

## 👨‍💻 Author

**Aniket Kumar Singh**  
B.Tech CSE (AI/ML) — VIT Bhopal University (2024–2028)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/your-linkedin)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/ANIKET640-a11y)
[![LeetCode](https://img.shields.io/badge/LeetCode-Anikett666-FFA116?style=flat&logo=leetcode)](https://leetcode.com/Anikett666)

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a ⭐ on GitHub — it means a lot!

---

<div align="center">
Built with ❤️ by Aniket Kumar Singh
</div>
