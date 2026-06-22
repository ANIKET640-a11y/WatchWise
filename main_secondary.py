import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/original"

if not TMDB_API_KEY:
    # Don't crash import-time in production if you prefer; but for you better fail early:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Watch Wise API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []
    trailer_url: Optional[str] = None
    reviews: List[dict] = []
    cast: List[dict] = []
    director: Optional[str] = None
    writers: List[str] = []
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    runtime: Optional[int] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB GET:
    - Network errors -> 502
    - TMDB API errors -> 502 with detail
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()

async def news_get():
    
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": "(movie OR actor OR actress OR film OR Hollywood OR Bollywood)",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 8,
        "apiKey": NEWS_API_KEY,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"News API error {r.status_code}"
        )

    return r.json()

async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    trailer_url = await get_movie_trailer(movie_id)
    reviews = await get_movie_reviews(movie_id)
    cast = await get_movie_cast(movie_id)
    director, writers = await get_movie_crew(movie_id)

    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
        trailer_url=trailer_url,
        reviews=reviews,
        cast=cast,
        director=director,
        writers=writers,
        vote_average=data.get('vote_average'),
        vote_count=data.get('vote_count'),
        runtime=data.get('runtime'),
    )



async def get_movie_trailer(movie_id: int):
    try:
        data = await tmdb_get(f"/movie/{movie_id}/videos", {"language": "en-US"})
        for video in data.get("results", []):
            if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                return f"https://www.youtube.com/watch?v={video['key']}"
    except Exception:
        pass
    return None

async def get_movie_reviews(movie_id: int, limit: int = 5):
    try:
        data = await tmdb_get(
            f"/movie/{movie_id}/reviews",
            {"language": "en-US", "page": 1},
        )
        reviews = []
        for r in data.get("results", [])[:limit]:
            reviews.append({
                "author": r.get("author"),
                "content": (r.get("content") or "")[:500],
                "rating": (r.get("author_details") or {}).get("rating")
            })
        return reviews
    except Exception:
        return []


async def get_movie_cast(movie_id: int, limit: int = 10):
    try:
        data = await tmdb_get(
            f"/movie/{movie_id}/credits",
            {"language": "en-US"}
        )
        cast = []
        for person in data.get("cast", [])[:limit]:
            cast.append({
                "name": person.get("name"),
                "character": person.get("character"),
                "profile_url": make_img_url(person.get("profile_path"))
            })
        return cast
    except Exception:
        return []


async def get_movie_crew(movie_id: int):
    try:
        data = await tmdb_get(
            f"/movie/{movie_id}/credits",
            {"language": "en-US"}
        )

        director = None
        writers = []

        for person in data.get("crew", []):
            if person.get("job") == "Director":
                director = person.get("name")

            if person.get("job") in ["Writer", "Screenplay", "Story"]:
                name = person.get("name")
                if name and name not in writers:
                    writers.append(name)

        return director, writers
    except Exception:
        return None, []
        

async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    # query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # sort descending
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix (usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load tfidf vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


# =========================
# ROUTES
# =========================

@app.get("/top10", response_model=List[TMDBMovieCard])
async def top10():
    data = await tmdb_get("/trending/movie/week", {"language":"en-US"})
    return await tmdb_cards_from_results(data.get("results", []), limit=10)

# Departments that indicate mainstream entertainment professionals
MAINSTREAM_DEPARTMENTS = {
    "Acting", "Directing", "Writing", "Production",
    "Camera", "Sound", "Editing", "Visual Effects",
    "Crew", "Art", "Costume & Make-Up"
}

# Hardcoded name fragments for known adult content performers
ADULT_NAME_FRAGMENTS = {
    "brazzers", "bangbros", "pornhub", "onlyfans", "bangbus",
    "mia khalifa", "riley reid", "lana rhoades", "abella danger",
    "alexis texas", "lisa ann", "jenna jameson", "stormy daniels"
}

def _is_adult_person(p: dict) -> bool:
    """
    Return True if this person should be filtered out as adult content.
    Uses TMDB's own flags as the primary signal — avoids false positives.
    """
    # 1. TMDB's own adult flag on the person object
    if p.get("adult") is True:
        return True

    # 2. Any known_for item explicitly flagged adult by TMDB
    known = p.get("known_for") or []
    if any(item.get("adult") is True for item in known):
        return True

    # 3. Hardcoded name blocklist as safety net
    name_lower = (p.get("name") or "").lower()
    if any(kw in name_lower for kw in ADULT_NAME_FRAGMENTS):
        return True

    return False


@app.get("/people/trending")
async def trending_people():
    people = []
    seen_ids = set()

    for page in range(1, 3):
        data = await tmdb_get(
            "/trending/person/week",
            {"language": "en-US", "include_adult": "false", "page": page}
        )
        for p in data.get("results", []):
            if p.get("id") in seen_ids:
                continue
            seen_ids.add(p.get("id"))

            # Must have a profile photo
            if not p.get("profile_path"):
                continue

            # Must be from a mainstream department
            dept = p.get("known_for_department", "")
            if dept not in MAINSTREAM_DEPARTMENTS:
                continue

            # Run adult content checks
            if _is_adult_person(p):
                continue

            known = p.get("known_for") or []
            people.append({
                "tmdb_id":    p.get("id"),
                "name":       p.get("name"),
                "profile_url": make_img_url(p.get("profile_path")),
                "known_for":  [k.get("title") or k.get("name") or "" for k in known[:2]],
                "department": dept,
                "popularity": p.get("popularity", 0)
            })

            if len(people) >= 20:
                break

        if len(people) >= 20:
            break

    return people


@app.get("/people/born-today")
async def born_today():
    """
    Returns real celebrities born on today's month and day.

    Strategy:
    - TMDB's /discover/person supports 'birthday_gte' and 'birthday_lte'
      but only matches the full date (YYYY-MM-DD), not just month/day.
    - So we query multiple birth years (last 100 years) in parallel,
      using the exact date for each year, then collect matching people.
    - Each call returns quickly; we fan out ~20 year-specific calls at once.
    """
    import asyncio
    from datetime import date

    today = date.today()
    month_str = str(today.month).zfill(2)
    day_str   = str(today.day).zfill(2)

    seen_ids: set = set()
    people = []

    # Build list of years to check (current year back to 1920)
    current_year = today.year
    years = list(range(current_year - 18, current_year - 100, -1))  # 18-100 years old

    async def fetch_for_year(year: int):
        date_str = f"{year}-{month_str}-{day_str}"
        try:
            data = await tmdb_get(
                "/discover/person",
                {
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "include_adult": "false",
                    "birthday_gte": date_str,
                    "birthday_lte": date_str,
                    "page": "1",
                }
            )
            return year, data.get("results", [])
        except Exception:
            return year, []

    # Run all year queries in parallel batches of 20
    batch_size = 20
    for i in range(0, len(years), batch_size):
        batch = years[i:i + batch_size]
        results = await asyncio.gather(*[fetch_for_year(y) for y in batch])

        for year, persons in results:
            for p in persons:
                pid = p.get("id")
                if not pid or pid in seen_ids:
                    continue
                if not p.get("profile_path"):
                    continue
                if _is_adult_person(p):
                    continue

                seen_ids.add(pid)
                age = current_year - year

                people.append({
                    "tmdb_id": pid,
                    "name": p.get("name"),
                    "profile_url": make_img_url(p.get("profile_path")),
                    "popularity": p.get("popularity", 0),
                    "birthday": f"{year}-{month_str}-{day_str}",
                    "age": age
                })

        # Stop early once we have enough
        if len(people) >= 20:
            break

    # Sort by popularity and return top 20
    people.sort(key=lambda x: x["popularity"], reverse=True)
    return people[:20]

@app.get("/person/{person_id}")
async def get_person(person_id: int):
    data = await tmdb_get(
        f"/person/{person_id}",
        {
            "language": "en-US",
            "append_to_response": "movie_credits,tv_credits,known_for,external_ids"
        }
    )
    return data

@app.get("/featured")
async def featured():

    data = await news_get()

    articles = []

    for article in data.get("articles", []):

        if not article.get("urlToImage"):
            continue

        articles.append(
            {
                "title": article.get("title"),
                "image": article.get("urlToImage"),
                "url": article.get("url"),
                "source": (
                    article.get("source") or {}
                ).get("name")
            }
        )

        if len(articles) == 6:
            break

    return articles

    data = await tmdb_get(
        "/movie/upcoming",
        {
            "language": "en-US",
            "page": 1
        }
    )

    featured_movies = []

    for movie in data.get("results", [])[:6]:

        featured_movies.append(
            {
                "tmdb_id": movie.get("id"),
                "title": movie.get("title"),
                "poster_url": make_img_url(
                    movie.get("poster_path")
                ),
                "overview": movie.get("overview")
            }
        )

    return featured_movies


@app.get("/person/{person_id}/images")
async def get_person_images(person_id: int):
    """Returns profile photos for a celebrity from TMDB."""
    data = await tmdb_get(
        f"/person/{person_id}/images",
        {}
    )
    return data


@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- WATCH PROVIDERS (exact deep-link per platform) ----------
PROVIDER_ID_MAP = {
    "netflix":   8,
    "prime":     9,
    "disney":    337,
    "apple":     350,
    "hbo":       1899,
    "hulu":      15,
    "paramount": 531,
}

@app.get("/movie/{tmdb_id}/watch-link")
async def get_watch_link(tmdb_id: int, platform: str = Query(...)):
    """
    Returns the exact streaming platform deep-link ONLY if the movie is
    confirmed available on that platform via TMDB. Returns 404 if not found.
    """
    provider_id = PROVIDER_ID_MAP.get(platform)
    if not provider_id:
        raise HTTPException(status_code=400, detail=f"Unknown platform: {platform}")

    try:
        data = await tmdb_get(f"/movie/{tmdb_id}/watch/providers", {"language": "en-US"})
        results = data.get("results", {})

        for region in ["IN", "US"] + list(results.keys()):
            region_data = results.get(region, {})
            link = region_data.get("link")
            if not link:
                continue
            all_providers = (
                region_data.get("flatrate", []) +
                region_data.get("buy", []) +
                region_data.get("rent", []) +
                region_data.get("free", [])
            )
            provider_ids = [p.get("provider_id") for p in all_providers]
            if provider_id in provider_ids:
                return {"url": link, "source": "tmdb_direct", "available": True}

        # NOT available on this platform — return explicit not-available
        raise HTTPException(status_code=404, detail="Not available on this platform")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"watch-link failed: {e}")


@app.get("/movie/{tmdb_id}/watch-providers")
async def get_watch_providers(tmdb_id: int):
    """
    Returns ALL confirmed streaming platforms for a movie in one call.
    Only includes platforms where the movie is actually available.
    Prioritises IN region, falls back to US, then any region.
    """
    try:
        data = await tmdb_get(f"/movie/{tmdb_id}/watch/providers", {"language": "en-US"})
        results = data.get("results", {})

        # Reverse map: provider_id -> our platform key
        id_to_platform = {v: k for k, v in PROVIDER_ID_MAP.items()}

        confirmed = {}  # platform_key -> {url, type}

        for region in ["IN", "US"] + list(results.keys()):
            region_data = results.get(region, {})
            link = region_data.get("link")
            if not link:
                continue

            type_map = {
                "flatrate": "stream",
                "free":     "free",
                "buy":      "buy",
                "rent":     "rent",
            }
            for access_type, label in type_map.items():
                for provider in region_data.get(access_type, []):
                    pid = provider.get("provider_id")
                    key = id_to_platform.get(pid)
                    if key and key not in confirmed:
                        confirmed[key] = {"url": link, "type": label, "region": region}

            # Stop as soon as we have at least one result from a preferred region
            if confirmed and region in ("IN", "US"):
                break

        return {"available": list(confirmed.keys()), "details": confirmed}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"watch-providers failed: {e}")



# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
    page: int = Query(1, ge=1, le=500),
):
    """
    Home feed for Streamlit (posters).
    category:
      - trending (trending/movie/day)
      - popular, top_rated, upcoming, now_playing  (movie/{category})
    page: TMDB page number for pagination
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US", "page": page})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": page})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns RAW TMDB shape with 'results' list.
    Streamlit will use it for:
      - dropdown suggestions
      - grid results
    """
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Given a TMDB movie ID:
    - fetch details
    - pick first genre
    - discover movies in that genre (popular)
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (TMDB) + posters

    NOTE:
    - It selects the BEST match from TMDB for the given query.
    - If you want MULTIPLE matches, use /tmdb/search
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by TMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )