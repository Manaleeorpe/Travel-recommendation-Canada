# Travel Recommendation Canada

A Flask-based hotel recommendation web application that helps users explore popular hotels and get personalized hotel suggestions based on similarity scoring.

## Features

- **Popular hotels dashboard** on the home page with:
  - Hotel name
  - Average rating
  - Hotel experience text
  - Address
  - Country
- **Hotel recommendation search** page where users can enter a hotel name and receive similar hotel recommendations.
- **Content-based recommendation logic** that uses precomputed similarity scores between hotels.
- **Amenity-aware vectorization workflow** using TF-IDF and cosine similarity for hotel feature representation.
- **Simple browser UI** with Bootstrap styling and Jinja2 templates.

## Tech Stack

### Backend
- **Python**
- **Flask** (routing, request handling, template rendering)
- **NumPy** (indexing and similarity handling)
- **Pandas** (hotel dataset manipulation)
- **scikit-learn** (`TfidfVectorizer`, `cosine_similarity`)
- **Pickle** (loading preprocessed model/data artifacts)

### Frontend
- **HTML + Jinja2 templates**
- **Bootstrap 5**
- **Bootstrap Icons**
- **CSS**
- **JavaScript / jQuery**

### Data / Model Artifacts
- `popular.pkl`
- `pt.pkl`
- `hotels.pkl`
- `similarity_scores.pkl`

## Project Structure

```text
Travel-recommendation-Canada/
├── README.md
├── popular.pkl
├── pt.pkl
├── hotels.pkl
├── similarity_scores.pkl
└── env/
    ├── app.py
    ├── templates/
    │   ├── index.html
    │   └── recommend.html
    └── static/
        ├── css/
        ├── js/
        └── img/
```

## Run Locally

1. Navigate to the project root.
2. Install required Python packages (if not already available):
   - Flask
   - NumPy
   - Pandas
   - scikit-learn
3. Start the Flask app:

```bash
python env/app.py
```

4. Open your browser at:

```text
http://127.0.0.1:5000
```

## Notes

- The recommendation flow expects hotel names that exist in the pivot/index dataset (`pt.pkl`).
- Core recommendation assets are loaded from pickle files at startup.
