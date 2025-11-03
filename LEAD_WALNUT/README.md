# SEO Content Quality & Duplicate Detector

## Project Overview
A comprehensive machine learning pipeline that analyzes web content for SEO quality assessment and duplicate detection. The system processes pre-scraped HTML content (65 cybersecurity web pages), extracts NLP features using state-of-the-art models, detects near-duplicate content via cosine similarity, and classifies content quality using a Random Forest classifier. Includes a real-time URL analysis function for on-demand content evaluation.

## Dataset Information
- **File**: `data (1).csv`
- **Total Rows**: 264,717 rows (NOTE: We'll work with 60-70 rows as per assignment)
- **Columns**: 
  - `url`: Website URLs
  - `html_content`: Raw HTML content from the URLs (pre-scraped)
- **Domain**: Cybersecurity websites (e.g., cm-alliance.com)

## Assignment Requirements (100 points + 25 bonus)

### 1. Data Collection & HTML Parsing (15%)
- Read CSV dataset with URLs and HTML content
- Parse HTML to extract: page title, main body text (clean)
- Calculate word count
- Handle parsing errors gracefully
- Save extracted data to CSV (without html_content column)

### 2. Text Preprocessing & Feature Engineering (25%)
- Clean extracted text (lowercase, remove whitespace)
- Extract features:
  - Basic metrics: word count, sentence count
  - Readability: Flesch Reading Ease score
  - Keywords: Top 5 keywords using TF-IDF
  - Embeddings: Sentence transformers OR TF-IDF vectors
- Save features to CSV

### 3. Duplicate Detection (20%)
- Compute cosine similarity on embeddings/TF-IDF
- Define threshold (e.g., > 0.80 = duplicate)
- Identify duplicate pairs
- Detect thin content (word_count < 500)
- Save results to CSV

### 4. Content Quality Scoring (25%)
- Build quality classifier (Low/Medium/High)
- Labeling criteria:
  - **High**: word_count > 1500 AND 50 <= readability <= 70
  - **Low**: word_count < 500 OR readability < 30
  - **Medium**: all other cases
- Train model (Logistic Regression or Random Forest)
- 70/30 train/test split
- Report: Accuracy, F1-score, confusion matrix, top features

### 5. Real-Time Analysis Demo (15%)
- Create `analyze_url(url)` function in Jupyter notebook
- Accepts URL, scrapes, extracts features
- Returns quality score and duplicate matches

### Bonus Points (+25)
- Streamlit app deployed to Streamlit Cloud (+15)
- Advanced NLP analysis (+7)
- Visualizations (+3)

## Game Plan Summary

### Time Allocation (240 minutes)
1. **Setup** (15 min) - Environment, dependencies, folder structure
2. **Scraping & Parsing** (40 min) - Parse HTML, extract text, save CSV
3. **Feature Engineering** (60 min) - Extract metrics, readability, keywords, embeddings
4. **Duplicate Detection** (35 min) - Compute similarity, save results
5. **Quality Model** (70 min) - Create labels, train model, evaluate
6. **Real-Time Demo** (25 min) - Build analyze_url() function
7. **Documentation** (25 min) - Write README, clean up code
8. **Buffer** (30 min) - Debugging and polish

### Implementation Strategy
- **HTML Parsing:** BeautifulSoup4 with lxml parser
- **Features:** Textstat (readability), TF-IDF (keywords), Sentence-Transformers (embeddings)
- **Similarity:** Cosine similarity on embeddings (threshold > 0.80)
- **Model:** Random Forest Classifier with synthetic labels
- **Baseline:** Rule-based word count classifier

### Key Libraries
- pandas, numpy, scikit-learn
- beautifulsoup4, lxml, requests
- sentence-transformers, textstat, nltk
- matplotlib, seaborn (visualizations)

## Tech Stack
- Python 3.9+
- Jupyter Notebook (mandatory)
- Libraries: BeautifulSoup4, pandas, scikit-learn, sentence-transformers, textstat
- Optional: Streamlit (for bonus)

## Time Allocation (4 hours)
1. Setup - 15 min
2. Scraping & Parsing - 40 min
3. Features - 60 min
4. Duplicates - 35 min
5. Quality Model - 70 min
6. Real-time Demo - 25 min
7. Documentation - 25 min
8. Buffer - 30 min

## Expected Directory Structure
```
LEAD_WALNUT/
├── data/
│   ├── data.csv                      # Original dataset
│   ├── extracted_content.csv         # Parsed content
│   ├── features.csv                  # Extracted features
│   └── duplicates.csv                # Duplicate pairs
├── notebooks/
│   └── seo_pipeline.ipynb            # Main notebook (REQUIRED)
├── models/
│   └── quality_model.pkl             # Saved model
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/LEAD_WALNUT.git
cd LEAD_WALNUT

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook notebooks/seo_pipeline.ipynb
```

### Quick Start
1. Open `notebooks/seo_pipeline.ipynb`
2. Run all cells sequentially (Kernel → Restart & Run All)
3. Review generated CSV files in `data/` folder
4. Test the `analyze_url()` function with any URL

## Implementation Status
- [x] Requirements gathered and game plan finalized
- [x] Environment setup and dependencies installed
- [x] Dataset prepared (65 rows sampled)
- [x] HTML parsing implemented with BeautifulSoup
- [x] Feature engineering (readability, keywords, embeddings)
- [x] Duplicate detection with cosine similarity
- [x] Quality classification model trained (Random Forest)
- [x] Real-time URL analysis function created
- [x] Complete Jupyter notebook pipeline
- [ ] Documentation and results summary
- [ ] Optional: Streamlit deployment (+15 bonus points)

## Key Decisions & Rationale

1. **HTML Parsing with BeautifulSoup + lxml**: Chosen for its robustness in handling malformed HTML and excellent performance. The lxml parser is faster than html.parser while maintaining accuracy.

2. **Sentence Transformers (all-MiniLM-L6-v2)**: Selected for generating semantic embeddings due to its optimal balance of speed and quality. This lightweight model (80MB) produces 384-dimensional vectors perfect for similarity detection.

3. **Similarity Threshold (0.80)**: Set at 80% to capture near-duplicates while avoiding false positives. This threshold balances precision and recall based on content similarity distribution analysis.

4. **Random Forest Classifier**: Chosen over Logistic Regression for its ability to handle non-linear relationships between features and capture feature interactions without manual engineering.

5. **Synthetic Labeling Criteria**: Used clear, non-overlapping thresholds (High: >1500 words + readability 50-70; Low: <500 words OR readability <30) to create training labels that reflect SEO best practices.

## Results Summary

### Model Performance
- **Random Forest Accuracy**: 0.833
- **Baseline Accuracy**: 0.444
- **Improvement**: 87.5% over baseline
- **Top Features**: flesch_reading_ease (importance: 0.497), word_count (0.283), sentence_count (0.219)

### Content Analysis
- **Pages Analyzed**: 57 (successfully parsed from 65 total)
- **Duplicate Pairs Found**: 2 pairs above 0.80 similarity threshold
- **Thin Content**: 10 pages (<500 words), representing 17.5% of dataset
- **Quality Distribution**: High (7), Medium (25), Low (25)

## Limitations

1. **Dataset Size**: Limited to 65 pages due to assignment constraints. Larger datasets would improve model generalization and provide more robust duplicate detection.

2. **Domain Specificity**: Trained primarily on cybersecurity content. Model may require retraining for other domains (e-commerce, news, etc.) to maintain accuracy.

3. **Static Threshold**: Uses fixed similarity threshold (0.80). Dynamic thresholding based on content type or industry standards could improve precision.

4. **Feature Scope**: Limited to basic text features. Advanced features (topic modeling, sentiment analysis, entity recognition) could enhance quality predictions.

## Key Deliverables
1. ✅ `notebooks/seo_pipeline.ipynb` - Complete end-to-end pipeline
2. ✅ `data/extracted_content.csv` - Parsed HTML content (no raw HTML)
3. ✅ `data/features.csv` - Extracted features with embeddings
4. ✅ `data/duplicates.csv` - Duplicate pairs above threshold
5. ✅ `models/quality_model.pkl` - Trained Random Forest model
6. ✅ `analyze_url()` function - Real-time analysis capability
7. ✅ Comprehensive README with setup and documentation

---
**Created**: November 3, 2025  
**Last Updated**: November 3, 2025  
**Assignment**: Data Science - SEO Content Quality & Duplicate Detector  
**Time Budget**: 4 hours (240 minutes)
