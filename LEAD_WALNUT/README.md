# SEO Content Quality & Duplicate Detector# SEO Content Quality & Duplicate Detector# SEO Content Quality & Duplicate Detector



A machine learning-powered tool that analyzes web content for quality assessment, duplicate detection, and SEO insights.



## FeaturesA machine learning-powered tool that analyzes web content for quality assessment, duplicate detection, and SEO insights.## Project Overview



- **HTML Parsing**: Extracts clean text from HTML contentA comprehensive machine learning pipeline that analyzes web content for SEO quality assessment and duplicate detection. The system processes pre-scraped HTML content (65 cybersecurity web pages), extracts NLP features using state-of-the-art models, detects near-duplicate content via cosine similarity, and classifies content quality using a Random Forest classifier. Includes a real-time URL analysis function for on-demand content evaluation.

- **Feature Engineering**: Calculates readability scores, keywords, and semantic embeddings

- **Duplicate Detection**: Identifies similar content using cosine similarity## Features

- **Quality Classification**: ML model predicting content quality (High/Medium/Low)

- **Real-Time Analysis**: Analyze any URL on-demand## Dataset Information

- **Interactive Dashboard**: Streamlit web application for deployment

- **HTML Parsing**: Extracts clean text from HTML content- **File**: `data (1).csv`

## Project Structure

- **Feature Engineering**: Calculates readability scores, keywords, and semantic embeddings- **Total Rows**: 264,717 rows (NOTE: We'll work with 60-70 rows as per assignment)

```

LEAD_WALNUT/- **Duplicate Detection**: Identifies similar content using cosine similarity- **Columns**: 

├── app.py                      # Streamlit web application

├── notebooks/- **Quality Classification**: ML model predicting content quality (High/Medium/Low)  - `url`: Website URLs

│   └── seo_pipeline.ipynb      # Complete ML pipeline notebook

├── data/- **Real-Time Analysis**: Analyze any URL on-demand  - `html_content`: Raw HTML content from the URLs (pre-scraped)

│   ├── data.csv                # Original dataset (65 samples)

│   ├── extracted_content.csv   # Parsed HTML content- **Interactive Dashboard**: Streamlit web application for easy deployment- **Domain**: Cybersecurity websites (e.g., cm-alliance.com)

│   ├── features.csv            # Engineered features

│   └── duplicates.csv          # Detected duplicate pairs

├── models/

│   └── quality_model.pkl       # Trained Random Forest model## Project Structure## Assignment Requirements (100 points + 25 bonus)

├── requirements.txt            # Python dependencies

├── .gitignore                  # Git ignore rules

└── README.md                   # This file

``````### 1. Data Collection & HTML Parsing (15%)



## InstallationLEAD_WALNUT/- Read CSV dataset with URLs and HTML content



### 1. Clone the repository├── app.py                      # Streamlit web application- Parse HTML to extract: page title, main body text (clean)

```bash

git clone <your-repo-url>├── notebooks/- Calculate word count

cd LEAD_WALNUT

```│   └── seo_pipeline.ipynb      # Complete ML pipeline notebook- Handle parsing errors gracefully



### 2. Create virtual environment (recommended)├── data/- Save extracted data to CSV (without html_content column)

```bash

python -m venv venv│   ├── data.csv                # Original dataset (65 samples)

source venv/bin/activate  # On Windows: venv\Scripts\activate

```│   ├── extracted_content.csv   # Parsed HTML content### 2. Text Preprocessing & Feature Engineering (25%)



### 3. Install dependencies│   ├── features.csv            # Engineered features- Clean extracted text (lowercase, remove whitespace)

```bash

pip install -r requirements.txt│   └── duplicates.csv          # Detected duplicate pairs- Extract features:

```

├── models/  - Basic metrics: word count, sentence count

### 4. Download NLTK data

```python│   └── quality_model.pkl       # Trained Random Forest model  - Readability: Flesch Reading Ease score

import nltk

nltk.download('punkt')├── requirements.txt            # Python dependencies  - Keywords: Top 5 keywords using TF-IDF

nltk.download('punkt_tab')

```├── .gitignore                  # Git ignore rules  - Embeddings: Sentence transformers OR TF-IDF vectors



## Usage└── README.md                   # This file- Save features to CSV



### Run Jupyter Notebook```



```bash### 3. Duplicate Detection (20%)

jupyter notebook notebooks/seo_pipeline.ipynb

```## Installation- Compute cosine similarity on embeddings/TF-IDF



**Execute cells in order to:**- Define threshold (e.g., > 0.80 = duplicate)

- Parse HTML content

- Extract features (word count, readability, keywords, embeddings)1. **Clone the repository**- Identify duplicate pairs

- Detect duplicate content

- Train quality classification model```bash- Detect thin content (word_count < 500)

- Test real-time URL analysis

git clone <your-repo-url>- Save results to CSV

### Launch Streamlit App

cd LEAD_WALNUT

```bash

streamlit run app.py```### 4. Content Quality Scoring (25%)

```

- Build quality classifier (Low/Medium/High)

**The web application provides:**

- **Analyze URL**: Enter any URL for instant quality analysis2. **Create virtual environment** (recommended)- Labeling criteria:

- **Dataset Overview**: Visualize statistics and distributions

- **Duplicates**: View detected duplicate content pairs```bash  - **High**: word_count > 1500 AND 50 <= readability <= 70



Open browser to: `http://localhost:8501`python -m venv venv  - **Low**: word_count < 500 OR readability < 30



## Model Performancesource venv/bin/activate  # On Windows: venv\Scripts\activate  - **Medium**: all other cases



| Metric | Value |```- Train model (Logistic Regression or Random Forest)

|--------|-------|

| **Random Forest Accuracy** | 83.3% |- 70/30 train/test split

| **Baseline Accuracy** | 44.4% |

| **Improvement** | 87.5% |3. **Install dependencies**- Report: Accuracy, F1-score, confusion matrix, top features



### Feature Importance```bash

1. Flesch Reading Ease: 49.7%

2. Word Count: 28.3%pip install -r requirements.txt### 5. Real-Time Analysis Demo (15%)

3. Sentence Count: 21.9%

```- Create `analyze_url(url)` function in Jupyter notebook

## Results Summary

- Accepts URL, scrapes, extracts features

- **Pages Analyzed**: 57 pages

- **Duplicate Pairs**: 2 pairs (similarity > 80%)4. **Download NLTK data** (if not auto-downloaded)- Returns quality score and duplicate matches

- **Thin Content**: 10 pages (17.5%)

- **Quality Distribution**:```python

  - High: 7 pages (12.3%)

  - Medium: 25 pages (43.9%)import nltk### Bonus Points (+25)

  - Low: 25 pages (43.9%)

nltk.download('punkt')- Streamlit app deployed to Streamlit Cloud (+15)

## Technical Details

nltk.download('punkt_tab')- Advanced NLP analysis (+7)

### Model Architecture

- **Algorithm**: Random Forest Classifier```- Visualizations (+3)

- **Estimators**: 100

- **Max Depth**: 10

- **Features**: word_count, sentence_count, flesch_reading_ease

## Usage## Game Plan Summary

### Embeddings

- **Model**: Sentence Transformers ('all-MiniLM-L6-v2')

- **Dimension**: 384

- **Similarity Metric**: Cosine Similarity### 1. Run Jupyter Notebook### Time Allocation (240 minutes)

- **Duplicate Threshold**: 0.80

1. **Setup** (15 min) - Environment, dependencies, folder structure

### Quality Labels

- **High**: word_count > 1500 AND readability 50-70```bash2. **Scraping & Parsing** (40 min) - Parse HTML, extract text, save CSV

- **Low**: word_count < 500 OR readability < 30

- **Medium**: All other casesjupyter notebook notebooks/seo_pipeline.ipynb3. **Feature Engineering** (60 min) - Extract metrics, readability, keywords, embeddings



## Key Decisions & Rationale```4. **Duplicate Detection** (35 min) - Compute similarity, save results



1. **Random Forest over Deep Learning**: Better interpretability, faster training, sufficient for tabular data5. **Quality Model** (70 min) - Create labels, train model, evaluate

2. **Sentence Transformers**: State-of-the-art semantic embeddings without fine-tuning

3. **Flesch Reading Ease**: Industry-standard readability metricRun all cells to:6. **Real-Time Demo** (25 min) - Build analyze_url() function

4. **0.80 Similarity Threshold**: Balances precision/recall for duplicate detection

5. **Stratified Split**: Maintains class distribution in train/test sets- Parse HTML content7. **Documentation** (25 min) - Write README, clean up code



## Deployment- Extract features8. **Buffer** (30 min) - Debugging and polish



### Local Deployment- Detect duplicates

```bash

streamlit run app.py- Train quality classification model### Implementation Strategy

```

- Test real-time URL analysis- **HTML Parsing:** BeautifulSoup4 with lxml parser

### Cloud Deployment (Streamlit Cloud)

1. Push to GitHub- **Features:** Textstat (readability), TF-IDF (keywords), Sentence-Transformers (embeddings)

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect repository and deploy app.py### 2. Launch Streamlit App- **Similarity:** Cosine similarity on embeddings (threshold > 0.80)



### Docker Deployment- **Model:** Random Forest Classifier with synthetic labels

```dockerfile

FROM python:3.9-slim```bash- **Baseline:** Rule-based word count classifier

WORKDIR /app

COPY requirements.txt .streamlit run app.py

RUN pip install -r requirements.txt

COPY . .```### Key Libraries

CMD ["streamlit", "run", "app.py"]

```- pandas, numpy, scikit-learn



## DependenciesThe web application provides:- beautifulsoup4, lxml, requests



- Python 3.9+- **Analyze URL**: Enter any URL for instant quality analysis- sentence-transformers, textstat, nltk

- pandas, numpy, scikit-learn

- BeautifulSoup4, lxml, requests- **Dataset Overview**: Visualize statistics and distributions- matplotlib, seaborn (visualizations)

- sentence-transformers, textstat, nltk

- streamlit, plotly- **Duplicates**: View detected duplicate content pairs

- See `requirements.txt` for complete list

## Tech Stack

## Limitations

## Model Performance- Python 3.9+

1. **Sample Size**: 65 pages (sampled from 264K for development)

2. **Domain-Specific**: Trained on cybersecurity content- Jupyter Notebook (mandatory)

3. **Synthetic Labels**: Quality labels created via heuristics

4. **Static Embeddings**: No fine-tuning for domain adaptation| Metric | Value |- Libraries: BeautifulSoup4, pandas, scikit-learn, sentence-transformers, textstat



## Future Enhancements|--------|-------|- Optional: Streamlit (for bonus)



- Expand dataset to full 264K pages| **Random Forest Accuracy** | 83.3% |

- Fine-tune embeddings on domain data

- Add multi-label classification (SEO factors)| **Baseline Accuracy** | 44.4% |## Time Allocation (4 hours)

- Implement content recommendations

- Add batch URL processing| **Improvement** | 87.5% |1. Setup - 15 min

- Integration with SEO tools

2. Scraping & Parsing - 40 min

## Author

### Feature Importance3. Features - 60 min

**LEAD_WALNUT Project**  

Date: November 3, 20251. Flesch Reading Ease: 49.7%4. Duplicates - 35 min



## License2. Word Count: 28.3%5. Quality Model - 70 min



This project is created for educational purposes as part of a placement assignment.3. Sentence Count: 21.9%6. Real-time Demo - 25 min



---7. Documentation - 25 min



**Note**: This is a placement assignment project demonstrating ML engineering skills including data processing, feature engineering, model training, evaluation, and deployment.## Results Summary8. Buffer - 30 min




- **Pages Analyzed**: 57 pages## Expected Directory Structure

- **Duplicate Pairs**: 2 pairs (similarity > 80%)```

- **Thin Content**: 10 pages (17.5%)LEAD_WALNUT/

- **Quality Distribution**:├── data/

  - High: 7 pages│   ├── data.csv                      # Original dataset

  - Medium: 25 pages│   ├── extracted_content.csv         # Parsed content

  - Low: 25 pages│   ├── features.csv                  # Extracted features

│   └── duplicates.csv                # Duplicate pairs

## Technical Details├── notebooks/

│   └── seo_pipeline.ipynb            # Main notebook (REQUIRED)

### Model Architecture├── models/

- **Algorithm**: Random Forest Classifier│   └── quality_model.pkl             # Saved model

- **Estimators**: 100├── requirements.txt

- **Max Depth**: 10├── .gitignore

- **Features**: word_count, sentence_count, flesch_reading_ease└── README.md

```

### Embeddings

- **Model**: Sentence Transformers ('all-MiniLM-L6-v2')## Setup Instructions

- **Dimension**: 384

- **Similarity Metric**: Cosine Similarity### Prerequisites

- **Duplicate Threshold**: 0.80- Python 3.9+

- pip package manager

### Quality Labels

- **High**: word_count > 1500 AND readability 50-70### Installation

- **Low**: word_count < 500 OR readability < 30```bash

- **Medium**: All other cases# Clone the repository

git clone https://github.com/yourusername/LEAD_WALNUT.git

## Key Decisions & Rationalecd LEAD_WALNUT



1. **Random Forest over Deep Learning**: Better interpretability, faster training, sufficient for tabular data# Install dependencies

2. **Sentence Transformers**: State-of-the-art semantic embeddings without fine-tuningpip install -r requirements.txt

3. **Flesch Reading Ease**: Industry-standard readability metric

4. **0.80 Similarity Threshold**: Balances precision/recall for duplicate detection# Run the Jupyter notebook

5. **Stratified Split**: Maintains class distribution in train/test setsjupyter notebook notebooks/seo_pipeline.ipynb

```

## Deployment Options

### Quick Start

### Local Deployment1. Open `notebooks/seo_pipeline.ipynb`

```bash2. Run all cells sequentially (Kernel → Restart & Run All)

streamlit run app.py3. Review generated CSV files in `data/` folder

```4. Test the `analyze_url()` function with any URL



### Cloud Deployment (Streamlit Cloud)## Implementation Status

1. Push to GitHub- [x] Requirements gathered and game plan finalized

2. Go to [share.streamlit.io](https://share.streamlit.io)- [x] Environment setup and dependencies installed

3. Connect repository- [x] Dataset prepared (65 rows sampled)

4. Deploy app.py- [x] HTML parsing implemented with BeautifulSoup

- [x] Feature engineering (readability, keywords, embeddings)

### Docker Deployment- [x] Duplicate detection with cosine similarity

```dockerfile- [x] Quality classification model trained (Random Forest)

FROM python:3.9-slim- [x] Real-time URL analysis function created

WORKDIR /app- [x] Complete Jupyter notebook pipeline

COPY requirements.txt .- [ ] Documentation and results summary

RUN pip install -r requirements.txt- [ ] Optional: Streamlit deployment (+15 bonus points)

COPY . .

CMD ["streamlit", "run", "app.py"]## Key Decisions & Rationale

```

1. **HTML Parsing with BeautifulSoup + lxml**: Chosen for its robustness in handling malformed HTML and excellent performance. The lxml parser is faster than html.parser while maintaining accuracy.

## Dependencies

2. **Sentence Transformers (all-MiniLM-L6-v2)**: Selected for generating semantic embeddings due to its optimal balance of speed and quality. This lightweight model (80MB) produces 384-dimensional vectors perfect for similarity detection.

- Python 3.9+

- pandas, numpy, scikit-learn3. **Similarity Threshold (0.80)**: Set at 80% to capture near-duplicates while avoiding false positives. This threshold balances precision and recall based on content similarity distribution analysis.

- BeautifulSoup4, lxml, requests

- sentence-transformers, textstat, nltk4. **Random Forest Classifier**: Chosen over Logistic Regression for its ability to handle non-linear relationships between features and capture feature interactions without manual engineering.

- streamlit, plotly

- See `requirements.txt` for complete list5. **Synthetic Labeling Criteria**: Used clear, non-overlapping thresholds (High: >1500 words + readability 50-70; Low: <500 words OR readability <30) to create training labels that reflect SEO best practices.



## Limitations## Results Summary



1. **Sample Size**: 65 pages (sampled from 264K for development)### Model Performance

2. **Domain-Specific**: Trained on cybersecurity content- **Random Forest Accuracy**: 0.833

3. **Synthetic Labels**: Quality labels created via heuristics- **Baseline Accuracy**: 0.444

4. **Static Embeddings**: No fine-tuning for domain adaptation- **Improvement**: 87.5% over baseline

5. **Scraping Constraints**: Respects robots.txt, handles timeouts- **Top Features**: flesch_reading_ease (importance: 0.497), word_count (0.283), sentence_count (0.219)



## Future Enhancements### Content Analysis

- **Pages Analyzed**: 57 (successfully parsed from 65 total)

- Expand dataset to full 264K pages- **Duplicate Pairs Found**: 2 pairs above 0.80 similarity threshold

- Fine-tune embeddings on domain data- **Thin Content**: 10 pages (<500 words), representing 17.5% of dataset

- Add multi-label classification (SEO factors)- **Quality Distribution**: High (7), Medium (25), Low (25)

- Implement content recommendations

- Add batch URL processing## Limitations

- Integration with SEO tools (Google Analytics, Search Console)

1. **Dataset Size**: Limited to 65 pages due to assignment constraints. Larger datasets would improve model generalization and provide more robust duplicate detection.

## License

2. **Domain Specificity**: Trained primarily on cybersecurity content. Model may require retraining for other domains (e-commerce, news, etc.) to maintain accuracy.

This project is created for educational purposes as part of a placement assignment.

3. **Static Threshold**: Uses fixed similarity threshold (0.80). Dynamic thresholding based on content type or industry standards could improve precision.

## Author

4. **Feature Scope**: Limited to basic text features. Advanced features (topic modeling, sentiment analysis, entity recognition) could enhance quality predictions.

**LEAD_WALNUT Project**  

Date: November 3, 2025## Key Deliverables

1. ✅ `notebooks/seo_pipeline.ipynb` - Complete end-to-end pipeline

## Contact2. ✅ `data/extracted_content.csv` - Parsed HTML content (no raw HTML)

3. ✅ `data/features.csv` - Extracted features with embeddings

For questions or feedback, please open an issue in the repository.4. ✅ `data/duplicates.csv` - Duplicate pairs above threshold

5. ✅ `models/quality_model.pkl` - Trained Random Forest model

---6. ✅ `analyze_url()` function - Real-time analysis capability

7. ✅ Comprehensive README with setup and documentation

**Note**: This is a placement assignment project demonstrating ML engineering skills including data processing, feature engineering, model training, evaluation, and deployment.

---
**Created**: November 3, 2025  
**Last Updated**: November 3, 2025  
**Assignment**: Data Science - SEO Content Quality & Duplicate Detector  
**Time Budget**: 4 hours (240 minutes)
