# Complete SEO Pipeline Implementation
# This file contains all the code that will be added to the Jupyter notebook

# Core Libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# HTML Parsing
from bs4 import BeautifulSoup
import requests
from time import sleep

# NLP & Text Processing
import textstat
from sentence_transformers import SentenceTransformer
import nltk

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import pickle
import json
import os
from pathlib import Path

print("✅ All libraries imported successfully!")

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# SECTION: Load Dataset
df = pd.read_csv('data/data.csv')
print(f"✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few URLs:")
for i, url in enumerate(df['url'].head(3)):
    print(f"  {i+1}. {url[:80]}...")

# SECTION: HTML Parsing Function
def parse_html(html_content):
    """
    Parse HTML content and extract title, body text, and word count.
    
    Args:
        html_content (str): Raw HTML content
        
    Returns:
        dict: Dictionary with title, body_text, and word_count
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title"
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            tag.decompose()
        
        # Extract body text
        body = soup.find('body')
        if body:
            text = body.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean text - remove extra whitespace
        text = ' '.join(text.split())
        
        # Word count
        word_count = len(text.split())
        
        return {
            'title': title_text,
            'body_text': text,
            'word_count': word_count
        }
    except Exception as e:
        print(f"⚠️ Parsing error: {e}")
        return None

# SECTION: Parse all HTML content
print("\\n" + "="*60)
print("PARSING HTML CONTENT")
print("="*60)

extracted_data = []
failed_count = 0

for idx, row in df.iterrows():
    parsed = parse_html(row['html_content'])
    if parsed:
        extracted_data.append({
            'url': row['url'],
            'title': parsed['title'],
            'body_text': parsed['body_text'],
            'word_count': parsed['word_count']
        })
    else:
        failed_count += 1
        
print(f"\\n✅ Successfully parsed: {len(extracted_data)} pages")
print(f"⚠️ Failed to parse: {failed_count} pages")

# Create DataFrame
extracted_df = pd.DataFrame(extracted_data)

# Save extracted content (without html_content for smaller file size)
extracted_df.to_csv('data/extracted_content.csv', index=False)
print(f"\\n💾 Saved to: data/extracted_content.csv")

# Display sample
print(f"\\nSample extracted content:")
print(extracted_df[['url', 'title', 'word_count']].head())

# SECTION: Text Preprocessing and Feature Engineering
print("\\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

def extract_sentence_count(text):
    """Extract sentence count from text."""
    try:
        sentences = nltk.sent_tokenize(text)
        return len(sentences)
    except:
        return len([s for s in text.split('.') if s.strip()])

def extract_readability(text):
    """Calculate Flesch Reading Ease score."""
    try:
        return textstat.flesch_reading_ease(text)
    except:
        return 50.0  # Default moderate readability

# Add basic features
print("\\n📊 Extracting basic features...")
extracted_df['sentence_count'] = extracted_df['body_text'].apply(extract_sentence_count)
extracted_df['flesch_reading_ease'] = extracted_df['body_text'].apply(extract_readability)

print(f"✅ Sentence count and readability scores calculated!")

# SECTION: Keyword Extraction using TF-IDF
print("\\n🔑 Extracting keywords using TF-IDF...")

def extract_top_keywords(texts, n_keywords=5):
    """Extract top N keywords for each document using TF-IDF."""
    try:
        vectorizer = TfidfVectorizer(
            max_features=100, 
            stop_words='english',
            max_df=0.85,
            min_df=2
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        keywords_list = []
        for doc_idx in range(tfidf_matrix.shape[0]):
            tfidf_scores = tfidf_matrix[doc_idx].toarray()[0]
            top_indices = tfidf_scores.argsort()[-n_keywords:][::-1]
            top_keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            if len(top_keywords) == 0:
                top_keywords = ['no_keywords']
            keywords_list.append('|'.join(top_keywords))
        
        return keywords_list
    except Exception as e:
        print(f"⚠️ Keyword extraction error: {e}")
        return ['no_keywords'] * len(texts)

extracted_df['top_keywords'] = extract_top_keywords(extracted_df['body_text'].tolist())
print(f"✅ Keywords extracted!")

# SECTION: Generate Embeddings
print("\\n🧠 Generating text embeddings (this may take a minute)...")

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(
        extracted_df['body_text'].tolist(), 
        show_progress_bar=True,
        batch_size=16
    )
    print(f"✅ Embeddings generated! Shape: {embeddings.shape}")
    
    # Store embeddings
    extracted_df['embedding'] = list(embeddings)
    
except Exception as e:
    print(f"⚠️ Error generating embeddings: {e}")
    print("Using fallback: TF-IDF vectors")
    
    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_embeddings = vectorizer.fit_transform(extracted_df['body_text'])
    embeddings = tfidf_embeddings.toarray()
    extracted_df['embedding'] = list(embeddings)
    print(f"✅ TF-IDF embeddings generated! Shape: {embeddings.shape}")

# SECTION: Save Features
features_df = extracted_df[['url', 'word_count', 'sentence_count', 'flesch_reading_ease', 'top_keywords', 'embedding']].copy()

# Convert embeddings to string for CSV storage
features_df['embedding_vector'] = features_df['embedding'].apply(lambda x: str(x.tolist() if hasattr(x, 'tolist') else x))
features_df_save = features_df.drop('embedding', axis=1)

features_df_save.to_csv('data/features.csv', index=False)
print(f"\\n💾 Saved features to: data/features.csv")
print(f"\\nFeature summary:")
print(features_df[['word_count', 'sentence_count', 'flesch_reading_ease']].describe())

# SECTION: Duplicate Detection
print("\\n" + "="*60)
print("DUPLICATE DETECTION")
print("="*60)

print("\\n🔍 Computing cosine similarity matrix...")
similarity_matrix = cosine_similarity(np.array(features_df['embedding'].tolist()))
print(f"✅ Similarity matrix computed! Shape: {similarity_matrix.shape}")

# Find duplicates
SIMILARITY_THRESHOLD = 0.80
print(f"\\n📋 Finding duplicates (threshold > {SIMILARITY_THRESHOLD})...")

duplicates = []
urls = features_df['url'].tolist()

for i in range(len(urls)):
    for j in range(i+1, len(urls)):
        sim_score = similarity_matrix[i][j]
        if sim_score > SIMILARITY_THRESHOLD:
            duplicates.append({
                'url1': urls[i],
                'url2': urls[j],
                'similarity': round(sim_score, 3)
            })

print(f"✅ Found {len(duplicates)} duplicate pairs!")

# Thin content detection
THIN_CONTENT_THRESHOLD = 500
features_df['is_thin'] = features_df['word_count'] < THIN_CONTENT_THRESHOLD
thin_content_count = features_df['is_thin'].sum()

print(f"\\n📄 Thin content analysis:")
print(f"   Pages with < {THIN_CONTENT_THRESHOLD} words: {thin_content_count}")
print(f"   Percentage: {(thin_content_count/len(features_df)*100):.1f}%")

# Save duplicates
if len(duplicates) > 0:
    duplicates_df = pd.DataFrame(duplicates)
    duplicates_df.to_csv('data/duplicates.csv', index=False)
    print(f"\\n💾 Saved duplicates to: data/duplicates.csv")
    print(f"\\nTop duplicate pairs:")
    print(duplicates_df.head())
else:
    pd.DataFrame(columns=['url1', 'url2', 'similarity']).to_csv('data/duplicates.csv', index=False)
    print(f"\\n💾 No duplicates found (saved empty file)")

# Summary statistics
print(f"\\n📊 DUPLICATE DETECTION SUMMARY")
print(f"=" * 40)
print(f"Total pages analyzed: {len(features_df)}")
print(f"Duplicate pairs found: {len(duplicates)}")
print(f"Thin content pages: {thin_content_count} ({(thin_content_count/len(features_df)*100):.1f}%)")
print(f="=" * 40)

# SECTION: Quality Label Creation
print("\\n" + "="*60)
print("CONTENT QUALITY CLASSIFICATION")
print("="*60)

def create_quality_labels(row):
    """
    Create quality labels based on word count and readability.
    
    Criteria:
    - High: word_count > 1500 AND 50 <= readability <= 70
    - Low: word_count < 500 OR readability < 30
    - Medium: all other cases
    """
    if row['word_count'] > 1500 and 50 <= row['flesch_reading_ease'] <= 70:
        return 'High'
    elif row['word_count'] < 500 or row['flesch_reading_ease'] < 30:
        return 'Low'
    else:
        return 'Medium'

features_df['quality_label'] = features_df.apply(create_quality_labels, axis=1)

print(f"\\n🏷️ Quality label distribution:")
print(features_df['quality_label'].value_counts())
print(f"\\nLabel percentages:")
print(features_df['quality_label'].value_counts(normalize=True) * 100)

# SECTION: Model Training - Data Preparation
print("\\n🎯 Preparing data for model training...")

# Features for ML model
X = features_df[['word_count', 'sentence_count', 'flesch_reading_ease']].copy()
y = features_df['quality_label'].copy()

print(f"\\nFeature matrix shape: {X.shape}")
print(f"Target distribution:")
print(y.value_counts())

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

print(f"\\n✅ Data split complete:")
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# SECTION: Train Random Forest Model
print("\\n🌲 Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    max_depth=10,
    min_samples_split=5
)

rf_model.fit(X_train, y_train)
print(f"✅ Model trained successfully!")

# Make predictions
y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)

print(f"\\n📊 RANDOM FOREST MODEL PERFORMANCE")
print(f"=" * 60)
print(f"\\nAccuracy: {rf_accuracy:.3f}")
print(f"\\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\\nFeature Importance:")
print(feature_importance)

# SECTION: Baseline Model
print("\\n" + "="*60)
print("BASELINE MODEL COMPARISON")
print("="*60)

def baseline_predict(word_count):
    """Simple rule-based classifier using only word count."""
    if word_count > 1000:
        return 'High'
    elif word_count < 500:
        return 'Low'
    else:
        return 'Medium'

baseline_preds = X_test['word_count'].apply(baseline_predict)
baseline_accuracy = accuracy_score(y_test, baseline_preds)

print(f"\\nBaseline Model Accuracy: {baseline_accuracy:.3f}")
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
print(f"\\nImprovement: {((rf_accuracy - baseline_accuracy) / baseline_accuracy * 100):.1f}%")

print(f"\\nBaseline Classification Report:")
print(classification_report(y_test, baseline_preds))

# SECTION: Save Model
print("\\n💾 Saving trained model...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
with open('models/quality_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print(f"✅ Model saved to: models/quality_model.pkl")

# Also save the feature columns for later use
model_metadata = {
    'feature_columns': list(X.columns),
    'label_classes': list(rf_model.classes_),
    'accuracy': rf_accuracy,
    'feature_importance': feature_importance.to_dict()
}

with open('models/model_metadata.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)

print(f"✅ Model metadata saved!")

# SECTION: Real-Time URL Analysis Function
print("\\n" + "="*60)
print("REAL-TIME URL ANALYSIS FUNCTION")
print("="*60)

def analyze_url(url, existing_embeddings=None, existing_urls=None, model=None, similarity_threshold=0.75):
    """
    Analyze a given URL for content quality and find similar pages.
    
    Args:
        url (str): URL to analyze
        existing_embeddings (array): Existing embeddings for similarity comparison
        existing_urls (list): List of existing URLs
        model: Trained classification model
        similarity_threshold (float): Threshold for similarity detection
        
    Returns:
        dict: Analysis results including quality score and similar pages
    """
    try:
        # 1. Scrape the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"🌐 Fetching URL: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
        print(f"✅ Page fetched successfully!")
        
        # 2. Parse HTML
        print(f"📝 Parsing HTML content...")
        parsed = parse_html(html_content)
        
        if not parsed:
            return {"error": "Failed to parse HTML content"}
        
        # 3. Extract features
        print(f"🔍 Extracting features...")
        sentence_count = extract_sentence_count(parsed['body_text'])
        readability = extract_readability(parsed['body_text'])
        word_count = parsed['word_count']
        
        # 4. Generate embedding
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = embedding_model.encode([parsed['body_text']])[0]
        
        # 5. Predict quality
        if model is None:
            with open('models/quality_model.pkl', 'rb') as f:
                model = pickle.load(f)
        
        features = pd.DataFrame({
            'word_count': [word_count],
            'sentence_count': [sentence_count],
            'flesch_reading_ease': [readability]
        })
        
        quality_label = model.predict(features)[0]
        quality_proba = model.predict_proba(features)[0]
        
        print(f"✅ Quality predicted: {quality_label}")
        
        # 6. Find similar content
        similar_pages = []
        if existing_embeddings is not None and existing_urls is not None:
            print(f"🔗 Finding similar pages...")
            similarities = cosine_similarity([embedding], existing_embeddings)[0]
            
            for idx, sim in enumerate(similarities):
                if sim > similarity_threshold and existing_urls[idx] != url:
                    similar_pages.append({
                        'url': existing_urls[idx],
                        'similarity': round(float(sim), 3)
                    })
            
            # Sort by similarity
            similar_pages = sorted(similar_pages, key=lambda x: x['similarity'], reverse=True)[:5]
        
        # 7. Return results
        result = {
            'url': url,
            'title': parsed['title'],
            'word_count': word_count,
            'sentence_count': sentence_count,
            'readability': round(readability, 2),
            'quality_label': quality_label,
            'quality_confidence': {
                class_name: round(float(prob), 3) 
                for class_name, prob in zip(model.classes_, quality_proba)
            },
            'is_thin': word_count < 500,
            'similar_to': similar_pages
        }
        
        return result
        
    except requests.RequestException as e:
        return {"error": f"Failed to fetch URL: {str(e)}"}
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

print(f"\\n✅ analyze_url() function defined!")
print(f"\\n📝 Function signature:")
print(f"   analyze_url(url, existing_embeddings=None, existing_urls=None, model=None)")

# SECTION: Demo - Analyze Sample URL
print("\\n" + "="*60)
print("DEMO: REAL-TIME URL ANALYSIS")
print("="*60)

# Prepare existing data for similarity comparison
existing_embeddings = np.array(features_df['embedding'].tolist())
existing_urls = features_df['url'].tolist()

# Test with a sample URL from our dataset
test_url = features_df['url'].iloc[0]

print(f"\\n🎯 Analyzing sample URL from dataset:")
print(f"   {test_url}\\n")

result = analyze_url(
    test_url, 
    existing_embeddings=existing_embeddings,
    existing_urls=existing_urls,
    model=rf_model,
    similarity_threshold=0.75
)

print(f"\\n📊 ANALYSIS RESULTS")
print(f"=" * 60)
print(json.dumps(result, indent=2))

# SECTION: Summary
print("\\n" + "="*60)
print("🎉 PIPELINE EXECUTION COMPLETE!")
print("="*60)

print(f"\\n✅ All deliverables created:")
print(f"   1. ✅ data/extracted_content.csv - Parsed HTML content")
print(f"   2. ✅ data/features.csv - Extracted features")
print(f"   3. ✅ data/duplicates.csv - Duplicate pairs")
print(f"   4. ✅ models/quality_model.pkl - Trained model")
print(f"   5. ✅ analyze_url() function - Real-time analysis")

print(f"\\n📊 Final Statistics:")
print(f"   - Pages processed: {len(features_df)}")
print(f"   - Duplicate pairs: {len(duplicates)}")
print(f"   - Thin content: {thin_content_count} ({(thin_content_count/len(features_df)*100):.1f}%)")
print(f"   - Model accuracy: {rf_accuracy:.3f}")
print(f"   - Baseline accuracy: {baseline_accuracy:.3f}")

print(f"\\n🎓 Next steps:")
print(f"   1. Review the generated CSV files in the data/ folder")
print(f"   2. Test analyze_url() with different URLs")
print(f"   3. (Optional) Build Streamlit app for deployment")
print(f"   4. Update README with results")
