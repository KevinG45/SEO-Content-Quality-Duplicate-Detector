# Script to add all remaining cells to notebook
# This comprehensive script will be converted to notebook cells

CELLS_TO_ADD = [
    {
        "type": "markdown",
        "content": "## 4. Text Preprocessing and Feature Engineering - Basic Metrics\n\nExtract sentence count and readability scores from parsed text."
    },
    {
        "type": "code",
        "content": """def extract_sentence_count(text):
    \"\"\"Extract sentence count from text.\"\"\"
    try:
        sentences = nltk.sent_tokenize(text)
        return len(sentences)
    except:
        return len([s for s in text.split('.') if s.strip()])

def extract_readability(text):
    \"\"\"Calculate Flesch Reading Ease score.\"\"\"
    try:
        return textstat.flesch_reading_ease(text)
    except:
        return 50.0  # Default moderate readability

# Add basic features
print("📊 Extracting basic features...")
extracted_df['sentence_count'] = extracted_df['body_text'].apply(extract_sentence_count)
extracted_df['flesch_reading_ease'] = extracted_df['body_text'].apply(extract_readability)

print(f"✅ Sentence count and readability scores calculated!")
print(f"\\nFeature statistics:")
print(extracted_df[['word_count', 'sentence_count', 'flesch_reading_ease']].describe())"""
    },
    {
        "type": "markdown",
        "content": "## 5. Feature Engineering - Keyword Extraction\n\nExtract top 5 keywords using TF-IDF vectorization."
    },
    {
        "type": "code",
        "content": """def extract_top_keywords(texts, n_keywords=5):
    \"\"\"Extract top N keywords for each document using TF-IDF.\"\"\"
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

print("🔑 Extracting keywords using TF-IDF...")
extracted_df['top_keywords'] = extract_top_keywords(extracted_df['body_text'].tolist())
print(f"✅ Keywords extracted!")
print(f"\\nSample keywords:")
for idx in range(min(3, len(extracted_df))):
    print(f"{idx+1}. {extracted_df.iloc[idx]['top_keywords']}")"""
    },
    {
        "type": "markdown",
        "content": "## 6. Feature Engineering - Text Embeddings\n\nGenerate semantic embeddings using Sentence Transformers (all-MiniLM-L6-v2)."
    },
    {
        "type": "code",
        "content": """print("🧠 Generating text embeddings (this may take a minute)...")

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
    print(f"✅ TF-IDF embeddings generated! Shape: {embeddings.shape}")"""
    },
    {
        "type": "markdown",
        "content": "## 7. Save Extracted Features\n\nCombine all features and save to CSV."
    },
    {
        "type": "code",
        "content": """# Create features DataFrame
features_df = extracted_df[['url', 'word_count', 'sentence_count', 'flesch_reading_ease', 'top_keywords', 'embedding']].copy()

# Convert embeddings to string for CSV storage
features_df['embedding_vector'] = features_df['embedding'].apply(lambda x: str(x.tolist() if hasattr(x, 'tolist') else x))
features_df_save = features_df.drop('embedding', axis=1)

features_df_save.to_csv('../data/features.csv', index=False)
print(f"💾 Saved features to: data/features.csv")
print(f"\\nFeature summary:")
print(features_df[['word_count', 'sentence_count', 'flesch_reading_ease']].describe())"""
    },
    {
        "type": "markdown",
        "content": "## 8. Duplicate Detection - Compute Similarity Matrix\n\nCalculate pairwise cosine similarity between all documents."
    },
    {
        "type": "code",
        "content": """print("=" * 60)
print("DUPLICATE DETECTION")
print("=" * 60)

print("\\n🔍 Computing cosine similarity matrix...")
similarity_matrix = cosine_similarity(np.array(features_df['embedding'].tolist()))
print(f"✅ Similarity matrix computed! Shape: {similarity_matrix.shape}")

# Visualize similarity distribution
plt.figure(figsize=(10, 6))
sim_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
plt.hist(sim_values, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Content Similarity Scores')
plt.axvline(x=0.80, color='r', linestyle='--', label='Threshold (0.80)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\\nSimilarity statistics:")
print(f"  Mean: {sim_values.mean():.3f}")
print(f"  Median: {np.median(sim_values):.3f}")
print(f"  Max: {sim_values.max():.3f}")"""
    },
    {
        "type": "markdown",
        "content": "## 9. Duplicate Detection - Find Duplicate Pairs\n\nIdentify document pairs above similarity threshold."
    },
    {
        "type": "code",
        "content": """# Find duplicates
SIMILARITY_THRESHOLD = 0.80
print(f"📋 Finding duplicates (threshold > {SIMILARITY_THRESHOLD})...")

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

print(f"✅ Found {len(duplicates)} duplicate pairs!")"""
    },
    {
        "type": "markdown",
        "content": "## 10. Thin Content Detection\n\nFlag pages with word count below 500."
    },
    {
        "type": "code",
        "content": """# Thin content detection
THIN_CONTENT_THRESHOLD = 500
features_df['is_thin'] = features_df['word_count'] < THIN_CONTENT_THRESHOLD
thin_content_count = features_df['is_thin'].sum()

print(f"📄 Thin content analysis:")
print(f"   Pages with < {THIN_CONTENT_THRESHOLD} words: {thin_content_count}")
print(f"   Percentage: {(thin_content_count/len(features_df)*100):.1f}%")

# Visualize word count distribution
plt.figure(figsize=(10, 6))
plt.hist(features_df['word_count'], bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=THIN_CONTENT_THRESHOLD, color='r', linestyle='--', label=f'Thin Content Threshold ({THIN_CONTENT_THRESHOLD})')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Distribution of Content Length')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""
    },
    {
        "type": "markdown",
        "content": "## 11. Save Duplicate Results\n\nSave identified duplicate pairs and summary statistics."
    },
    {
        "type": "code",
        "content": """# Save duplicates
if len(duplicates) > 0:
    duplicates_df = pd.DataFrame(duplicates)
    duplicates_df.to_csv('../data/duplicates.csv', index=False)
    print(f"💾 Saved duplicates to: data/duplicates.csv")
    print(f"\\nTop duplicate pairs:")
    print(duplicates_df.head(10))
else:
    pd.DataFrame(columns=['url1', 'url2', 'similarity']).to_csv('../data/duplicates.csv', index=False)
    print(f"💾 No duplicates found (saved empty file)")

# Summary statistics
print(f"\\n📊 DUPLICATE DETECTION SUMMARY")
print(f"=" * 40)
print(f"Total pages analyzed: {len(features_df)}")
print(f"Duplicate pairs found: {len(duplicates)}")
print(f"Thin content pages: {thin_content_count} ({(thin_content_count/len(features_df)*100):.1f}%)")
print(f"=" * 40)"""
    }
]

# Save as JSON for reference
import json
with open('notebook_cells_config.json', 'w') as f:
    json.dump(CELLS_TO_ADD, f, indent=2)

print(f"✅ Created configuration for {len(CELLS_TO_ADD)} additional cells!")
