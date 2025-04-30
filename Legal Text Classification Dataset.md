```python
import pandas as pd

# Load the data
df = pd.read_csv('legal_text_classification.csv')

# Examine the first few lines
print(df.head())

```


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Drop rows with missing text
df_clean = df.dropna(subset=['case_text'])

# Preprocessing function
def clean_text(text): 
text = text.lower() 
text = re.sub(r'\d+', '', text) #removenumbers 
text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation 
text = text.strip() 
text = " ".join([word for word in text.split() if word not in stop_words]) 
return text

df_clean['clean_text'] = df_clean['case_text'].apply(clean_text)
```

# Finding Main Themes from Texts with Topic Modeling (LDA)


```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Loading the dataset
data = pd.read_csv('legal_text_classification.csv')

# Selecting the case texts
texts = data['case_text'].dropna()

# Creating a word count matrix with CountVectorizer
count_vect = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X_counts = count_vect.fit_transform(texts)

# Creating and training the LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_counts)

# Function to visualize keywords and print weights
def plot_topics(model, feature_names, no_top_words): 
fig, axes = plt.subplots(5, 1, figsize=(14, 25)) # 5 subplots for 5 subjects 
for idx, topic in enumerate(model.components_): 
top_features_indices = topic.argsort()[:-no_top_words - 1:-1] 
top_features = [feature_names[i] for i in top_features_indices] 
weights = topic[top_features_indices] 

# Print to console 
print(f"Topic {idx + 1}:") 
for word, weight in zip(top_features, weights): 
print(f" {word}: {weight:.2f}") 
print("-" * 80) 

# Show in chart 
ax = axes[idx] 
bars = ax.barh(top_features, weights, color='skyblue')
ax.set_title(f"Topic {idx + 1}", fontsize=16)
ax.invert_yaxis() # Sort words from top to bottom

# Write values ​​at the end of each bar
for bar in bars:
width = bar.get_width()
ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
f'{width:.2f}', va='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('lda_topics_with_values.jpg', format='jpg')
plt.show()

# Call the function
plot_topics(lda, count_vect.get_feature_names_out(), 10)
```

# Text Similarity with Cosine Similarity


```python
# Let's install the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Reading the Dataset
df = pd.read_csv('legal_text_classification.csv')

# 2. Selecting the texts
texts = df['case_text'].astype(str)

# 3. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85)
X_tfidf = vectorizer.fit_transform(texts)

# 4. Cosine Similarity Matrix
cos_sim = cosine_similarity(X_tfidf)

# 5. For which two cases will similarity be checked? (You can change it here)
case_index_1 = 0 # Example: Case at index 10
case_index_2 = 50 # Example: Case at index 100

# 6. Let's find similar cases for a given case with a function
def find_similar_cases(case_index, top_n=5):
similarities = cos_sim[case_index]
top_matches = similarities.argsort()[::-1][1:top_n+1] # We get the first 5 with the highest score
results = []
for idx in top_matches:
case_id = df.iloc[idx]['case_id']
case_title = df.iloc[idx]['case_title']
similarity_score = similarities[idx]
excerpt = texts.iloc[idx][:300] # First 300 character summary 
results.append({ 
'Case ID': case_id, 
'Case Title': case_title, 
'Similarity Score': f"{similarity_score:.3f}", 
'Excerpt': excerpt 
}) 
results_df = pd.DataFrame(results) 
return results_df

#7. Let's find the similarities for both cases separately and print them
results_case_1 = find_similar_cases(case_index_1)
results_case_2 = find_similar_cases(case_index_2)

print(f"\nMost Similar Cases to Case {case_index_1}:")
print(results_case_1)

print(f"\nMost Similar Cases to Case {case_index_2}:")
print(results_case_2)

```


```python
# Let's install the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Reading the Dataset
df = pd.read_csv('legal_text_classification.csv')

# 2. Selecting the texts
texts = df['case_text'].astype(str)

# 3. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85)
X_tfidf = vectorizer.fit_transform(texts)

# 4. Cosine Similarity Matrix
cos_sim = cosine_similarity(X_tfidf)

# 5. For which two cases will similarity be checked? (You can change it here)
case_index_1 = 30 # Example: Case at index 30
case_index_2 = 50 # Example: Case at index 50

# 6. Let's find similar cases for a given case with a function
def find_similar_cases(case_index, top_n=5):
similarities = cos_sim[case_index]
top_matches = similarities.argsort()[::-1][1:top_n+1] # We get the first 5 with the highest score
results = []
for idx in top_matches:
case_id = df.iloc[idx]['case_id']
case_title = df.iloc[idx]['case_title']
similarity_score = similarities[idx]
excerpt = texts.iloc[idx][:300] # First 300 character summary 
results.append({ 
'Case ID': case_id, 
'Case Title': case_title, 
'Similarity Score': f"{similarity_score:.3f}", 
'Excerpt': excerpt 
}) 
results_df = pd.DataFrame(results) 
return results_df

#7. Let's find the similarities for both cases separately and print them
results_case_1 = find_similar_cases(case_index_1)
results_case_2 = find_similar_cases(case_index_2)

print(f"\nMost Similar Cases to Case {case_index_1}:")
print(results_case_1)

print(f"\nMost Similar Cases to Case {case_index_2}:")
print(results_case_2)
```

# Finding Similar Decisions (Semantic Similarity Search)


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Read the file and discard empty case_texts
df = pd.read_csv("legal_text_classification.csv")
df = df.dropna(subset=['case_text']) # ✔️ Clear lines with NaN

texts = df['case_text'].tolist()

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(texts)

# Query text
query = "The contract dispute regarding intellectual property"
query_vec = vectorizer.transform([query])

# Calculate similarity
similarities = cosine_similarity(query_vec, X_vec).flatten()
top_indices = similarities.argsort()[-5:][::-1]

# Print results
print("Most similar cases:\n")
for i in top_indices: 
print(f"Case Title: {df.iloc[i]['case_title']}") 
print(df.iloc[i]['case_text'][:300], '...\n')
```


```python
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt

# Read data set and clear NaN rows
df = pd.read_csv("legal_text_classification.csv")
df = df.dropna(subset=['case_text'])

# Gender based keywords
gender_terms = { 
"male": ["he", "him", "his", "man", "men", "boy", "father", "husband"], 
"female": ["she", "her", "hers", "woman", "women", "girl", "mother", "wife"]
}

# Count function
def count_terms(text, term_list): 
if not isinstance(text, str): 
return 0 
text = text.lower() 
return sum(text.count(term) for term in term_list)

# Calculate numbers
df['male_mentions'] = df['case_text'].apply(lambda x: count_terms(x, gender_terms['male']))
df['female_mentions'] = df['case_text'].apply(lambda x: count_terms(x, gender_terms['female']))

# Histogram plot
plt.figure(figsize=(12, 6))

# Set common bins
max_count = max(df['male_mentions'].max(), df['female_mentions'].max())
bins = range(0, max_count + 2) # intervals 1 by 1

# Plot histograms
plt.hist(df['male_mentions'], bins=bins, alpha=0.5, label='Male Mentions', color='blue', edgecolor='black', density=False)
plt.hist(df['female_mentions'], bins=bins, alpha=0.5, label='Female Mentions', color='pink', edgecolor='black', density=False)

# Graphics settings
plt.legend()
plt.title("Distribution of Gender-Related Term Mentions in Judicial Texts")
plt.xlabel("Number of Mentions per Document")
plt.ylabel("Number of Documents")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 📄 Print total statistics
print(f"Total male term mentions across all documents: {df['male_mentions'].sum()}")
print(f"Total female term mentions across all documents: {df['female_mentions'].sum()}")
print(f"Documents mentioning male terms: {sum(df['male_mentions'] > 0)}")
print(f"Documents mentioning female terms: {sum(df['female_mentions'] > 0)}")

```


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Data
data = { 
"Metric":[ 
"Total male term mentions", 
"Total female term mentions", 
"Documents mentioning male terms", 
"Documents mentioning female terms" 
], 
"Value": [1440253, 167725, 24776, 21557]
}

# Create DataFrame
summary_df = pd.DataFrame(data)

# Heatmap
plt.figure(figsize=(8, 3))
pivot_df = summary_df.pivot_table(index="Metric", values="Value")
sns.heatmap(pivot_df, annot=True, fmt='g', cmap="YlGnBu", cbar=False)
plt.title("Summary of Gender-Term Mentions in Judicial Texts")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```


```python

```

# Classification of Legal Texts with Machine Learning: Comparison of Logistic Regression, Naive Bayes and Linear SVM Methods"


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# NLTK Installation
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Loading Data
df = pd.read_csv("legal_text_classification.csv")
df_clean = df.dropna(subset=['case_text'])

# Text Cleaning and Lemmatization
def clean_text(text): 
text = text.lower() 
text = re.sub(r'\d+', '', text) 
text = text.translate(str.maketrans('', '', string.punctuation)) 
words = text.split() 
words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words] 
return " ".join(words)

df_clean.loc[:, 'clean_text'] = df_clean['case_text'].apply(clean_text)

# Class Distribution
print("\nClass Distribution:\n", df_clean['case_outcome'].value_counts())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_clean['clean_text'])
y = df_clean['case_outcome']

# Training-Test Distinction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Models
models = { 
"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42), 
"Naive Bayes": MultinomialNB(), 
"Linear SVM": LinearSVC()
}

# Setting to Plot All Confusion Matrices Underneath
fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(8, 18))

for idx, (name, clf) in enumerate(models.items()): 
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 

print(f"\n=== {name} ===") 
print("Classification Report:\n", classification_report(y_test, y_pred)) 

# Numerical Confusion Matrix 
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_) 
print("Confusion Matrix (Numeric Values):\n", cm) 

# Visual Confusion Matrix 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_) 
disp.plot(ax=axes[idx], cmap='Blues', colorbar=False) 
axes[idx].set_title(f"{name} - Confusion Matrix") 
axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

# Classification of Legal Texts with Machine Learning: Logistic Regression, Naive Bayes, Linear SVM and SMOTE with Balancing and GridSearchCV Hyperparameter Optimization


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from imbleearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

#NLTK Setup
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Loading Data
df = pd.read_csv("legal_text_classification.csv")
df_clean = df.dropna(subset=['case_text'])

# Text Cleaning and Lemmatization
def clean_text(text): 
text = text.lower() 
text = re.sub(r'\d+', '', text) 
text = text.translate(str.maketrans('', '', string.punctuation)) 
words = text.split() 
words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words] 
return " ".join(words)

df_clean.loc[:, 'clean_text'] = df_clean['case_text'].apply(clean_text)

# Encode Class Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_clean['case_outcome'])
X_text = df_clean['clean_text']

#TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Balancing with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Training-Test Distinction
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# GridSearchCV for Logistic Regression
log_param_grid = { 
'C': [0.1, 1, 10], 
'penalty': ['l2'], 
'solver': ['lbfgs']
}
log_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
log_grid.fit(X_train, y_train)

log_best_model = log_grid.best_estimator_
log_y_pred = log_best_model.predict(X_test)

print(f"\nBest Parameters (Logistic Regression): {log_grid.best_params_}")
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, log_y_pred))

# GridSearchCV for Naive Bayes
nb_param_grid = { 
'alpha': [0.1, 1, 10]
}
nb_grid = GridSearchCV(MultinomialNB(), nb_param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
nb_grid.fit(X_train, y_train)

nb_best_model = nb_grid.best_estimator_
nb_y_pred = nb_best_model.predict(X_test)

print(f"\nBest Parameters (Naive Bayes): {nb_grid.best_params_}")
print("\nClassification Report (Naive Bayes):\n", classification_report(y_test, nb_y_pred))

# GridSearchCV for Linear SVM (kernel parameterless)
svm_param_grid = {
'C': [0.1, 1, 10],
'penalty': ['l2'], # We specify the penalty type for LinearSVC
'loss': ['squared_hinge'] # We specify the loss function for LinearSVC
}
svm_grid = GridSearchCV(LinearSVC(max_iter=1000, random_state=42), svm_param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
svm_grid.fit(X_train, y_train)

svm_best_model = svm_grid.best_estimator_
svm_y_pred = svm_best_model.predict(X_test)

print(f"\nBest Parameters (Linear SVM): {svm_grid.best_params_}")
print("\nClassification Report (Linear SVM):\n", classification_report(y_test, svm_y_pred))

# Confusion Matrix for Logistic Regression
plt.figure(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_test, log_y_pred, display_labels=label_encoder.classes_, cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Confusion Matrix for Naive Bayes
plt.figure(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_test, nb_y_pred, display_labels=label_encoder.classes_, cmap='Blues')
plt.title("Naive Bayes Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Confusion Matrix for Linear SVM
plt.figure(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_test, svm_y_pred, display_labels=label_encoder.classes_, cmap='Blues')
plt.title("Linear SVM Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-validation for Logistic Regression
log_cv_scores = cross_val_score(log_best_model, X_resampled, y_resampled, cv=5, scoring='f1_macro')
print(f"\nCross-validation F1 Macro scores (Logistic Regression): {log_cv_scores}")
print(f"Mean F1 Macro (Logistic Regression): {np.mean(log_cv_scores):.4f}")

# Cross-validation for Naive Bayes
nb_cv_scores = cross_val_score(nb_best_model, X_resampled, y_resampled, cv=5, scoring='f1_macro')
print(f"\nCross-validation F1 Macro scores (Naive Bayes): {nb_cv_scores}")
print(f"Mean F1 Macro (Naive Bayes): {np.mean(nb_cv_scores):.4f}")

# Cross-validation for Linear SVM
svm_cv_scores = cross_val_score(svm_best_model, X_resampled, y_resampled, cv=5, scoring='f1_macro')
print(f"\nCross-validation F1

```


```python
# Confusion Matrix and Numeric Values ​​- Single Column

fig, axs = plt.subplots(3, 1, figsize=(8, 18)) # 3 rows, 1 column

# Logistic Regression
cm_log = confusion_matrix(y_test, log_y_pred)
print("\n=== Logistic Regression Confusion Matrix (Numeric Values) ===")
print(cm_log)
ConfusionMatrixDisplay(cm_log, display_labels=label_encoder.classes_).plot(ax=axs[0], cmap='Blues', colorbar=False)
axs[0].set_title("Logistic Regression Confusion Matrix")
axs[0].tick_params(axis='x', rotation=45)

# Naive Bayes
cm_nb = confusion_matrix(y_test, nb_y_pred)
print("\n=== Naive Bayes Confusion Matrix (Numeric Values) ===")
print(cm_nb)
ConfusionMatrixDisplay(cm_nb, display_labels=label_encoder.classes_).plot(ax=axs[1], cmap='Blues', colorbar=False)
axs[1].set_title("Naive Bayes Confusion Matrix")
axs[1].tick_params(axis='x', rotation=45)

# Linear SVM
cm_svm = confusion_matrix(y_test, svm_y_pred)
print("\n=== Linear SVM Confusion Matrix (Numeric Values) ===")
print(cm_svm)
ConfusionMatrixDisplay(cm_svm, display_labels=label_encoder.classes_).plot(ax=axs[2], cmap='Blues', colorbar=False)
axs[2].set_title("Linear SVM Confusion Matrix")
axs[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```




```python


```


```python

```


```python

```


```python

```


```python

```


```python

```
