import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.stats import ttest_ind

import statsmodels.api as sm

from sklearn.feature_extraction.text import CountVectorizer


file_path = "Response.xlsx"

df = pd.read_excel(file_path)

df.columns = df.columns.str.strip()

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== ALL COLUMN NAMES =====")
for i, col in enumerate(df.columns):
    print(i, ":", repr(col))


if 'Timestamp' in df.columns:
    df = df.drop(columns=['Timestamp'])


threshold = len(df.columns) * 0.7
df = df.dropna(thresh=threshold)


for col in df.columns:

    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')

    else:
        df[col] = df[col].fillna(df[col].mean())

# =========================================================
# RENAMING THE  IMPORTANT COLUMNS
# =========================================================

rename_dict = {}

for col in df.columns:

    if "How many hours per day do you spend on social media" in col:
        rename_dict[col] = "SM_Hours"

    elif "How often do you encounter educational or career related content" in col:
        rename_dict[col] = "Educational_Content_Exposure"

    elif "How often do you search for study/career information" in col:
        rename_dict[col] = "Career_Search_Frequency"

    elif "How frequently do you use AI tools for academic purposes" in col:
        rename_dict[col] = "AI_Usage_Frequency"

    elif "Did social media influence your final study decision" in col:
        rename_dict[col] = "SM_Final_Decision"

    elif "Did AI tools influence your final study decision" in col:
        rename_dict[col] = "AI_Final_Decision"

    elif "How confident are you about your chosen study path" in col:
        rename_dict[col] = "Decision_Confidence"

    elif "Which platform provides you with the most education-related information" in col:
        rename_dict[col] = "Education_Platform"

    elif "What advice would you give future students" in col:
        rename_dict[col] = "Student_Advice"

    elif "Which AI tool do you use most often" in col:
        rename_dict[col] = "AI_Tool"

# Rename columns
df = df.rename(columns=rename_dict)

print("\n===== RENAMED COLUMNS =====")
print(df.columns)

# =========================================================
#  CLEAN & MAP SOCIAL MEDIA HOURS
# =========================================================
df['SM_Hours'] = (
    df['SM_Hours']
    .astype(str)
    .str.strip()
    .str.replace(' - ', '-', regex=False)
)

sm_map = {
    'Less than 1 hour': 1,
    '1-3 hours': 2,
    '4-6 hours': 3,
    'More than 6 hours': 4
}

df['SM_Hours'] = df['SM_Hours'].map(sm_map)

# =========================================================
# CONVERT YES/NO TO BINARY
# =========================================================
binary_map = {
    'Yes': 1,
    'YES': 1,
    'No': 0,
    'NO': 0
}

df['SM_Final_Decision'] = df['SM_Final_Decision'].map(binary_map)
df['AI_Final_Decision'] = df['AI_Final_Decision'].map(binary_map)

# =========================================================
# CREATE SOCIAL MEDIA INFLUENCE SCORE
# =========================================================
sm_cols = []

for col in df.columns:

    if col.startswith('A.'):
        sm_cols.append(col)

    elif col.startswith('B.'):
        sm_cols.append(col)

    elif col.startswith('C.'):
        sm_cols.append(col)

    elif col.startswith('D.'):
        sm_cols.append(col)

    elif col.startswith('E.'):
        sm_cols.append(col)

    elif col.startswith('F.'):
        sm_cols.append(col)

    elif col.startswith('G.'):
        sm_cols.append(col)

df['SM_Influence'] = df[sm_cols].mean(axis=1)

# =========================================================
# CREATE AI INFLUENCE SCORE
# =========================================================
ai_cols = []

for col in df.columns:

    if col.startswith('H.'):
        ai_cols.append(col)

    elif col.startswith('I.'):
        ai_cols.append(col)

    elif col.startswith('J.'):
        ai_cols.append(col)

    elif col.startswith('K.'):
        ai_cols.append(col)

    elif col.startswith('L.'):
        ai_cols.append(col)

    elif col.startswith('M.'):
        ai_cols.append(col)

df['AI_Influence'] = df[ai_cols].mean(axis=1)

# =========================================================
# CONVERT IMPORTANT COLUMNS TO NUMERIC
# =========================================================
numeric_cols = [
    'SM_Influence',
    'AI_Influence',
    'SM_Hours',
    'AI_Usage_Frequency',
    'Decision_Confidence',
    'Educational_Content_Exposure',
    'Career_Search_Frequency'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================================================
# CHECK MISSING VALUES
# =========================================================
print("\n===== MISSING VALUES =====")
print(df[numeric_cols].isna().sum())

# =========================================================
# AI VS SOCIAL MEDIA COMPARISON
# =========================================================
means = df[['SM_Influence', 'AI_Influence']].mean()

plt.figure(figsize=(6,5))

plt.bar(means.index, means.values)

plt.title('AI vs Social Media Influence')
plt.ylabel('Mean Score')

plt.grid(axis='y')

plt.show()

print("\n===== MEAN SCORES =====")
print(means)

# =========================================================
# CORRELATION ANALYSIS
# =========================================================
corr_vars = [
    'SM_Influence',
    'AI_Influence',
    'Decision_Confidence',
    'SM_Hours',
    'Educational_Content_Exposure',
    'Career_Search_Frequency',
    'AI_Usage_Frequency'
]

corr_matrix = df[corr_vars].corr()

plt.figure(figsize=(8,6))

plt.imshow(corr_matrix)

plt.colorbar()

plt.xticks(range(len(corr_vars)), corr_vars, rotation=45)
plt.yticks(range(len(corr_vars)), corr_vars)

for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        plt.text(
            j,
            i,
            round(corr_matrix.iloc[i, j], 2),
            ha='center',
            va='center'
        )

plt.title('Correlation Matrix')

plt.tight_layout()

plt.show()

print("\n===== CORRELATION MATRIX =====")
print(corr_matrix)

# =========================================================
# OLS REGRESSION
# =========================================================
X = df[['SM_Influence', 'AI_Influence', 'SM_Hours', 'AI_Usage_Frequency']]
y = df['Decision_Confidence']

X = X.replace([np.inf, -np.inf], np.nan)

ols_data = pd.concat([X, y], axis=1)

ols_data = ols_data.dropna()

X = ols_data[['SM_Influence', 'AI_Influence', 'SM_Hours', 'AI_Usage_Frequency']]
y = ols_data['Decision_Confidence']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\n===== OLS REGRESSION SUMMARY =====")
print(model.summary())

# =========================================================
#  MACHINE LEARNING REGRESSION
# =========================================================
ml_data = df[[
    'SM_Influence',
    'AI_Influence',
    'SM_Hours',
    'AI_Usage_Frequency',
    'Decision_Confidence'
]].dropna()

X_ml = ml_data[[
    'SM_Influence',
    'AI_Influence',
    'SM_Hours',
    'AI_Usage_Frequency'
]]

y_ml = ml_data['Decision_Confidence']

X_train, X_test, y_train, y_test = train_test_split(
    X_ml,
    y_ml,
    test_size=0.2,
    random_state=42
)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

plt.figure(figsize=(6,5))

plt.scatter(y_test, y_pred)

plt.xlabel('Actual Confidence')
plt.ylabel('Predicted Confidence')

plt.title('Regression Performance')

plt.grid()

plt.show()

# =========================================================
# T-TEST
# =========================================================
median_sm = df['SM_Hours'].median()

high_sm = df[df['SM_Hours'] > median_sm]['SM_Influence']
low_sm = df[df['SM_Hours'] <= median_sm]['SM_Influence']

t_stat, p_val = ttest_ind(high_sm, low_sm)

print("\n===== T-TEST RESULTS =====")
print("T-statistic:", t_stat)
print("P-value:", p_val)

# =========================================================
# LOGISTIC REGRESSION
# =========================================================
log_data = df[[
    'AI_Influence',
    'AI_Usage_Frequency',
    'AI_Final_Decision'
]].dropna()

X = log_data[['AI_Influence', 'AI_Usage_Frequency']]
y = log_data['AI_Final_Decision']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# =========================================================
# CLUSTERING ANALYSIS
# =========================================================
cluster_data = df[[
    'SM_Influence',
    'AI_Influence',
    'SM_Hours',
    'AI_Usage_Frequency'
]].dropna()

# Standardize
scaler = StandardScaler()

X_scaled = scaler.fit_transform(cluster_data)

# KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)

cluster_data['Cluster'] = cluster_labels

cluster_summary = cluster_data.groupby('Cluster').mean()

print("\n===== CLUSTER SUMMARY =====")
print(cluster_summary)

# =========================================================
# CLUSTER VISUALIZATION
# =========================================================
plt.figure(figsize=(7,6))

plt.scatter(
    cluster_data['SM_Influence'],
    cluster_data['AI_Influence'],
    c=cluster_data['Cluster']
)

plt.xlabel('Social Media Influence')
plt.ylabel('AI Influence')

plt.title('Student Segments')

plt.grid()

plt.show()

# =========================================================
# PLATFORM ANALYSIS
# =========================================================
platform_counts = df['Education_Platform'].value_counts()

plt.figure(figsize=(7,5))

plt.bar(
    platform_counts.index,
    platform_counts.values
)

plt.xticks(rotation=45)

plt.ylabel('Count')

plt.title('Educational Platform Usage')

plt.grid(axis='y')

plt.show()

# =========================================================
# TEXT ANALYSIS
# =========================================================
text_data = df['Student_Advice'].dropna()

vectorizer = CountVectorizer(stop_words='english')

X_text = vectorizer.fit_transform(text_data)

word_freq = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'freq': X_text.sum(axis=0).tolist()[0]
}).sort_values(by='freq', ascending=False)

print("\n===== TOP KEYWORDS =====")
print(word_freq.head(10))

# Plot keywords
top_words = word_freq.head(10)

plt.figure(figsize=(7,5))

plt.barh(
    top_words['word'],
    top_words['freq']
)

plt.gca().invert_yaxis()

plt.xlabel('Frequency')

plt.title('Top Keywords in Student Advice')

plt.show()

# =========================================================
# AI VS SOCIAL MEDIA DISTRIBUTION
# =========================================================

plt.figure(figsize=(8,5))

plt.hist(
    df['SM_Influence'].dropna(),
    bins=10,
    alpha=0.7,
    label='Social Media Influence'
)

plt.hist(
    df['AI_Influence'].dropna(),
    bins=10,
    alpha=0.7,
    label='AI Influence'
)

plt.xlabel('Influence Score')
plt.ylabel('Frequency')

plt.title('Distribution of AI vs Social Media Influence')

plt.legend()

plt.grid()

plt.show()

# =========================================================
# DECISION CONFIDENCE BY AI USAGE
# =========================================================

confidence_by_ai = df.groupby(
    'AI_Usage_Frequency'
)['Decision_Confidence'].mean()

plt.figure(figsize=(7,5))

plt.plot(
    confidence_by_ai.index,
    confidence_by_ai.values,
    marker='o'
)

plt.xlabel('AI Usage Frequency')
plt.ylabel('Average Decision Confidence')

plt.title('Decision Confidence vs AI Usage Frequency')

plt.grid()

plt.show()

print("\n===== CONFIDENCE BY AI USAGE =====")
print(confidence_by_ai)

# =========================================================
# DECISION CONFIDENCE BY SOCIAL MEDIA HOURS
# =========================================================

confidence_by_sm = df.groupby(
    'SM_Hours'
)['Decision_Confidence'].mean()

plt.figure(figsize=(7,5))

plt.plot(
    confidence_by_sm.index,
    confidence_by_sm.values,
    marker='o'
)

plt.xlabel('Social Media Hours')
plt.ylabel('Average Decision Confidence')

plt.title('Decision Confidence vs Social Media Hours')

plt.grid()

plt.show()

print("\n===== CONFIDENCE BY SOCIAL MEDIA HOURS =====")
print(confidence_by_sm)

# =========================================================
# AI TOOLS VS AI INFLUENCE
# =========================================================

ai_tool_influence = df.groupby(
    'AI_Tool'
)['AI_Influence'].mean()

plt.figure(figsize=(8,5))

plt.bar(
    ai_tool_influence.index,
    ai_tool_influence.values
)

plt.xticks(rotation=45)

plt.ylabel('Average AI Influence')

plt.title('AI Influence Across AI Tools')

plt.grid(axis='y')

plt.show()

# =========================================================
# PLATFORM VS SOCIAL MEDIA INFLUENCE
# =========================================================

platform_sm = df.groupby(
    'Education_Platform'
)['SM_Influence'].mean()

plt.figure(figsize=(8,5))

plt.bar(
    platform_sm.index,
    platform_sm.values
)

plt.xticks(rotation=45)

plt.ylabel('Average Social Media Influence')

plt.title('Social Media Influence Across Platforms')

plt.grid(axis='y')

plt.show()

# =========================================================
# ELBOW METHOD
# =========================================================

inertia = []

for k in range(1, 8):

    km = KMeans(
        n_clusters=k,
        random_state=42
    )

    km.fit(X_scaled)

    inertia.append(km.inertia_)

plt.figure(figsize=(7,5))

plt.plot(
    range(1,8),
    inertia,
    marker='o'
)

plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.title('Elbow Method')

plt.grid()

plt.show()

# =========================================================
# REGRESSION COEFFICIENT VISUALIZATION
# =========================================================

coefficients = pd.DataFrame({

    'Variable': [
        'SM_Influence',
        'AI_Influence',
        'SM_Hours',
        'AI_Usage_Frequency'
    ],

    'Coefficient': model.params[1:]
})

plt.figure(figsize=(7,5))

plt.bar(
    coefficients['Variable'],
    coefficients['Coefficient']
)

plt.ylabel('Coefficient Value')

plt.title('OLS Regression Coefficients')

plt.grid(axis='y')

plt.show()

print("\n===== REGRESSION COEFFICIENTS =====")
print(coefficients)

# =========================================================
# FINAL DECISION COMPARISON
# =========================================================

final_decision = {

    'Social Media':
        df['SM_Final_Decision'].mean(),

    'AI Tools':
        df['AI_Final_Decision'].mean()
}

plt.figure(figsize=(6,5))

plt.bar(
    final_decision.keys(),
    final_decision.values()
)

plt.ylabel('Average Influence Rate')

plt.title('Final Study Decision Influence')

plt.grid(axis='y')

plt.show()

print("\n===== FINAL DECISION INFLUENCE =====")
print(final_decision)

# =========================================================
# DECISION CONFIDENCE DISTRIBUTION
# =========================================================

plt.figure(figsize=(7,5))

plt.hist(
    df['Decision_Confidence'].dropna(),
    bins=5
)

plt.xlabel('Decision Confidence')
plt.ylabel('Frequency')

plt.title('Study Path Confidence Distribution')

plt.grid()

plt.show()

# =========================================================
# PROFESSIONAL CLUSTER LABELS
# =========================================================

cluster_labels = {

    0: 'AI-Driven Decision Makers',

    1: 'Social Media Influenced Explorers',

    2: 'Independent Low-Engagement Students'
}

cluster_data['Cluster_Name'] = cluster_data[
    'Cluster'
].map(cluster_labels)

print("\n===== CLUSTER LABELS =====")

print(
    cluster_data[
        ['Cluster', 'Cluster_Name']
    ].head()
)

# =========================================================
# PLATFORM PREFERENCE PIE CHART
# =========================================================

platform_counts = df[
    'Education_Platform'
].value_counts()

plt.figure(figsize=(8,8))

plt.pie(
    platform_counts.values,
    labels=platform_counts.index,
    autopct='%1.1f%%'
)

plt.title('Preferred Educational Platforms')

plt.show()

df.to_csv(
    'cleaned_analysis.csv',
    index=False
)


# =========================================================
#  PAIRPLOT
# =========================================================

sns.pairplot(df[[
    'SM_Influence',
    'AI_Influence',
    'SM_Hours',
    'AI_Usage_Frequency',
    'Decision_Confidence'
]])
plt.show()

# =========================================================
# BOX PLOT: AI INFLUENCE VS USAGE
# =========================================================

plt.figure(figsize=(7,5))
sns.boxplot(x='AI_Usage_Frequency', y='AI_Influence', data=df)
plt.title('AI Influence across AI Usage Levels')
plt.grid()
plt.show()

# =========================================================
# BOX PLOT: SM INFLUENCE VS HOURS
# =========================================================

plt.figure(figsize=(7,5))
sns.boxplot(x='SM_Hours', y='SM_Influence', data=df)
plt.title('Social Media Influence across Usage Levels')
plt.grid()
plt.show()

# =========================================================
#  REGRESSION PLOT: SM vs CONFIDENCE
# =========================================================

plt.figure(figsize=(6,5))
sns.regplot(x='SM_Influence', y='Decision_Confidence', data=df)
plt.title('SM Influence vs Decision Confidence')
plt.grid()
plt.show()

# =========================================================
#  REGRESSION PLOT: AI vs CONFIDENCE
# =========================================================

plt.figure(figsize=(6,5))
sns.regplot(x='AI_Influence', y='Decision_Confidence', data=df)
plt.title('AI Influence vs Decision Confidence')
plt.grid()
plt.show()

# =========================================================
# HEATMAP CORRELATION
# =========================================================

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# =========================================================
# VIOLIN PLOT
# =========================================================

plt.figure(figsize=(7,5))
sns.violinplot(x='SM_Hours', y='Decision_Confidence', data=df)
plt.title('Confidence Distribution across SM Usage')
plt.grid()
plt.show()

# =========================================================
# CLUSTER SIZE DISTRIBUTION
# =========================================================

df_clusters = cluster_data['Cluster'].value_counts().sort_index()

plt.figure(figsize=(6,4))
plt.bar(df_clusters.index, df_clusters.values)
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Cluster Size Distribution")
plt.grid(axis='y')
plt.show()

# =========================================================
# PCA VISUALIZATION OF CLUSTERS
# =========================================================

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_data['Cluster'],
    cmap='viridis'
)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Cluster Visualization using PCA')
plt.grid()
plt.show()

# =========================================================
# FEATURE IMPORTANCE (LINEAR REGRESSION)
# =========================================================

feature_names = ['SM_Influence','AI_Influence','SM_Hours','AI_Usage_Frequency']
importance = lr.coef_

plt.figure(figsize=(7,5))
plt.bar(feature_names, importance)
plt.title('Feature Importance (Regression)')
plt.grid(axis='y')
plt.show()

# =========================================================
# MEDIATION INSIGHT (SM → AI → CONFIDENCE APPROX)
# =========================================================

plt.figure(figsize=(6,5))
sns.scatterplot(x=df['SM_Influence'], y=df['AI_Influence'])
plt.title('Mediation Check: SM Influence → AI Influence')
plt.grid()
plt.show()

df.to_csv(
    'cleaned_analysis.csv',
    index=False
)

print("\n===== ANALYSIS COMPLETED SUCCESSFULLY =====")