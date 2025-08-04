#  Step 3: Feature Selection
#  3.1 Feature Importance (Random Forest)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Train Random Forest on full dataset
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled_df, y)

# Get feature importances
importances = rf.feature_importances_
features = X_scaled_df.columns
indices = np.argsort(importances)[::-1]  # descending order

# Plot top 15 features
plt.figure(figsize=(10, 6))
plt.title("Top 15 Feature Importances (Random Forest)")
plt.bar(range(15), importances[indices[:15]], align="center")
plt.xticks(range(15), features[indices[:15]], rotation=90)
plt.tight_layout()
plt.show()


# 3.2 Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X_scaled_df, y)

# Define X_selected
X_selected = X_scaled_df.loc[:, rfe.support_]


# Get selected features
selected_rfe = X_scaled_df.columns[rfe.support_]
print("Top 10 features selected by RFE:")
print(selected_rfe)

# 3.3 Chi-Square Test
from sklearn.feature_selection import SelectKBest, chi2

# Chi-Square needs non-negative values
X_positive = X_scaled_df.copy()
X_positive[X_positive < 0] = 0  # Replace negative values with 0 just for chi2 test

chi2_selector = SelectKBest(chi2, k=10)
chi2_selector.fit(X_positive, y)

# Show selected features
selected_chi2 = X_scaled_df.columns[chi2_selector.get_support()]
print("Top 10 features selected by Chi-Square Test:")
print(selected_chi2)