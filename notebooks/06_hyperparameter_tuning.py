# Step 6: Hyperparameter Tuning
# 6.1: Grid Search for Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define model
rf = RandomForestClassifier(random_state=42)

# Define grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Setup grid search
grid_search_rf = GridSearchCV(
    rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
)

# Fit
grid_search_rf.fit(X_train, y_train)

print("Best parameters (Random Forest):", grid_search_rf.best_params_)
print("Best F1 score:", grid_search_rf.best_score_)


# 6.2: Randomized Search for SVM (Faster)
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Define model
svm = SVC(probability=True)

# Define search space
param_dist = {
    'C': uniform(0.1, 10),
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

# Random search
random_search_svm = RandomizedSearchCV(
    svm, param_distributions=param_dist, n_iter=20, cv=5,
    scoring='f1', n_jobs=-1, random_state=42, verbose=1
)

# Fit
random_search_svm.fit(X_train, y_train)

print("Best parameters (SVM):", random_search_svm.best_params_)
print("Best F1 score:", random_search_svm.best_score_)


# 6.3: Evaluate the Best Model on Test Set
from sklearn.metrics import classification_report

# Use best model (example: random forest)
best_rf = grid_search_rf.best_estimator_
y_pred = best_rf.predict(X_test)

print("Evaluation on Test Set (Best RF Model):")
print(classification_report(y_test, y_pred))