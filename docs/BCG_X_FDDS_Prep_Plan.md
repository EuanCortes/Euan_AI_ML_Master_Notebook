# BCG X — Forward Deployed AI Data Scientist
## CodeSignal Technical Assessment — 6-Day Prep Plan

**Assessment focus areas:** Data Cleaning & Preprocessing · Data Querying & Retrieval · Exploratory Data Analysis & Visualization · Machine Learning & Predictive Modeling
**Language:** Python only (mandatory — non-Python = disqualification)
**Format:** Timed, proctored, auto-graded on exact outputs. One browser tab allowed for *syntax reference only*. Scratch paper allowed.

---

## Guiding principles

1. **Speed and fluency beat depth.** The test rewards fast, correct dataframe wrangling and a clean model pipeline — not theory. Every hour should end with you *faster* at something.
2. **Learn libraries through practice, not in isolation.** Do the fluency pass once, then spend the bulk of your time on end-to-end timed runs.
3. **Produce exact outputs.** Auto-graders check specific returned values, DataFrame shapes, or variables. Always give the precise thing asked for.
4. **Bank easy points first.** Tasks usually escalate in difficulty — never get stuck on a hard one while easy points sit unclaimed.

---

## The 6-Day Schedule

> Put the heavy blocks (Days 1, and the full timed runs) on your non-work days. On work days, do the lighter single-task drills.

| Day | Focus | Load |
|-----|-------|------|
| **Day 1** | Rapid fluency pass through all libraries + build your cheat sheet (this doubles as your allowed reference tab) | Heavy |
| **Day 2** | First full end-to-end run (untimed), then a second one timed | Heavy |
| **Day 3** | Two timed full runs on new datasets (1 classification, 1 regression) | Heavy |
| **Day 4** | *(work day)* Single-task drills: cleaning gymnastics + one groupby/merge set | Light |
| **Day 5** | Two timed full runs + drill your weakest area from the week | Heavy |
| **Day 6** | One final timed run, review cheat sheet, **logistics + environment check** | Light |

**Day 6 logistics checklist:** quiet room (be alone), single monitor, connected to power, camera lighting good and full face visible, unexpired government photo ID ready, all other apps/tabs closed except the one test tab + one syntax tab, phone away, test that your Python environment / notebook runs.

---

## Cheat Sheet — Functions & Methods to Know Cold

Build a personal `.md` or paper cheat sheet with a one-line idiom for each. Don't just read them — run each once.

### NumPy
```
np.array, np.arange, np.linspace, np.zeros, np.ones, np.full
.shape, .reshape, .ravel, .flatten, .T (transpose)
np.mean, np.median, np.std, np.var, np.sum, np.min, np.max, np.percentile
np.argmin, np.argmax, np.argsort, np.sort, np.unique
axis=0 vs axis=1 (know which is which cold)
Boolean masking: arr[arr > 5]
np.where(cond, a, b)
np.isnan, np.nan, np.isfinite
np.concatenate, np.vstack, np.hstack, np.stack
np.dot, @ operator, np.linalg.inv, np.linalg.norm
Broadcasting rules (scalar op array, row op matrix)
np.random.seed, np.random.rand, np.random.randn, np.random.choice
```

### pandas — Loading & Inspecting
```
pd.read_csv (know: sep, header, names, na_values, parse_dates, dtype, usecols, nrows, index_col)
pd.read_excel, pd.read_json, pd.read_sql
.head, .tail, .sample, .shape, .info, .describe, .dtypes
.columns, .index, .values
.nunique, .unique, .value_counts (normalize=True is useful)
.memory_usage
```

### pandas — Selecting & Filtering
```
df['col'], df[['a','b']]
.loc[rows, cols]  (label-based)
.iloc[rows, cols] (position-based)
Boolean filtering: df[df['x'] > 5]
Multiple conditions: df[(df.a > 1) & (df.b < 3)]  # parentheses matter
.isin([...]), .between(a, b)
.query("x > 5 and y == 'a'")
.filter(), .where()
```

### pandas — Cleaning & Preprocessing
```
.isna / .isnull, .notna, .any, .sum (for counting NaNs per column)
.dropna(subset=, how=, thresh=, axis=)
.fillna(value / method='ffill'/'bfill' / dict per column)
.drop(columns=), .drop_duplicates(subset=, keep=)
.rename(columns={})
.astype(), pd.to_numeric(errors='coerce'), pd.to_datetime(), pd.to_timedelta()
.replace(), .map(), .clip(lower=, upper=)
String ops: .str.strip, .str.lower, .str.upper, .str.contains,
            .str.replace, .str.split, .str.extract, .str.len, .str.startswith
.duplicated()
pd.cut (binning continuous), pd.qcut (quantile bins)
Handling outliers via quantile clipping / IQR
```

### pandas — Aggregation & Reshaping (this is "Querying & Retrieval")
```
.groupby('col').agg({'x':'mean','y':['sum','count']})
.groupby([...]).size() / .count() / .mean() / .transform()
.agg, .apply, .transform (know the difference)
.pivot_table(index=, columns=, values=, aggfunc=)
.pivot, .melt, .stack, .unstack
.merge(right, on=, how='inner'/'left'/'right'/'outer', left_on=, right_on=, suffixes=)
pd.concat([df1, df2], axis=0/1)
.join()
.sort_values(by=, ascending=), .sort_index(), .nlargest, .nsmallest
.rank()
.reset_index(), .set_index()
.crosstab (pd.crosstab)
```

### pandas — Dates & Windows (know at least the basics)
```
.dt accessor: .dt.year, .dt.month, .dt.day, .dt.dayofweek, .dt.hour
pd.date_range
.resample('M'/'D'/'W').agg(...)
.rolling(window=).mean(), .expanding(), .shift(), .diff(), .pct_change()
```

### Matplotlib / Seaborn — the six you must be fast at
```
import matplotlib.pyplot as plt
import seaborn as sns

Histogram:        plt.hist(x)  /  sns.histplot(data, x=)
Scatter:          plt.scatter(x, y)  /  sns.scatterplot(data, x=, y=, hue=)
Boxplot:          sns.boxplot(data, x=, y=)  (spotting outliers)
Bar chart:        plt.bar(...) / sns.barplot / .value_counts().plot(kind='bar')
Line chart:       plt.plot(x, y)
Correlation heatmap: sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

Also: plt.figure(figsize=), plt.title, plt.xlabel, plt.ylabel,
      plt.legend, plt.xticks(rotation=), plt.subplots(), ax=, plt.tight_layout, plt.show
sns.pairplot, sns.countplot, sns.lineplot
```

### scikit-learn — Preprocessing
```
from sklearn.model_selection import train_test_split
    train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
    scaler.fit_transform(X_train) / scaler.transform(X_test)  # fit on train ONLY
pd.get_dummies(df, columns=[...], drop_first=True)

from sklearn.impute import SimpleImputer   # strategy='mean'/'median'/'most_frequent'

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
```

### scikit-learn — Models
```
Regression:
  from sklearn.linear_model import LinearRegression, Ridge, Lasso
  from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
  from sklearn.tree import DecisionTreeRegressor

Classification:
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.svm import SVC
  from sklearn.naive_bayes import GaussianNB

Clustering (less likely but know it):
  from sklearn.cluster import KMeans

Common API for all: .fit(X_train, y_train), .predict(X), .predict_proba(X), .score()
```

### scikit-learn — Evaluation
```
Regression:
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  RMSE = mean_squared_error(y, pred, squared=False)  # or np.sqrt(mse)

Classification:
  from sklearn.metrics import accuracy_score, precision_score, recall_score,
       f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

Cross-validation:
  from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GridSearchCV
```

---

## Evaluation Metrics — What They Mean & When to Use Each

The cheat sheet above tells you *how to compute* each metric; this section tells you *which to pick and how to read it*. Know these conceptually — the auto-grader may ask for a specific one, and the human interviews will ask you to justify your choice.

### Regression metrics
| Metric | What it is | Units | Use when |
|--------|-----------|-------|----------|
| **RMSE** (root mean squared error) | √(mean of squared errors). Squaring punishes large errors heavily. | Same as target | Big misses are especially costly; the default for most regression tasks. |
| **MAE** (mean absolute error) | Mean of absolute errors; treats all errors linearly. | Same as target | You want robustness to outliers / a more forgiving error measure. |
| **R²** (coefficient of determination) | Proportion of variance in the target explained by the model. | Unitless | Communicating "how much better than the mean." 1.0 = perfect, 0 = no better than predicting the mean, **can be negative** for a bad model. |
| **MSE** | Mean of squared errors (RMSE before the square root). | Target² | Rarely reported directly (units are squared); mostly an intermediate. |

- **RMSE vs MAE** is the classic talking point: RMSE penalizes outliers more because of the squaring; MAE is more robust to them. If a few huge errors matter a lot, prefer RMSE.

### Classification metrics
| Metric | What it answers | Watch out for |
|--------|-----------------|---------------|
| **Accuracy** | Fraction of predictions correct. | **Misleading on imbalanced data** — 99% accuracy is trivial if 99% of rows are one class. |
| **Precision** | Of everything I predicted positive, how many actually were? (`TP / (TP + FP)`) | Optimize when **false positives are costly** (e.g. flagging a good customer as fraud). |
| **Recall** (sensitivity) | Of all the actual positives, how many did I catch? (`TP / (TP + FN)`) | Optimize when **false negatives are costly** (e.g. missing a disease, missing churn). |
| **F1 score** | Harmonic mean of precision and recall. | The **go-to single number for imbalanced classes** when you care about both precision and recall. |
| **Confusion matrix** | Raw TP / FP / FN / TN grid. | Everything above is derived from it — read it first to understand *how* the model is wrong. |
| **ROC-AUC** | How well the model **ranks** positives above negatives across all thresholds. 0.5 = random, 1.0 = perfect. | **Needs `predict_proba` (probabilities), not predicted labels** — the classic gotcha. Threshold-independent. |
| **PR-AUC** (precision-recall AUC) | Area under the precision-recall curve. | Preferred over ROC-AUC when positives are **rare** (severe imbalance). |
| **Log loss** | Penalizes confident wrong probability predictions. | When you're scored on probability quality, not just labels. |

- **The imbalance rule of thumb:** if classes are roughly balanced, accuracy is fine. If they're skewed, reach for precision/recall/F1 and ROC-AUC (or PR-AUC), not accuracy.
- **Threshold gotcha:** accuracy/precision/recall/F1 depend on the 0.5 decision threshold and on labels; ROC-AUC/PR-AUC use probabilities and are threshold-independent.

### Model-selection criteria (classical stats — likely *not* on a sklearn test)
| Metric | What it is | Note |
|--------|-----------|------|
| **AIC** (Akaike Information Criterion) | Balances model fit against complexity; **lower is better**. Used to compare candidate models. | Lives in **`statsmodels`, not scikit-learn** — sklearn's `LinearRegression` won't give you an AIC. |
| **BIC** (Bayesian Information Criterion) | Like AIC but penalizes complexity more strongly (bigger penalty as sample size grows). | Also statsmodels. Prefers simpler models than AIC. |
| **Adjusted R²** | R² penalized for the number of predictors. | Use when comparing models with different feature counts. |

> **On AIC/BIC:** understand the one-liner (lower = better fit-vs-complexity trade-off, used to *compare* models) in case it comes up in conversation, but don't drill it for a Python + sklearn CodeSignal test — they're a statsmodels / classical-regression concept and unlikely to appear. Spend your time on the precision/recall/F1/ROC-AUC cluster instead.

### One-line decision guide
- **Regression?** Report RMSE (+ R² for context); mention MAE if outliers matter.
- **Balanced classification?** Accuracy is fine, plus a confusion matrix.
- **Imbalanced classification?** F1 + ROC-AUC (or PR-AUC), never accuracy alone.
- **Comparing model complexity in classical stats?** AIC/BIC/adjusted R² (statsmodels).

---

### SQL-via-pandas (skim only — 20 min)
```
import sqlite3
conn = sqlite3.connect(':memory:')
df.to_sql('table', conn, index=False)
pd.read_sql("SELECT ... FROM table WHERE ... GROUP BY ...", conn)
```

---

## Questions You Should Be Able to Answer for EACH Dataset

Treat this as a repeatable checklist. For every dataset you practice with, run through all five phases. If you can do all of these fast and correctly, you're ready.

### Phase 1 — Load & Inspect
1. Load the file into a DataFrame; confirm shape (rows × columns).
2. Print dtypes. Which columns are wrong types (numbers stored as strings, dates as objects)?
3. How many missing values per column? What % of each column is missing?
4. How many duplicate rows are there?
5. Get summary statistics for numeric columns. For categorical columns, get value counts.
6. Which columns are numeric vs categorical vs datetime?

### Phase 2 — Clean & Preprocess
7. Drop or impute missing values (justify which columns get which treatment).
8. Fix incorrect dtypes (convert to numeric/datetime; coerce errors).
9. Clean string columns (strip whitespace, standardize case, fix inconsistent labels like "Male"/"male"/"M").
10. Remove duplicate rows.
11. Handle outliers (cap via IQR or quantiles, or flag them).
12. Encode categorical variables (one-hot for nominal, label/ordinal where appropriate).
13. Scale/normalize numeric features (and know *why* you fit the scaler on train only).
14. Create at least one new feature (e.g., extract year from a date; bin an age; ratio of two columns).

### Phase 3 — Query & Retrieve (the "SQL-in-pandas" muscle)
15. Filter rows meeting multiple conditions.
16. Group by a categorical column and compute mean/sum/count of a numeric column.
17. Group by two columns and aggregate multiple metrics at once.
18. Find the top N rows by some value (`nlargest`).
19. Build a pivot table (category × category → aggregated value).
20. Merge/join this dataset with a second table on a key.
21. Compute a group-wise statistic and broadcast it back to each row (`transform`).
22. Answer a "business" question: e.g., "Which category has the highest average X?" / "What's the churn rate by contract type?"

### Phase 4 — EDA & Visualization
23. Plot the distribution of the target variable (histogram or countplot).
24. Plot the distribution of 2–3 key features.
25. Boxplot a numeric feature grouped by a category (spot differences + outliers).
26. Scatter plot two numeric features; color by the target (`hue`).
27. Correlation heatmap of numeric features — which are most correlated with the target?
28. Bar chart of a categorical feature vs the mean target.
29. State 2–3 insights in plain English from the plots.

### Phase 5 — Model & Evaluate
30. Define X (features) and y (target). Handle categoricals so the model accepts them.
31. Train/test split (with `stratify` for classification).
32. Fit a baseline model (LinearRegression or LogisticRegression).
33. Fit a stronger model (RandomForest).
34. Predict on the test set.
35. **Regression:** report RMSE, MAE, R². **Classification:** report accuracy, precision, recall, F1, confusion matrix, ROC-AUC.
36. Which features are most important (`.feature_importances_` or coefficients)?
37. Run cross-validation and report the mean score.
38. (Stretch) Wrap preprocessing + model in a `Pipeline` / `ColumnTransformer`.
39. (Stretch) Tune one hyperparameter with `GridSearchCV`.

---

## Datasets to Practice With

Pick a mix so you cover both **classification** and **regression**, plus at least one "messy" dataset and one multi-table merge. All are free on Kaggle or built into sklearn/seaborn.

### Best all-rounders (do these first)
| Dataset | Task | Why | Where |
|---------|------|-----|-------|
| **Titanic** | Classification | The classic. Missing values, mixed types, feature engineering, clean sklearn pipeline. | Kaggle: "Titanic - Machine Learning from Disaster" |
| **House Prices: Advanced Regression** | Regression | Lots of columns, many with missing values — great cleaning + regression practice. | Kaggle: "House Prices - Advanced Regression Techniques" |
| **Telco Customer Churn** | Classification | Very "consulting-flavored" — churn rate by segment, business questions, categorical-heavy. | Kaggle: "Telco Customer Churn" (IBM) |
| **Adult / Census Income** | Classification | Predict income >50K. Categorical encoding, class imbalance. | Kaggle "Adult Census Income" / UCI |

### Built into libraries (instant, no download — great for quick timed drills)
| Dataset | Task | Load |
|---------|------|------|
| California Housing | Regression | `from sklearn.datasets import fetch_california_housing` |
| Breast Cancer | Classification | `from sklearn.datasets import load_breast_cancer` |
| Diabetes | Regression | `from sklearn.datasets import load_diabetes` |
| Iris | Classification (easy) | `from sklearn.datasets import load_iris` |
| Wine | Classification | `from sklearn.datasets import load_wine` |
| Tips | Regression/EDA | `sns.load_dataset('tips')` |
| Penguins | Classification/EDA | `sns.load_dataset('penguins')` — has NaNs, good cleaning practice |
| Flights | Time/EDA/pivot | `sns.load_dataset('flights')` — great for pivot_table & resample |

### Messy-data & merge practice
| Dataset | Use for | Where |
|---------|---------|-------|
| **NYC Airbnb Open Data** | Cleaning, outliers, groupby, geospatial-ish EDA | Kaggle: "New York City Airbnb Open Data" |
| **Superstore Sales** | Multi-table thinking, pivot tables, business queries | Kaggle: "Sample - Superstore" |
| **Netflix Movies and TV Shows** | String cleaning, date parsing, `.str` methods | Kaggle: "Netflix Movies and TV Shows" |
| **E-Commerce / Online Retail** | groupby, merge, datetime, revenue queries | Kaggle/UCI: "Online Retail" |

### Suggested rotation over the week
- **Day 2:** Titanic (untimed, learn the flow), then Penguins (timed, short).
- **Day 3:** Telco Churn (classification, timed) + California Housing (regression, timed).
- **Day 4 (work day):** NYC Airbnb — cleaning + groupby drills only, no modeling.
- **Day 5:** House Prices (regression, timed) + Adult Census (classification, timed) + revisit weakest phase.
- **Day 6:** Breast Cancer or Wine — one clean full timed run to finish confident.

---

## Final reminders

- **Fit scalers/encoders on training data only**, then transform test. Data leakage is a classic silent mistake.
- **Set `random_state`** everywhere so results are reproducible (graders may expect it).
- **Read the exact output requested** — a returned variable, a specific number rounded to N places, a DataFrame of a given shape. Give exactly that.
- **`errors='coerce'`** is your friend for messy numeric/date columns.
- **Watch the parentheses** in multi-condition filters: `df[(a) & (b)]`.
- If stuck, **move on and bank easier points** — come back if time allows.
- You may only search **official documentation for syntax** — not solutions. Your cheat sheet should mean you rarely need to.

Good luck — you've got the runway for this.
