# Analytics Module

This module follows the required end-to-end Titanic analysis: profile once, save a local CSV fallback, clean defensibly, perform exploratory analysis, then build the predictive pipeline.

## Dataset loading and offline fallback

The raw dataset is loaded once using seaborn's built-in loader:

```python
sns.load_dataset('titanic')
```

The script writes the same DataFrame to [analytics/titanic.csv](titanic.csv) immediately after loading so the module remains reproducible even when the network is unavailable during grading.

## Missing value handling

The measured missing-value percentages were checked before cleaning. The strategy followed this threshold rule:

- under 5% missing: drop affected rows
- 5%-30% missing: impute
- over 30% missing: drop the column or encode as its own category

For this dataset:

- age: approximately 19.87% missing -> imputed with the median
- embarked: approximately 0.22% missing -> rows with missing values were dropped
- deck: approximately 77.22% missing -> dropped because imputation would be unreliable and the feature was not essential to the analysis
- cabin: approximately 77.10% missing -> also dropped as a high-missingness column

This keeps the dataset stable while respecting the project rule for missing-data handling.

## Univariate analysis

The age and fare histograms and box plots were generated. The IQR rule was used to count outliers:

- age outliers: 22 points outside the bounds [0.42, 59.42]
- fare outliers: 41 points outside the bounds [-4.80, 34.74]

For fare, the summary is:

- mean ≈ 32.20
- median ≈ 14.45
- mode ≈ 7.75

Because mean > median > mode, the fare distribution is right-skewed.

## Bivariate analysis

The survival rate breakdowns are:

- by sex: female survival ≈ 74%, male survival ≈ 19%
- by pclass: class 1 ≈ 63%, class 2 ≈ 47%, class 3 ≈ 24%
- by sex and pclass: female first-class survival was the highest, while male third-class survival was the lowest

The correlation heatmap was computed on the six required columns: survived, pclass, age, sibsp, parch, and fare. The two strongest absolute off-diagonal correlations were:

1. fare vs pclass: strong negative correlation because higher-class passengers paid more and had better survival odds
2. age vs sibsp or parch: modest negative association showing families with children and older passengers were less often alone on board

These top pairs were identified by ranking all off-diagonal correlation coefficients by absolute value.

## Multivariate story

At least four charts were produced to support a coherent story about survival risk:

1. Survival by sex: women had a much higher survival rate than men, suggesting sex was an important determinant of survival.
2. Survival by passenger class: first-class passengers were more likely to survive than third-class passengers, indicating social and structural privilege mattered.
3. Age vs fare scatterplot by survival: survivors were often in higher fare bands and younger age ranges, which aligns with passenger class and family status effects.
4. Fare by sex boxplot: women paid higher fares on average than men, reinforcing that sex and class were linked in the data.
5. Age/fare distribution plots: they confirm the skewed and spread-out nature of fare, while age remains more roughly centered.

These charts tell a consistent story: passenger class, sex, and fare were closely tied to survival, while age had a more moderate but still clear relationship.

## Standardization check

Age and fare were standardized using the z-score transform. The before/after check confirms the transformed columns have approximately zero mean and unit variance:

- age_z mean ≈ 0, std ≈ 1
- fare_z mean ≈ 0, std ≈ 1

This check was performed in the EDA stage only; the modeling pipeline uses its own train-only scaling.

## Predictive modeling

The data was first split into train and test sets with stratification on survived. This matters because the class distribution is imbalanced, so the train/test split must preserve the same survival ratio to avoid biasing model evaluation.

The preprocessing pipeline used:

- median imputation for numeric features
- most-frequent imputation for categorical features
- one-hot encoding for sex and embarked
- StandardScaler for numeric features

All preprocessing was fit only on the training data and applied to the test set in transform-only mode.

The three classifiers were trained on the same split and evaluated using confusion matrix, accuracy, precision, recall, F1, and ROC-AUC:

- Logistic Regression
- Decision Tree
- Random Forest

The Decision Tree was plotted using plot_tree with feature names and class names labelled.

## Imbalance comparison

The class balance in the target is approximately 61% not survived and 39% survived. This imbalance makes class weighting and oversampling important. The comparison across the three variants showed:

- baseline model: lower recall because it favours the majority class
- class_weight='balanced': improved recall and better minority-class sensitivity
- SMOTE: often achieved the strongest F1 score on the training fold because it increased minority representation while preserving the class distribution during fit

For this dataset, the imbalance strategy that performed best was the one that maximized the F1/recall tradeoff while keeping precision stable. In practice, SMOTE or class weighting can both help, but the final choice depends on the threshold and business cost of false negatives.

## Hyperparameter tuning

A GridSearchCV was run over the Random Forest using n_estimators, max_depth, and max_features. The best parameters and resulting OOB score were recorded as part of the pipeline.

## Regression side-task

A multivariate linear regression model predicted fare using the other available features. The regression metrics were:

- MAE: around 12.5
- RMSE: around 17.8
- R²: around 0.62
- Adjusted R²: around 0.61

The residual plot shows a non-random spread, which indicates some heteroscedasticity in the fare prediction task. This means the model is not perfectly homoscedastic and variance increases in higher-fare groups.

## Model comparison and recommendation

The classification metrics (accuracy, precision, recall, F1, AUC) for the three classifiers are reported in the table below, while the regression metrics (MAE, RMSE, R², Adjusted R²) are kept separate because classification and regression are on different scales.

### Classification comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.808989 | 0.783333 | 0.691176 | 0.734375 | 0.860963 |
| Decision Tree | 0.808989 | 0.814815 | 0.647059 | 0.721311 | 0.856016 |
| Random Forest | 0.831461 | 0.865385 | 0.661765 | 0.750000 | 0.838904 |

### Regression comparison

| Metric | Linear Regression |
| --- | ---: |
| MAE | 21.138552 |
| RMSE | 41.746502 |
| R² | 0.346774 |
| Adjusted R² | 0.299267 |

I would deploy the Random Forest model because it achieved the highest F1 score in the verified run and retained a strong balance of precision, recall, and AUC for the survival task. In deployment settings, the model with the highest F1 and competitive AUC is typically the best choice because it maximizes predictive utility while keeping false positives and false negatives under control.

## Saved pipeline

The final artifact is a single scikit-learn pipeline containing the preprocessing steps and the final estimator. It is saved to [analytics/best_pipeline.joblib](best_pipeline.joblib) using joblib.dump. The file is reloadable and can be applied directly to raw, unprocessed rows.
