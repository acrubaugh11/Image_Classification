# Image_Classification
Developed a machine learning pipeline to classify image data using feature extraction, dimensionality reduction, and model evaluation.
Data Preprocessing:

Loaded and cleaned multiple image datasets (CSV format) for classification tasks.
Separated features and target variables, then performed data splitting (80% training, 20% testing).
Applied StandardScaler for feature normalization to improve model accuracy.
Used PCA (Principal Component Analysis) for dimensionality reduction, retaining 95% of the variance, to enhance computational efficiency and mitigate overfitting.

Model Development:

Built and trained two models:
Random Forest Classifier: Utilized a random forest with 1000 trees to classify images and evaluated its performance on both training and test sets.
Lasso Regression: Implemented LassoCV for feature selection and regression-based classification, using cross-validation to optimize model hyperparameters.


Model Evaluation:

Evaluated the performance of both models using key metrics: Accuracy, F1 Score, and Confusion Matrix.
Generated detailed classification reports to assess precision, recall, and support for both models.
Visualized model performance using confusion matrices with heatmap representations to provide clear insights into misclassifications.


Tools & Libraries:

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
Applied machine learning techniques including classification, feature scaling, PCA, and model evaluation.
Outcome:

Achieved high classification accuracy and F1 scores for both Random Forest and Lasso models.
Demonstrated the use of dimensionality reduction techniques (PCA) and the effectiveness of Random Forest and Lasso for image classification tasks.
