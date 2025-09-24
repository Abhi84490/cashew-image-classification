Cashew Grading and Classification System
This project is a machine learning-based system for the automated classification of cashew nuts into different grades. The core task is to accurately identify a cashew's grade from its image, addressing the challenge of distinguishing between 23 distinct classes. This work is part of a larger dissertation and represents the initial development and evaluation phase.

Project Overview
The system processes cashew images and classifies them using a variety of models. The current work focuses on comparative analysis to determine the most effective model for this multi-class classification problem. The primary goals are to:

Develop and implement several machine learning models for image classification.

Evaluate the performance of each model on a 23-class cashew dataset.

Analyze key performance metrics such as accuracy, precision, recall, and F1-score to identify the strengths and weaknesses of each approach.

Key Components & Results
The project explores several classification techniques, with the following key findings:

Random Forest: Achieved a high overall accuracy of 80%. The model performed well across most classes, with a macro average F1-score of 0.77 and a weighted average F1-score of 0.80.

Support Vector Machine (SVM): This model also showed strong performance, with an accuracy of 80%. Its balanced performance is reflected in its macro average F1-score of 0.77 and weighted average F1-score of 0.80.

Multilayer Perceptron (MLP): The neural network approach yielded a competitive accuracy of 78%.

K-Nearest Neighbors (KNN): The KNN model achieved an accuracy of 73%.

Naive Bayes: This model, while simpler, had a lower accuracy of 69%.

Ensemble Model: A custom ensemble of the best-performing models (Random Forest, SVM, and MLP) was created to boost the accuracy to 83%.

Future Work
The next steps for this project will involve:

Advanced Feature Engineering: Exploring deep learning-based embedding techniques to capture more complex visual features.

Model Optimization: Fine-tuning the hyperparameters of the top-performing models (Random Forest, SVM, MLP) to further improve accuracy.

Real-time Classification: Developing a system capable of classifying cashew grades in a real-time environment.

Robustness Testing: Evaluating the models' performance on new, unseen data to ensure generalizability and robustness.

Deployment: Building a user interface for practical application in industrial settings.
