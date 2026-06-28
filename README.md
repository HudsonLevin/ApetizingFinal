# Food Appeal Classification Model 🍲✨

A deep-learning image classification system using a **Siamese Network architecture** with **MobileNetV2** as the feature extractor. This model classifies food image pairs based on human preference ("Appetizing" vs. "Unappetizing").

## 🚀 Key Features

* **Siamese Network Architecture:** Designed to compare two images simultaneously to predict which one is more appetizing, leveraging the power of Transfer Learning.
* **Bias Correction:** During development, the model faced a "data bias" where it prioritized studio lighting over food features. This was resolved by engineering a hard-negative dataset (ruined/poorly lit food), resulting in a more robust decision boundary.
* **Production-Ready Modular Code:** Organized into functional modules for maintainability:
    * `Apetizing.py`: Core training script (Siamese Network construction, Data Augmentation, and Model Training).
    * `confusionmatrix.py`: Evaluation tool for model diagnostics and performance analysis.
    * `longkong.py`: Inference utility for batch-processing test sets.
    * `testquestionairs.py`: Visualization tool for side-by-side model predictions vs. ground truth.

## 📂 Project Structure

* `Apetizing.py`: Siamese Network setup, data ingestion via `ImageDataGenerator`, and training loop with `ReduceLROnPlateau` and `EarlyStopping`.
* `confusionmatrix.py`: Generates confusion matrices to visualize classification accuracy using `seaborn` and `scikit-learn`.
* `longkong.py`: Handles batch prediction on external test sets and exports results to CSV.
* `testquestionairs.py`: A visualization script to display the model’s choices side-by-side with actual human labels for qualitative analysis.

## 🛠️ Technologies Used

* **Python 3.x**
* **TensorFlow / Keras** (MobileNetV2 feature extractor)
* **Scikit-learn** (Model evaluation and metrics)
* **Pandas & Numpy** (Data processing and augmentation)
* **Matplotlib & Seaborn** (Data visualization)

## 💡 Engineering Highlights
This project demonstrates end-to-end data engineering skills—from handling unbalanced datasets (scraping/curating images to fix data imbalance) to optimizing a Siamese network to reach over 90% validation accuracy.
