# Food Appeal Classification Model 🍲✨

A deep-learning image classification system (MobileNetV2 / TensorFlow) developed to sort food photos as “appetizing” vs. “unappetizing” based on human ratings.

## 🚀 Engineering Highlights
* **Bias Diagnosis & Debugging:** Discovered the model was “cheating” by keying on professional studio lighting and composition rather than the food itself.
* **Hard-Negative Mining:** Engineered a hard-negative dataset (introducing ruined, dropped, and poorly-lit food examples) to sharpen the decision boundary and improve real-world accuracy.
* **Architecture:** Implemented a **Siamese Network** to compare image pairs, achieving over 90% validation accuracy after data curation and augmentation.

## 📂 Project Structure
* `Apetizing.py`: Core training script (Siamese Network construction, Data Augmentation, and Model Training).
* `confusionmatrix.py`: Evaluation tool for model diagnostics and performance analysis.
* `longkong.py`: Inference utility for batch-processing test sets.
* `testquestionairs.py`: Visualization tool for side-by-side model predictions vs. ground truth.

## 🛠️ Tech Stack
* **Python** (TensorFlow/Keras, Pandas, NumPy, Scikit-learn)
* **Pre-trained Model:** MobileNetV2

---
*The trained model is saved as `apetizing_model_test_uni.keras`.*
