# Thesis Project

**Title**: Enhancing Emotion Recognition in Images and Videos for Underrepresented Groups Using Transformer

**Abstract**: Emotion recognition models struggle with class imbalance, demographic biases, and cultural sensitivity, leading to unfair performance across different groups. This study explores the potential of Transformer-based models, particularly Vision Transformer (ViT), to enhance both accuracy and fairness in Facial Emotion Recognition (FER). We introduce a fairness-aware ViT framework by integrating data augmentation, architectural modifications, and a fairness-aware CLS token weighting mechanism. Our optimized ViT model (ViT+Aug+T3+A3) achieves 82.72\% accuracy, which surpasses CNN and VGG while enhancing fairness across gender and race groups. However, challenges remain in addressing age-related disparities. To evaluate fairness, we employ Equalized Odds, Demographic Parity, and proposed a Balanced Fairness-Accuracy metric. Results are statistically validated using McNemar’s test. Despite these improvements, limitations such as dataset constraints and high temporal cost persist. Future research could focus on multimodal emotion recognition, efficiency-focused and fairness-aware techniques to further enhance equitable FER systems.

**Keywords**: Facial Emotion Recognition, Fairness, Vision Transformer, Balanced Fairness-Accuracy Score, Deep Learning

**Dataset**: Downloaded from [here](https://www.kaggle.com/datasets/hoanguyensgu/raf-db/data).

**File Description**:

`raf_labels.csv`: A CSV file containing ground truth labels (emotion, gender, race and age) for the RAF-DB dataset.

`01_...`, `02_...`,,, `05_...`: Contain scripts for training and evaluating various models (ViT, CNN, and VGG) using different techniques and optimization strategies.

`anaysis_notebooks`: Jupyter notebooks used for data analysis, model comparison, and visualization of results.

