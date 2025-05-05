# Statistical Learning 2025 - Assignment 3

**Group Project for the course: Statistical Learning (2025)**  
Leiden University

## 🎯 Project Overview

In this assignment, we explore a self-chosen dataset and apply at least **three different statistical learning techniques** to investigate meaningful research questions. The assignment focuses on the full analysis pipeline: from data exploration to methodology selection, application, and comparison of techniques.

This project culminates in a **20-minute group presentation**, highlighting key insights and justifying the analytical choices.

## 👨‍👩‍👧 Group Members

| Full Name          | Student ID  | GitHub Username |
|--------------------|-------------|------------------|
| Noah Bos           | s2400393    | bosnbos          |
| Andre Blokland     | sxxxxxxx    | andreblokland1   |
| Carla F            | sxxxxxxx    | xxxxxxxx         |



## 📊 Dataset

- **Name**: Spambase  
- **Source**: [UCI ML Repository - Spambase](https://archive.ics.uci.edu/dataset/94/spambase)  
- **Description**:  
  This dataset contains 4,601 labeled emails with 57 numeric features describing content-based characteristics (e.g. frequency of specific words, capital letters, symbols). The goal is to predict whether an email is **spam (1)** or **not spam (0)**.

- **Format**: CSV / ARFF
- **Target variable**: `spam` (binary)

- **Relevance**: Ideal for supervised learning, especially binary classification and feature importance analysis.

## ❓ Research Question(s)

- *RQ1: What factors are associated with ...?*  
- *RQ2: Can we predict ... based on ...?*  
- *RQ3: Are there clusters or segments within ...?*

## 🧪 Techniques Used

We apply the following statistical learning methods:

1. **Exploratory Data Analysis (EDA)**  
   - Summary statistics, missing values, outlier detection
   - Visualizations: histograms, boxplots, scatterplots

2. **[Method 1: e.g., Logistic Regression]**  
   - Justification: *why it fits your question*
   - Results & interpretation

3. **[Method 2: e.g., Classification Tree / Random Forest]**  
   - Description
   - Comparison with previous method

4. **[Optional Method 3+: e.g., Clustering (K-means / Hierarchical)]**  
   - Used for identifying groups or patterns

> You can include additional methods not covered in class if used correctly.


## 📁 Project Structure

```text
project/
├── data/                   # Raw and cleaned dataset files
├── notebooks/              # Jupyter notebooks for analysis
├── presentation/           # Final slides (PDF, source)
├── README.md               # Project overview
└── requirements.txt        # Python dependencies (optional)
