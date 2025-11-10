# Technical Projects Portfolio

This repository showcases two Python-based analytics projects focused on quantitative modeling, probabilistic reasoning, and data-driven decision analysis.

---

## **1. Blackjack Outcome Analytics Engine**
**Tools:** Python, pandas, NumPy, matplotlib, seaborn  
**Dataset:** [900,000 Hands of Blackjack Results – Kaggle](https://www.kaggle.com/datasets/mojocolors/900000-hands-of-blackjack-results?resource=download)

**Summary**  
Analyzed ~900K blackjack hands to model probability distributions, outcome frequencies, and variance across player–dealer interactions.  
Implemented exploratory and regression analysis to model conditional win, loss, and bust rates, using heatmaps and linear fits to identify equilibrium thresholds consistent with game-theoretic optimal strategy and decision-making under uncertainty.  
Incorporated Bayesian weighting to iteratively update outcome probabilities from prior hand distributions, enhancing predictive precision under uncertainty.

**Key Features**
- Exploratory data analysis (EDA) and visualization of outcome frequencies  
- Regression modeling for conditional probabilities  
- Bayesian weighting to update expected values  
- Game-theoretic optimization and decision analysis framework  

**Files**
- `Blackjack Outcome Analytics Engine.py` — full project code  

---

## **2. Predictive House Valuation Model**
**Tools:** Python, pandas, scikit-learn, seaborn  
**Dataset:** [Predictive House Valuation Model Dataset (CSV)](./Predictive%20House%20Valuation%20Model.csv)

**Summary**  
Performed exploratory data analysis and correlation mapping to assess feature importance, mitigate multicollinearity, and train regression models with normalization, cross-validation, and error-metric evaluation to optimize predictive accuracy.

**Key Features**
- Feature engineering and correlation analysis  
- Regression modeling and performance evaluation  
- Normalization, cross-validation, and residual diagnostics  
- Visualization of predicted vs. actual property values  

**Files**
- `Predictive House Valuation Model.py` — model code  
- `Predictive House Valuation Model.csv` — dataset  

---

### **Author**
**Samuel Chen**  
New York University Leonard N. Stern School of Business  
📧 sc10793@stern.nyu.edu • [LinkedIn](https://linkedin.com/in/samuelchen2170) • [GitHub](https://github.com/schen2170)

---

### **Notes**
- The blackjack dataset is publicly available via Kaggle (linked above).  
- The housing dataset is provided locally for demonstration purposes and can be reproduced using the code in this repository.
