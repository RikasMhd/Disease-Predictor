# 🩺 Symptom-Based Disease Prediction
 
A machine learning project that predicts the most likely disease from 3 symptoms,
with a confidence score and the top 3 possibilities.
 
> ⚠️ **Disclaimer:** Educational project on a small dataset. Not a medical diagnostic tool.
 
## Demo
- Live app: https://disease-predictor-dbnvbtijjwoxfducadjsfy.streamlit.app/
- Screenshot: ![image1](image.png)  ![image2](image-1.png)
 
## Dataset
- 82 patient records, 9 diseases: Parkinson's, Diabetes, Lupus, Cardiovascular Disease,
  Alzheimer's, Migraine, Stroke, General Anxiety, Meningitis
- Columns: age, weight, blood pressure, glucose, insulin, smoking habit, 3 symptoms, disease
- Only the 3 symptoms are used as features at the moment
 
## Approach
1. Data cleaning: fixed a duplicate "Parkinson's" label caused by a Unicode apostrophe (’ vs ')
2. Combined the 3 symptoms into one text string
3. Vectorized with `CountVectorizer(binary=True)`
4. Compared models using 5-fold stratified cross-validation:
 
| Model | CV Accuracy |
|---|---|
| Naive Bayes | 95.00% |
| Logistic Regression | 96.25% |
| Random Forest | 95.07% |
 
5. Final model: Logistic Regression (CV accuracy 96.25% ± 3.06, test accuracy 95.24% on 21 rows)
 
The differences between models are about one sample, so results on a dataset this
small should be read with caution.
 
## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
 
## Files
- `disease.ipynb` – data cleaning, model comparison, training, prediction
- `app.py` – Streamlit web app
- `medical_dataset.csv` – dataset
- `disease_model.pkl` – saved model (created by running the notebook)
 
## Future improvements
- Use age, glucose, insulin, blood pressure and smoking habit as extra features
- Larger, real-world dataset
- Better handling of symptom synonyms (e.g. "blurry vision" / "blurred vision")
 