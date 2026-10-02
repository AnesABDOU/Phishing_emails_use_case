# Phishing URL Detection

This project tries to find phishing URLs. Each URL is labeled as good (0) or bad (1).
I test 4 models and then mix them together to get a better result.

## Data

- `training_data.csv`: URLs with a label (`good` or `bad`)
- `test_data_to_send.csv`: URLs without a label, used for the final predictions

About 11% of the URLs are phishing, so the data is not balanced. I use `class_weight='balanced'` to handle this.

The data files are not in this repo.

## What the notebook does

1. Quick look at the data (size, missing values, label balance, URL length)
2. Clustering with TF-IDF + PCA + KMeans to look for patterns in good URLs
3. Machine learning models with TF-IDF on characters (3 to 5 letters):
   - Random Forest (with GridSearch)
   - Logistic Regression
4. Deep learning models on tokenized URLs:
   - CNN
   - Bidirectional LSTM
5. Comparison of the models (classification report, confusion matrix, ROC curve)
6. A mixed model: `0.6 * LR + 0.3 * CNN + 0.1 * LSTM`
7. Predictions on the test file, saved as CSV

## Results

Scores on the phishing class (20% of the training data kept for testing):

| Model               | Precision | Recall | F1   |
|---------------------|-----------|--------|------|
| Random Forest       | 0.92      | 0.87   | 0.89 |
| Logistic Regression | 0.81      | 0.94   | 0.87 |
| CNN                 | 0.95      | 0.88   | 0.91 |
| LSTM                | 0.97      | 0.83   | 0.90 |
| **Mixed model**     | **0.94**  | **0.93** | **0.94** |

The CNN is the best single model. The mixed model gives the best balance: it misses fewer phishing URLs and blocks fewer good ones.

Logistic Regression is very fast to train (about 30 seconds) and still gives good results. Random Forest is the slowest, mostly because of the GridSearch.

## How to run

The notebook was made in Google Colab.

1. Open `Email_phishing.ipynb` in Colab
2. Upload `training_data.csv` and `test_data_to_send.csv` when asked
3. Run the cells in order

Main libraries: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`

## Next steps

- Add more features from the URLs (length, special characters, IP address, etc.)
- Make the CNN deeper and train it for more epochs
- Test more ways to mix the models
