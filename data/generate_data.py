
import pandas as pd
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, n_classes=2, random_state=42)

# Create a DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

# Create a synthetic text dataset
text_data = []
for index, row in df.iterrows():
    text = ""
    if row['target'] == 0:
        text += "This is a sample sentence about category A. "
        text += " ".join([f'word{i}' for i, val in enumerate(row[:-1]) if val > 0])
    else:
        text += "This is a different kind of text for category B. "
        text += " ".join([f'word{i}' for i, val in enumerate(row[:-1]) if val < 0])
    text_data.append(text)

text_df = pd.DataFrame(text_data, columns=['text'])
text_df['category'] = y

# Save the dataset to a CSV file
text_df.to_csv(r"c:\Users\2185206\naive_bayes\data\synthetic_text_data.csv", index=False)

print("Synthetic text dataset generated and saved to data/synthetic_text_data.csv")

