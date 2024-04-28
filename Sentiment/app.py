from flask import Flask, render_template, request
import re
import numpy as np
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

app= Flask(__name__)

global file_name, status
status = 0
accuracies = []
accur = {}
# Importing the dataset :
DATASET_COLUMNS=['target','text']
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('Data1.csv',
                 encoding=DATASET_ENCODING)
df.columns = DATASET_COLUMNS

# Display of the first 5 lines :
df.sample(5)

df.info()



# Replacing the values to ease understanding :
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['target'] = le.fit_transform(df['target'])
print(df['target'])

# Selecting the text and Target column for our further analysis :
data = df[['text','target']]
data



# Making statement text in lower case :
data['text'] = data['text'].str.lower()
data['text'].tail()


def preprocessing(text):
    if isinstance(text, str):  # Check if text is a string
        # Convert text to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'(www.[^s]+)|(https?://[^s]+)', ' ', text)

        # Remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join(word for word in text.split() if word not in stop_words)

        # Remove repeating characters
        text = re.sub(r'(.)\1+', r'\1', text)

        # Tokenize text
        tokens = word_tokenize(text)

        # Apply stemming
        st = nltk.PorterStemmer()
        tokens = [st.stem(word) for word in tokens]

        # Apply lemmatization
        lm = nltk.WordNetLemmatizer()
        tokens = [lm.lemmatize(word) for word in tokens]

        return ' '.join(tokens)
    else:
        return ''  # Return empty string for non-string values

# Example usage:
# Assuming dataset is your DataFrame containing 'text' column
data['processed_text'] = data['text'].apply(preprocessing)
print(data['processed_text'].head())
# Assuming dataset_test is your DataFrame containing 'text' column


 
# Separating input feature and label :
print(data.columns)

X = data.processed_text
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorizing text data
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB()
}

for name, model in models.items():
    if name == 'Naive Bayes':
        # Convert sparse matrices to dense arrays
        model.fit(X_train_transformed.toarray(), y_train)
        train_preds = model.predict(X_train_transformed.toarray())
        test_preds = model.predict(X_test_transformed.toarray())
    else:
        model.fit(X_train_transformed, y_train)
        train_preds = model.predict(X_train_transformed)
        test_preds = model.predict(X_test_transformed)

    train_conf_matrix = confusion_matrix(y_train, train_preds)
    test_conf_matrix = confusion_matrix(y_test, test_preds)

    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    accuracies.append((name, train_accuracy, test_accuracy))
    accur[name] = (train_accuracy, test_accuracy)

    print(f"Model Name : {name}")
    print(f"Train Accuracy : {train_accuracy}")
    print(f"Test Accuracy : {test_accuracy}")
    print("Confusion Matrix (Train):")
    print(train_conf_matrix)
    print("Confusion Matrix (Test):")
    print(test_conf_matrix)
    print()

test_accuracies = [accur[model][1] for model in models]
print("Accuracies : ",accuracies)
print("Accuracy :",accur)
# Find the model with the highest test accuracy
best_model_name = max(accur, key=lambda k: accur[k][0])
best_model_train_accuracy = accur[best_model_name][0]
best_model_test_accuracy = accur[best_model_name][1]
print("Best Model Name : ",best_model_name)
print(best_model_train_accuracy)
print(best_model_test_accuracy)
best_model = models[best_model_name]
filename = 'best_train_model.pkl'
pickle.dump(best_model, open(filename, 'wb'))
plt.figure(figsize=(10, 6))
plt.bar(models.keys(), test_accuracies, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Test Accuracy')
plt.title('Comparison of Test Accuracies of Different Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot as an image file
plt.savefig('static/model.png')
# List to store training accuracies
train_accuracies = [accur[model][0] for model in models]

plt.figure(figsize=(10, 6))
plt.bar(models.keys(), train_accuracies, color='lightgreen')
plt.xlabel('Models')
plt.ylabel('Training Accuracy')
plt.title('Comparison of Training Accuracies of Different Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot as an image file
plt.savefig('static/training_model.png')


@app.route('/', methods=['GET', 'POST'])
def single():
    if request.method == 'POST':
        filename = 'best_train_model.pkl'
        model = pickle.load(open(filename, 'rb'))
        st1 = request.form['single_comm']
        st = preprocessing(st1)
        st = vectorizer.transform([st])
        pred = model.predict(st)
        print(pred)

        if pred==0:
            predi = 'Negative'
        elif pred==1:
            predi = 'Neutral'
        elif pred==2:
            predi = "Positive"
        else:
            predi = None

        from textblob import TextBlob

        # Perform sentiment analysis
        analysis = TextBlob(st1)

        # Get polarity (sentiment) score
        polarity = analysis.sentiment.polarity

        # Get subjectivity score
        subjectivity = analysis.sentiment.subjectivity

        # Take the absolute value of the polarity score
        polarity = abs(polarity)
        # Confidence level associated with the sentiment prediction made by the sentiment analysis model
        conf = polarity * 100
            
        print("Polarity:", polarity * 100)
        print("Subjectivity:", subjectivity * 100)
        return render_template('index.html', pred=predi, conf=conf, acc=best_model_train_accuracy*100, bmn=best_model_name)
    else:
        # Handle other HTTP methods
        return render_template('index.html', predi=None, conf=0, acc=best_model_train_accuracy*100, bmn=best_model_name)

@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

'''
@app.route('')
def accuracy():
    return render_template('about.html')
'''
if __name__ == '__main__':
    app.run(debug=True)