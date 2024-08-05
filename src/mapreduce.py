import re
import time
from collections import Counter
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import text2emotion as te

def map_function(email):
    # Tokenize the email text into words
    words = re.findall(r'\b\w+\b', email.lower())
    word_counts = Counter(words)
    return word_counts

def process_file(file_path, email_column='body', chunk_size=1000):
    print(f"Processing file: {file_path}")
    all_word_counts = []
    all_emotion_scores = []
    labels = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        print(f"Processing chunk of size {chunk.shape}")
        # Fill NaN values in the email column with an empty string
        chunk[email_column] = chunk[email_column].fillna('')
        for email in chunk[email_column]:
            word_counts = map_function(email)
            emotion_scores = te.get_emotion(email)
            all_word_counts.append(word_counts)
            all_emotion_scores.append(emotion_scores)
        labels.extend(chunk['label'].tolist())
    return all_word_counts, all_emotion_scores, labels

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():
    print("Starting main function")
    start_time = time.time()

    # Base directory for the data file
    # IMPORTANT TO CHANGE THIS TO WHATEVER YOURE PATH IS:
    base_dir = '/Users/bernardo/Programming/Git/UF-CS-Classwork/CAP4770 - Intro to Data Science/CAP4770-Phishing-Project/Data/'

    # List of CSV files to process
    csv_files = [
        base_dir + 'CEAS_08.csv',
        base_dir + 'Enron.csv',
        base_dir + 'Ling.csv',
        base_dir + 'Nazario.csv',
        base_dir + 'Nigerian_Fraud.csv',
        base_dir + 'SpamAssasin.csv'
    ]

    # Specify the correct column name for email text
    email_column = 'body'

    # Process each file and combine results
    all_word_counts = []
    all_emotion_scores = []
    labels = []
    for file_path in csv_files:
        file_word_counts, file_emotion_scores, file_labels = process_file(file_path, email_column=email_column)
        all_word_counts.extend(file_word_counts)
        all_emotion_scores.extend(file_emotion_scores)
        labels.extend(file_labels)

    print("Finished processing files")
    print(f"Time taken to process files: {time.time() - start_time} seconds")

    # Get a list of all unique words across all emails
    word_freq = Counter(word for word_counts in all_word_counts for word in word_counts.keys())

    # Limit vocabulary to the top 10,000 most frequent words
    top_words = [word for word, _ in word_freq.most_common(10000)]
    word_index = {word: i for i, word in enumerate(top_words)}
    print(f"Unique words count: {len(word_index)}")

    # Create a sparse matrix where each row corresponds to an email and each column to a word
    row_ind = []
    col_ind = []
    data = []
    for i, word_counts in enumerate(all_word_counts):
        for word, count in word_counts.items():
            if word in word_index:
                row_ind.append(i)
                col_ind.append(word_index[word])
                data.append(count)

    X = csr_matrix((data, (row_ind, col_ind)), shape=(len(all_word_counts), len(word_index)))
    print(f"Feature matrix shape: {X.shape}")
    print(f"Time taken to create feature matrix: {time.time() - start_time} seconds")

    y = pd.Series(labels)
    print(f"Labels length: {len(y)}")

    # Ensure X and y have the same number of samples
    if X.shape[0] != y.shape[0]:
        print("Mismatch between number of samples in features and labels.")
        return

    # Split the data
    print("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split completed")

    # Initialize and train the classifier model
    print("Training the model")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed")

    # Make predictions and evaluate the model
    print("Making predictions")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))
    print(f"Total time taken: {time.time() - start_time} seconds")

    # Now do a second model with emotion scores
    # Create a sparse matrix where each row corresponds to an email and each column to a word
    row_ind_emotion = []
    col_ind_emotion = []
    data_emotion = []
    emotion_index = {'Angry': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4}
    for i, emotions in enumerate(all_emotion_scores):
        for emotion in emotions:
            row_ind_emotion.append(i)
            col_ind_emotion.append(emotion_index[emotion])
            data_emotion.append(emotions[emotion])

    X_emotion = csr_matrix((data_emotion, (row_ind_emotion, col_ind_emotion)), shape=(len(all_emotion_scores), len(emotion_index)))
    print(f"Feature matrix shape: {X_emotion.shape}")
    print(f"Time taken to create feature matrix: {time.time() - start_time} seconds")

    y_emotion = pd.Series(labels)
    print(f"Labels length: {len(y_emotion)}")

    # Ensure X and y have the same number of samples
    if X_emotion.shape[0] != y_emotion.shape[0]:
        print("Mismatch between number of samples in features and labels.")
        return

    # Split the data
    print("Splitting data into train and test sets")
    X_emotion_train, X_emotion_test, y_emotion_train, y_emotion_test = train_test_split(X_emotion, y_emotion, test_size=0.2, random_state=42)
    print("Data split completed")

    # Initialize and train the classifier model
    print("Training the model")
    model_emotion = RandomForestClassifier(n_estimators=100, random_state=42)
    model_emotion.fit(X_emotion_train, y_emotion_train)
    print("Model training completed")

    # Make predictions and evaluate the model
    print("Making predictions")
    y_emotion_pred = model_emotion.predict(X_emotion_test)
    accuracy = accuracy_score(y_emotion_test, y_emotion_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_emotion_test, y_emotion_pred))
    print(f"Total time taken: {time.time() - start_time} seconds")

    # Plotting the confusion matrix
    cm_emotion = confusion_matrix(y_emotion_test, y_emotion_pred)
    plt.figure()
    plot_confusion_matrix(cm_emotion, classes=['Actual', 'Phishing'], title='Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
