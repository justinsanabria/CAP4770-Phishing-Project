import re
import time
from collections import defaultdict, Counter
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def map_function(email):
    # Tokenize the email text into words
    words = re.findall(r'\b\w+\b', email.lower())
    word_counts = Counter(words)
    return word_counts

def process_file(file_path, email_column='body', chunk_size=1000):
    print(f"Processing file: {file_path}")
    all_word_counts = []
    labels = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        print(f"Processing chunk of size {chunk.shape}")
        # Fill NaN values in the email column with an empty string
        chunk[email_column] = chunk[email_column].fillna('')
        for email in chunk[email_column]:
            word_counts = map_function(email)
            all_word_counts.append(word_counts)
        labels.extend(chunk['label'].tolist())
    return all_word_counts, labels

def main():
    print("Starting main function")
    start_time = time.time()

    # Base directory for the data file
    # IMPORTANT TO CHANGE THIS TO WHATEVER YOURE PATH IS:
    base_dir = '/Users/andresportillo/Documents/CAP4770-Phishing-Project/data/'

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
    labels = []
    for file_path in csv_files:
        file_word_counts, file_labels = process_file(file_path, email_column=email_column)
        all_word_counts.extend(file_word_counts)
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

if __name__ == "__main__":
    main()
