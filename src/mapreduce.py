import re
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def map_function(email):
    # Tokenize the email text into words
    words = re.findall(r'\b\w+\b', email.lower())
    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1
    return word_counts

def reduce_function(word, counts):
    return sum(counts)

def process_file(file_path, email_column='body'):
    # Load dataset
    data = pd.read_csv(file_path)

    # Check if the specified column exists
    if email_column not in data.columns:
        print(f"Column '{email_column}' not found in {file_path}")
        return defaultdict(int), []

    # Fill NaN values in the email column with an empty string
    data[email_column] = data[email_column].fillna('')

    # Apply map function to each email
    mapped_results = []
    for email in data[email_column]:
        word_counts = map_function(email)
        mapped_results.append(word_counts)

    # Combine all word counts from mapped results
    combined_counts = defaultdict(list)
    for word_counts in mapped_results:
        for word, count in word_counts.items():
            combined_counts[word].append(count)

    return combined_counts, data['label'].tolist()

def main():
    # List of CSV files to process
    csv_files = [
        'data/CEAS_08.csv',
        'data/Enron.csv',
        'data/Ling.csv',
        'data/Nazario.csv',
        'data/Nigerian_Fraud.csv',
        'data/SpamAssasin.csv'
    ]

    # Specify the correct column name for email text
    email_column = 'body'

    # Process each file and combine results
    overall_combined_counts = defaultdict(list)
    labels = []
    for file_path in csv_files:
        file_combined_counts, file_labels = process_file(file_path, email_column=email_column)
        for word, counts in file_combined_counts.items():
            overall_combined_counts[word].extend(counts)
        labels.extend(file_labels)

    # Apply reduce function to overall combined counts
    reduced_results = {word: reduce_function(word, counts) for word, counts in overall_combined_counts.items()}

    # Prepare data for classifier model
    X = pd.DataFrame([reduced_results])
    y = pd.Series(labels)

    # Ensure X and y have the same number of samples
    if X.shape[0] != y.shape[0]:
        print("Mismatch between number of samples in features and labels.")
        return

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the classifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
