# Sentiment-Analysis-using-LSTM

This project aims to analyze the sentiment of tweets using Natural Language Processing (NLP) techniques and a deep learning model built with TensorFlow/Keras.

## Steps:

1. **Data Preparation:**
    - Import necessary libraries: pandas, numpy, matplotlib, seaborn, re, sklearn (train_test_split, TfidfVectorizer, LabelEncoder, metrics), TensorFlow/Keras (callbacks, preprocessing, models, layers, regularizers), spacy, nltk (stopwords, WordNetLemmatizer), string, and warnings.
    - Load training and validation datasets from CSV files (`twitter_training.csv`, `twitter_validation.csv`).
    - Explore the data using `head()`, `shape`, `info()`, `duplicated()`, and `isnull()`.
    - Add headers to the dataframes.
    - Clean the data by handling missing values and removing duplicates.
    - Drop unnecessary columns ('Header1', 'company').

2. **Preprocessing:**
    - Download NLTK resources (stopwords, wordnet).
    - Define a preprocessing function (`preprocess_test`) to:
        - Remove numbers.
        - Remove punctuation and convert text to lowercase.
        - Perform lemmatization and remove stop words.
    - Apply the preprocessing function to the 'text' column of both training and validation datasets, creating new columns 'train_text' and 'test_text'.

3. **Data Splitting and Encoding:**
    - Split the data into training and validation sets (texts and labels).
    - Encode the labels using LabelEncoder.

4. **Tokenization and Padding:**
    - Initialize a Tokenizer and fit it on the training texts.
    - Convert texts to sequences using the tokenizer.
    - Pad the sequences to a uniform length using `pad_sequences`.

5. **Model Building:**
    - Create a sequential model with:
        - An embedding layer.
        - Two bidirectional LSTM layers with dropout and batch normalization.
        - A dense layer with ReLU activation and dropout.
        - An output layer with softmax activation for multi-class classification.
    - Compile the model with 'sparse_categorical_crossentropy' loss, 'adam' optimizer, and 'accuracy' metric.

6. **Model Training:**
    - Define an EarlyStopping callback to prevent overfitting.
    - Train the model using `fit()` with training and validation data, specifying epochs and batch size.

7. **Model Evaluation:**
    - Evaluate the model on the test data using `evaluate()`.
    - Print the test loss and accuracy.

8. **Visualization:**
    - Plot the training and validation accuracy and loss curves using matplotlib.


## Requirements:

- Python 3
- Libraries: pandas, numpy, matplotlib, seaborn, re, sklearn, TensorFlow/Keras, spacy, nltk, string.

## Usage:

1. Ensure you have the necessary libraries installed.
2. Place the training and validation datasets (`twitter_training.csv`, `twitter_validation.csv`) in the same directory as the notebook.
3. Run the notebook cells sequentially to execute the steps outlined above.

## Notes:

- The model architecture and hyperparameters can be further tuned for improved performance.
- Additional features can be engineered from the text data for better sentiment analysis.
- The model can be saved and loaded for future use.
