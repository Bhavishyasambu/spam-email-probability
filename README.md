# Probability-Based Spam Email Detection and Analysis

## Project Objective
The objective of this college project is to analyze an UNKNOWN email and determine whether it is likely to be SPAM or NOT SPAM. The system uses mathematical probability concepts, specifically conditional probability, total probability, and Bayes' theorem, to make its predictions. The user does not know the answer beforehand; they enter an unknown email, and the system acts as a real-world spam filter to predict its classification.

## Problem Statement
In the modern digital age, distinguishing between legitimate emails and spam is critical. This project demonstrates how fundamental statistical and probability theories can be applied to solve the real-world problem of text classification.

## Technologies Used
* **Python**: Core logic and probability engine.
* **Flask**: Web framework to connect the backend Python engine to the frontend HTML.
* **Pandas**: To read and process the historical dataset (`emails.csv`).
* **NumPy**: Used for data manipulation in charting.
* **Matplotlib**: To generate dynamic probability distribution charts.
* **HTML/CSS**: For the frontend user interface.
* **CSV**: To store the historical labeled dataset.
*(Note: No JavaScript or external machine learning libraries like scikit-learn are used. All mathematics are implemented from scratch).*

## Project Architecture
1. **HTML Frontend**: A simple form where the user submits an unknown email.
2. **Flask Backend**: Receives the POST request and passes the text to the engine.
3. **Probability Engine**: Processes the text, calculates log-probabilities based on the historical dataset, and determines the final prediction.
4. **Pandas Dataset**: The engine reads the `emails.csv` reference dataset to gather initial counts and frequencies.
5. **Matplotlib**: Generates visual representations of the probability distributions.
6. **Results**: The calculated probabilities and final prediction are rendered back to the user via Jinja2 templates.

## Dataset Explanation
The system relies on a historical dataset located at `data/emails.csv`. This dataset acts as the "reference memory" or ground truth for the engine. It contains emails labeled as either `spam` or `not_spam`. The engine dynamically counts the total emails and word frequencies in this file every time the server starts or the dashboard loads. The system **never** automatically adds its own predictions back into this dataset, ensuring the dataset remains a pure, verified source of truth.

## Probability Concepts Used

### Prior Probabilities
* **P(Spam)**: The probability that any random email is spam based on the dataset (Spam Emails / Total Emails).
* **P(Not Spam)**: The probability that any random email is not spam (Not Spam Emails / Total Emails).

### Conditional Probability
* **P(Word | Spam)**: The probability of a specific word appearing, given that the email is already known to be spam.
* **P(Word | Not Spam)**: The probability of a specific word appearing, given that the email is not spam.

### Law of Total Probability
To find the absolute probability of a word appearing in any email:
`P(Word) = [P(Word | Spam) × P(Spam)] + [P(Word | Not Spam) × P(Not Spam)]`

### Bayes' Theorem
Used to reverse the conditional probability. If we know a word appears, what is the probability the email is spam?
`P(Spam | Word) = [P(Word | Spam) × P(Spam)] / P(Word)`

### Laplace Smoothing
If a user enters a word that doesn't exist in our dataset, its probability would be 0, which would break the entire multiplication chain. Laplace Smoothing prevents this by adding a small constant (1) to the numerator and the total vocabulary size to the denominator:
`P(Word | Class) = (Word Count in Class + 1) / (Total Words in Class + Vocabulary Size)`
This ensures unknown words are handled gracefully without automatically leaning too heavily toward spam.

### Naive Bayes-style Calculation & Log Probabilities
To calculate the probability for an entire sentence of multiple words, we use a Naive Bayes approach. Instead of just multiplying probabilities (which can cause numerical underflow and result in 0 for long emails), we add the **logarithms** of the probabilities:
`Log Spam Score = log(P(Spam)) + sum(log(P(Word | Spam)))`
The scores are then converted back into normalized percentages (using the log-sum-exp trick) to give the final P(Spam | Email) and P(Not Spam | Email).

## Data Preprocessing
Before math is applied, the text is cleaned:
1. Converted to lowercase.
2. Punctuation is removed.
3. Tokenized into individual words.
4. Common stopwords (like 'the', 'is', 'and') are ignored to prevent them from skewing the results, while keeping meaningful indicators.

## How to Install
1. Ensure Python 3.x is installed.
2. Clone this repository.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to: `http://127.0.0.1:5000`

## Example Usage
**Input**: `"Congratulations! You won a free prize. Claim your reward now."`
**Output**: 
- High Spam Probability (e.g., 98.5%)
- Detected words: `congratulations`, `won`, `free`, `prize`, `claim`, `reward`
- Prediction: **SPAM**

**Input**: `"Please submit your assignment tomorrow."`
**Output**:
- High Not Spam Probability (e.g., 95.2%)
- Prediction: **NOT SPAM**

## Limitations
- **Naive Assumption**: The model assumes all words are completely independent of each other, which is not true in real human language (e.g., "credit" and "card" appear together often).
- **Dataset Size**: The accuracy of the model is entirely dependent on the size and quality of `emails.csv`. A small dataset means the model hasn't learned enough vocabulary.

## Future Improvements
- Implement a supervised feedback loop where a verified administrator can review predictions and manually add them to the dataset to improve accuracy over time.
- Implement N-grams (looking at pairs of words instead of single words) to capture context.
