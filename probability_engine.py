import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import os
import math

# Use Agg backend for matplotlib so it doesn't try to open windows in a background thread
matplotlib.use('Agg')

class ProbabilityEngine:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        
        # Prior Probabilities
        self.total_emails = 0
        self.spam_count = 0
        self.not_spam_count = 0
        self.p_spam = 0.0
        self.p_not_spam = 0.0
        
        # Word Probabilities
        self.spam_word_counts = {}
        self.not_spam_word_counts = {}
        self.total_spam_words = 0
        self.total_not_spam_words = 0
        self.vocabulary = set()
        
        # Common Stopwords (excluding spam-sensitive words like 'free', 'urgent', 'winner')
        self.stopwords = {
            'the', 'is', 'a', 'an', 'to', 'of', 'and', 'in', 'for', 'on', 'with', 
            'this', 'that', 'you', 'your', 'we', 'are', 'be', 'it', 'i', 'me', 'my',
            'at', 'by', 'from', 'as', 'but', 'or', 'so', 'if', 'then', 'else', 'when'
        }
        
        # Load and train immediately
        self.load_and_train()
        
    def clean_text(self, text):
        """
        Preprocesses text:
        - Convert to lowercase
        - Remove punctuation
        - Tokenize
        - Remove stopwords (while keeping meaningful words)
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation using regex
        text = re.sub(r'[^\w\s]', '', text)
        # Split into words (handles repeated spaces automatically)
        words = text.split()
        
        # Remove common stopwords, keep meaningful words
        cleaned_words = [w for w in words if w not in self.stopwords]
        
        return cleaned_words

    def load_and_train(self):
        """Loads dataset and calculates prior and conditional probabilities."""
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.total_emails = len(self.df)
            
            # Reset counts
            self.spam_word_counts = {}
            self.not_spam_word_counts = {}
            self.total_spam_words = 0
            self.total_not_spam_words = 0
            self.vocabulary = set()
            
            # Count spam and not_spam
            spam_emails = self.df[self.df['label'] == 'spam']
            not_spam_emails = self.df[self.df['label'] == 'not_spam']
            
            self.spam_count = len(spam_emails)
            self.not_spam_count = len(not_spam_emails)
            
            # Calculate Prior Probabilities
            self.p_spam = self.spam_count / self.total_emails if self.total_emails > 0 else 0.5
            self.p_not_spam = self.not_spam_count / self.total_emails if self.total_emails > 0 else 0.5
            
            # Calculate Conditional Probabilities (Multinomial approach)
            for text in spam_emails['email']:
                words = self.clean_text(text)
                for word in words:
                    self.vocabulary.add(word)
                    self.spam_word_counts[word] = self.spam_word_counts.get(word, 0) + 1
                    self.total_spam_words += 1
                    
            for text in not_spam_emails['email']:
                words = self.clean_text(text)
                for word in words:
                    self.vocabulary.add(word)
                    self.not_spam_word_counts[word] = self.not_spam_word_counts.get(word, 0) + 1
                    self.total_not_spam_words += 1
                    
        except Exception as e:
            print(f"Error loading dataset: {e}")
            
    def get_word_probabilities(self, word):
        """
        Calculates P(Word | Spam) and P(Word | Not Spam) with Laplace Smoothing.
        Laplace Smoothing prevents P = 0 if a word is never seen in a category.
        
        Formula: (count + 1) / (total_words_in_class + vocab_size)
        """
        word = word.lower()
        vocab_size = len(self.vocabulary)
        
        spam_containing_word = self.spam_word_counts.get(word, 0)
        not_spam_containing_word = self.not_spam_word_counts.get(word, 0)
        
        # P(Word | Spam)
        p_word_given_spam = (spam_containing_word + 1) / (self.total_spam_words + vocab_size)
        
        # P(Word | Not Spam)
        p_word_given_not_spam = (not_spam_containing_word + 1) / (self.total_not_spam_words + vocab_size)
        
        return p_word_given_spam, p_word_given_not_spam
        
    def analyze_email(self, email_text):
        """
        Uses Naive Bayes to classify an email.
        Uses Log Probabilities to prevent numerical underflow.
        """
        if not email_text or not email_text.strip():
            return {
                "classification": "UNKNOWN",
                "spam_probability": 0,
                "not_spam_probability": 0,
                "error": "Please enter an email to analyze."
            }

        words = self.clean_text(email_text)
        
        if not words:
            return {
                "classification": "UNKNOWN",
                "spam_probability": self.p_spam * 100,
                "not_spam_probability": self.p_not_spam * 100,
                "error": "No meaningful words were detected. Please enter a longer email.",
                "message": "Limited evidence was found; prediction is influenced mainly by the dataset prior."
            }
        
        # We use log probabilities to avoid underflow
        # log_score = log(P(Class)) + sum(log(P(Word | Class)))
        log_spam_score = math.log(self.p_spam)
        log_not_spam_score = math.log(self.p_not_spam)
        detected_words = []
        word_analysis = []
        bayes_calculation = {}
        
        for word in words:
            p_w_spam, p_w_not_spam = self.get_word_probabilities(word)
            
            # Add to log scores
            log_spam_score += math.log(p_w_spam)
            log_not_spam_score += math.log(p_w_not_spam)
            
            # Calculate standard probabilities for UI display
            # Law of Total Probability: P(Word) = P(Word|Spam)*P(Spam) + P(Word|Not Spam)*P(Not Spam)
            p_word = (p_w_spam * self.p_spam) + (p_w_not_spam * self.p_not_spam)
            
            # Bayes Theorem for this single word: P(Spam | Word) = P(Word | Spam)*P(Spam) / P(Word)
            p_spam_given_w = (p_w_spam * self.p_spam) / p_word
            
            # Only add each word once to the analysis table for cleaner UI
            if word not in detected_words:
                word_analysis.append({
                    "word": word,
                    "p_word_given_spam": p_w_spam,
                    "p_word_given_not_spam": p_w_not_spam,
                    "p_word": p_word,
                    "p_spam_given_word": p_spam_given_w
                })
                detected_words.append(word)
            
        # Normalize the final probabilities using the log-sum-exp trick to prevent overflow
        max_log = max(log_spam_score, log_not_spam_score)
        spam_score_exp = math.exp(log_spam_score - max_log)
        not_spam_score_exp = math.exp(log_not_spam_score - max_log)
        
        total_exp = spam_score_exp + not_spam_score_exp
        
        final_spam_prob = spam_score_exp / total_exp
        final_not_spam_prob = not_spam_score_exp / total_exp
            
        classification = "SPAM" if final_spam_prob >= 0.5 else "NOT SPAM"
        
        # Determine the "most spammy" word for the Bayes Calculation example in UI
        if word_analysis:
            # Sort by P(Spam | Word) descending
            word_analysis = sorted(word_analysis, key=lambda x: x['p_spam_given_word'], reverse=True)
            top_word = word_analysis[0]
            
            bayes_calculation = {
                "word": top_word['word'],
                "formula": f"P(Spam | '{top_word['word']}') = [P('{top_word['word']}' | Spam) × P(Spam)] / P('{top_word['word']}')",
                "substitution": f"= ({top_word['p_word_given_spam']:.4f} × {self.p_spam:.4f}) / {top_word['p_word']:.4f}",
                "result_decimal": top_word['p_spam_given_word'],
                "result_percentage": top_word['p_spam_given_word'] * 100
            }
            
        # Add a note if evidence was very limited
        message = ""
        if len(words) <= 2:
            message = "Limited evidence was found; prediction is influenced mainly by the dataset prior."
        
        return {
            "classification": classification,
            "spam_probability": final_spam_prob * 100,
            "not_spam_probability": final_not_spam_prob * 100,
            "detected_words": detected_words,
            "probabilities": word_analysis,
            "bayes_analysis": bayes_calculation,
            "log_spam_score": log_spam_score,
            "log_not_spam_score": log_not_spam_score,
            "message": message
        }

    def get_statistics(self):
        """Returns current dataset statistics."""
        return {
            "total_emails": self.total_emails,
            "spam_emails": self.spam_count,
            "not_spam_emails": self.not_spam_count,
            "spam_percentage": (self.spam_count / self.total_emails) * 100 if self.total_emails else 0,
            "not_spam_percentage": (self.not_spam_count / self.total_emails) * 100 if self.total_emails else 0,
            "vocabulary_size": len(self.vocabulary)
        }

    def generate_charts(self, output_dir="static/charts", current_words=None):
        """Generates charts for the dashboard using matplotlib."""
        # Use absolute paths for safety
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "spam-email-probability", "static", "charts")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. Spam vs Not Spam Distribution
        plt.figure(figsize=(6, 4))
        labels = ['Spam', 'Not Spam']
        sizes = [self.spam_count, self.not_spam_count]
        colors = ['#ff6b6b', '#4ecdc4']
        explode = (0.1, 0)
        
        if sum(sizes) > 0:
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, startangle=140)
            plt.title('Dataset Distribution (Prior Probabilities)')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'distribution.png'))
        plt.close()
        
        # 2. P(Word | Spam) vs P(Word | Not Spam)
        if current_words:
            # Use unique words from the current email (up to 10)
            words = list(dict.fromkeys(current_words))[:10]
            chart_title = 'Conditional Probabilities of Email Words'
        else:
            # Find the 10 most frequent words overall to compare their conditional probabilities
            word_freqs = {w: self.spam_word_counts.get(w, 0) + self.not_spam_word_counts.get(w, 0) for w in self.vocabulary}
            top_words = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
            words = [w[0] for w in top_words]
            chart_title = 'Conditional Probabilities of Top Words'
            
        if words:
            p_spam = [self.get_word_probabilities(w)[0] for w in words]
            p_not_spam = [self.get_word_probabilities(w)[1] for w in words]
            
            x = np.arange(len(words))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 5))
            rects1 = ax.bar(x - width/2, p_spam, width, label='P(Word | Spam)', color='#ff6b6b')
            rects2 = ax.bar(x + width/2, p_not_spam, width, label='P(Word | Not Spam)', color='#4ecdc4')
            
            ax.set_ylabel('Probability')
            ax.set_title(chart_title)
            ax.set_xticks(x)
            ax.set_xticklabels(words, rotation=45)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'word_conditionals.png'))
            plt.close()
            
        return ["distribution.png", "word_conditionals.png"]
