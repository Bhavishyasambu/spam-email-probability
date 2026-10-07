from flask import Flask, render_template, request, redirect, url_for, flash
from probability_engine import ProbabilityEngine
import os
import time
import csv

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flash_messages'

# Absolute path for robust execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "data", "emails.csv")
charts_dir = os.path.join(BASE_DIR, "static", "charts")

# Ensure directories exist
os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
os.makedirs(charts_dir, exist_ok=True)

# Initialize probability engine
engine = ProbabilityEngine(dataset_path)

# Generate charts on startup
engine.generate_charts(output_dir=charts_dir)

@app.route("/", methods=["GET"])
def index():
    """Render the main frontend dashboard with current statistics."""
    stats = engine.get_statistics()
    # Regenerate global charts so it doesn't show the last analyzed email
    engine.generate_charts(output_dir=charts_dir)
    timestamp = int(time.time())
    return render_template("index.html", stats=stats, timestamp=timestamp)

@app.route("/analyze", methods=["POST"])
def analyze():
    """Endpoint to analyze an email via HTML form submission."""
    email_text = request.form.get("email", "")
    
    if not email_text.strip():
        flash("Please enter an email to analyze.", "error")
        return redirect(url_for("index"))
        
    # Analyze the email
    result = engine.analyze_email(email_text)
    
    if "error" in result and result.get("classification") == "UNKNOWN":
        flash(result["error"], "error")
        return redirect(url_for("index"))
        
    # Add the analyzed email to the dataset to simulate learning
    label = "spam" if result["classification"] == "SPAM" else "not_spam"
    with open(dataset_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([email_text.replace('\n', ' '), label])
        
    # Retrain the engine with the new data and update charts
    engine.load_and_train()
    engine.generate_charts(
        output_dir=charts_dir, 
        current_words=result.get("detected_words"),
        current_email=email_text
    )
        
    # Get current statistics and timestamp for UI
    stats = engine.get_statistics()
    timestamp = int(time.time())
    
    return render_template("index.html", 
                           stats=stats, 
                           analysis=result, 
                           current_email=email_text,
                           timestamp=timestamp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
