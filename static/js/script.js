let currentAnalyzedEmail = "";

// Fetch and display initial statistics when page loads
document.addEventListener("DOMContentLoaded", () => {
    fetchStatistics();
});

function fetchStatistics() {
    fetch('/api/statistics')
        .then(response => response.json())
        .then(data => {
            document.getElementById('statTotal').innerText = data.total_emails;
            document.getElementById('statSpam').innerText = data.spam_emails;
            document.getElementById('statNotSpam').innerText = data.not_spam_emails;
            
            document.getElementById('priorSpam').innerText = `${data.spam_percentage.toFixed(2)}%`;
            document.getElementById('priorNotSpam').innerText = `${data.not_spam_percentage.toFixed(2)}%`;
        })
        .catch(err => console.error("Error fetching stats:", err));
}

function analyzeEmail() {
    const emailInput = document.getElementById('emailInput').value;
    const errorMsg = document.getElementById('errorMessage');
    const loading = document.getElementById('loadingIndicator');
    const resultsContainer = document.getElementById('resultsContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');

    currentAnalyzedEmail = emailInput;

    // Reset UI
    errorMsg.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    
    if (!emailInput.trim()) {
        errorMsg.innerText = "Please enter an email message to analyze.";
        errorMsg.classList.remove('hidden');
        return;
    }

    // UI Loading state
    analyzeBtn.disabled = true;
    loading.classList.remove('hidden');

    // API Request
    fetch('/api/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email: emailInput })
    })
    .then(response => response.json())
    .then(data => {
        analyzeBtn.disabled = false;
        loading.classList.add('hidden');
        
        if (data.error) {
            errorMsg.innerText = data.error;
            errorMsg.classList.remove('hidden');
            return;
        }

        displayResults(data);
    })
    .catch(error => {
        analyzeBtn.disabled = false;
        loading.classList.add('hidden');
        errorMsg.innerText = "An error occurred while connecting to the probability engine.";
        errorMsg.classList.remove('hidden');
        console.error("Error:", error);
    });
}

function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.classList.remove('hidden');
    
    // Reset confirmation message
    const confirmMessage = document.getElementById('confirmMessage');
    if (confirmMessage) {
        confirmMessage.classList.add('hidden');
        confirmMessage.innerText = '';
    }

    // 1. Classification Badge
    const finalResult = document.getElementById('finalResult');
    finalResult.innerText = data.classification;
    if (data.classification === "SPAM") {
        finalResult.className = "classification-badge badge-spam";
    } else {
        finalResult.className = "classification-badge badge-not-spam";
    }

    // 2. Probability Bars
    const spamBar = document.getElementById('spamBar');
    const notSpamBar = document.getElementById('notSpamBar');
    const spamPercent = document.getElementById('spamPercent');
    const notSpamPercent = document.getElementById('notSpamPercent');

    // Adding small delay for smooth animation
    setTimeout(() => {
        spamBar.style.width = `${data.spam_probability}%`;
        notSpamBar.style.width = `${data.not_spam_probability}%`;
    }, 100);
    
    spamPercent.innerText = `${data.spam_probability.toFixed(2)}%`;
    notSpamPercent.innerText = `${data.not_spam_probability.toFixed(2)}%`;

    // 3. Detected Words Tags
    const wordList = document.getElementById('detectedWordsList');
    wordList.innerHTML = '';
    
    if (data.detected_words.length === 0) {
        wordList.innerHTML = '<p class="text-muted">No familiar words detected in vocabulary.</p>';
    } else {
        data.detected_words.forEach(word => {
            const span = document.createElement('span');
            span.className = 'tag';
            span.innerText = word;
            wordList.appendChild(span);
        });
    }

    // 4. Probability Table
    const tbody = document.getElementById('probabilityTableBody');
    tbody.innerHTML = '';
    
    data.probabilities.forEach(item => {
        const tr = document.createElement('tr');
        
        // Highlight high probability words
        if(item.p_spam_given_word > 0.5) {
            tr.style.backgroundColor = 'rgba(239, 35, 60, 0.05)';
        }

        tr.innerHTML = `
            <td><strong>${item.word}</strong></td>
            <td>${(item.p_word_given_spam * 100).toFixed(2)}%</td>
            <td>${(item.p_word_given_not_spam * 100).toFixed(2)}%</td>
            <td>${(item.p_word * 100).toFixed(2)}%</td>
            <td><strong>${(item.p_spam_given_word * 100).toFixed(2)}%</strong></td>
        `;
        tbody.appendChild(tr);
    });

    // 5. Bayes Calculation Explanation
    const bayesCard = document.querySelector('.bayes-card');
    if (data.bayes_analysis && data.bayes_analysis.word) {
        bayesCard.classList.remove('hidden');
        document.getElementById('bayesWordTitle').innerText = `For word: '${data.bayes_analysis.word}'`;
        
        document.getElementById('bayesSubstitution').innerHTML = `
            <p class="eq">= <span class="fraction">
                <span class="top">${data.bayes_analysis.substitution.split(' / ')[0].replace('= (', '(')}</span>
                <span class="bottom">${data.bayes_analysis.substitution.split(' / ')[1]}</span>
            </span></p>
        `;
        
        document.getElementById('bayesFinalResult').innerHTML = `
            = ${data.bayes_analysis.result_decimal.toFixed(4)} = <strong>${data.bayes_analysis.result_percentage.toFixed(2)}%</strong>
        `;
    } else {
        bayesCard.classList.add('hidden');
    }

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

function confirmEmail(label) {
    if (!currentAnalyzedEmail) return;
    
    const confirmMessage = document.getElementById('confirmMessage');
    confirmMessage.classList.remove('hidden');
    confirmMessage.innerText = "Adding to dataset and recalculating probabilities...";
    confirmMessage.style.backgroundColor = "#e2eafc";
    confirmMessage.style.color = "var(--primary)";
    
    fetch('/api/add-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: currentAnalyzedEmail, label: label })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            confirmMessage.innerText = data.error;
            confirmMessage.style.backgroundColor = "#ffe5e5";
            confirmMessage.style.color = "var(--spam)";
        } else {
            confirmMessage.innerText = "Email successfully added to dataset. Dataset Updated!";
            confirmMessage.style.backgroundColor = "#e6fffa";
            confirmMessage.style.color = "var(--not-spam)";
            
            // Reload the charts by adding a timestamp query string
            const chartImg = document.querySelector('.chart-container img');
            if (chartImg) {
                chartImg.src = `/static/charts/distribution.png?t=${new Date().getTime()}`;
            }
            
            // Fetch updated statistics
            fetchStatistics();
        }
    })
    .catch(err => {
        confirmMessage.innerText = "An error occurred.";
        confirmMessage.style.backgroundColor = "#ffe5e5";
        confirmMessage.style.color = "var(--spam)";
        console.error(err);
    });
}
