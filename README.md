# 💰 AI Finance Tracker

A simple web app that tracks expenses and automatically categorizes them using Machine Learning.

---

## 🚀 Features

* Add expenses with description and amount
* Automatic category prediction
* Real-time total calculation
* Clean UI with Tailwind CSS

---

## 🧠 How it Works

The app uses a basic ML pipeline:

* **TF-IDF Vectorizer** → converts text into features
* **Naive Bayes Classifier** → predicts category

Example:

```
Input: "auto fare 200"
Output: "Transport"
```

---

## 🛠️ Tech Stack

* Frontend: HTML, Tailwind CSS, JavaScript
* Backend: Flask (Python)
* ML: scikit-learn

---

## 📁 Project Structure

```
AI-Finance-Tracker/
│
├── app.py
└── templates/
    └── index.html
```

---

## ⚙️ Installation & Run

1. Install dependencies:

```
pip install flask scikit-learn
```

2. Run the app:

```
python app.py
```

3. Open in browser:

```
http://127.0.0.1:5000/
```

---

## ⚠️ Limitations

* Small dataset (may misclassify some inputs)
* Basic ML model (not highly accurate)
