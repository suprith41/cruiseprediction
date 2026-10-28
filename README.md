# 🚢 Oceanic Cruise Passenger Survival Predictor

A modern, high-precision web frontend and predictive intelligence application built in Python for estimating cruise passenger survival and safety likelihoods based on historical voyage manifest data.

---

## 🌟 Key Features

1. **Interactive Web Form**:
   - **Passenger Identity**: Full Name, Courtesy Title (Auto, Mr, Mrs, Miss, Master, Rare/Officer/Royalty), Gender selector, and Age slider with live age tier badge (Infant, Child, Youth, Adult, Senior).
   - **Accommodations**: Cabin Class selection cards (1st Class Promenade Suite, 2nd Class Saloon, 3rd Class Steerage) with fare slider and embarkation port selector (Cherbourg, Queenstown, Southampton).
   - **Companionship Steppers**: Fine-grained counters for Siblings/Spouse and Parents/Children, calculating live party size and solo traveler indicators.

2. **1-Click Passenger Archetypes (Quick Presets)**:
   - *Lady Eleanor Vance* (1st Class VIP Patron)
   - *Arthur Pendelton* (3rd Class Young Tradesman)
   - *Master Leo Davies* (2nd Class Child with Family)
   - *Miss Clara Higgins* (2nd Class Solo Professional)
   - *Archibald Sterling* (1st Class Wealthy Merchant)

3. **Multi-Model Consensus & Explainability**:
   - **Primary Model**: Logistic Regression (~82% Cross-Validation accuracy).
   - **Ensemble Model**: Random Forest Classifier (~82% CV accuracy).
   - **Diagnostic Explanations**: Factor-by-factor breakdown explaining positive survival drivers and risk factors.
   - **Collectible Boarding Pass**: Dynamically generated retro-modern boarding pass with custom ticket number, accommodation status, and survival seal.

4. **Fleet Analytics Explorer**:
   - Real statistical survival distributions across cabin classes (63% in 1st vs 24% in 3rd), gender protocols (74% female vs 19% male), and age brackets.

---

## 🚀 How to Run

The application is completely self-contained using Python's standard library for the server and pre-installed `scikit-learn` / `pandas` / `numpy` for the machine learning engine:

```bash
python3 app.py 8000
```

Open your browser to:
👉 **[http://localhost:8000](http://localhost:8000)** (or `http://127.0.0.1:8000`)

---

## 📁 Project Architecture

```
cruise prediction/
├── app.py                # Multi-threaded Python HTTP server & REST API
├── model.py              # ML engine (Feature engineering, Logistic Regression & Random Forest)
├── cruise_data.csv       # Historical cruise voyage manifest dataset
├── index.html            # Responsive UI with glassmorphism design system
├── static/
│   ├── style.css         # Oceanic luxury CSS theme, animations & layouts
│   ├── app.js            # Client-side form handling, gauge animations & API integration
│   └── cruise_hero.jpg   # High-resolution cruise liner hero artwork
└── titanic-solution.ipynb# Original exploration notebook
```
