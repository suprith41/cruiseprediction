"""
Cruise Passenger Survival Prediction Engine
Trains model on historical cruise voyage dataset and provides prediction inference
with feature engineering and explainability factors.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

DATA_PATH = os.path.join(os.path.dirname(__file__), 'cruise_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cruise_models.joblib')

FEATURE_COLUMNS = [
    'Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone',
    'Sex_male',
    'Embarked_Q', 'Embarked_S',
    'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare'
]

def extract_title(name):
    """Extract and standardize courtesy title from passenger name."""
    if not isinstance(name, str):
        return 'Mr'
    try:
        parts = name.split(',')
        if len(parts) > 1:
            title = parts[1].split('.')[0].strip()
        else:
            title = name.strip()
    except Exception:
        title = 'Mr'

    rare_titles = [
        'Lady', 'Countess', 'the Countess', 'Capt', 'Col', 'Don', 'Dr',
        'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
    ]
    if title in rare_titles:
        return 'Rare'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Mme':
        return 'Mrs'
    elif title in ['Mr', 'Mrs', 'Miss', 'Master']:
        return title
    else:
        return 'Rare'

class CruiseModelEngine:
    def __init__(self):
        self.lr_model = None
        self.rf_model = None
        self.medians = {}
        self.lr_cv_score = 0.0
        self.rf_cv_score = 0.0
        self.stats = {}
        self.is_ready = False

    def train_or_load(self):
        """Train models from cruise_data.csv or load existing cache."""
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Cruise dataset not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)
        
        # Calculate dataset statistics for insights tab
        self._calculate_fleet_stats(df)

        # Imputation references
        age_median = float(df['Age'].median())
        fare_median = float(df['Fare'].median())
        embarked_mode = str(df['Embarked'].mode()[0])
        
        self.medians = {
            'age': age_median,
            'fare': fare_median,
            'embarked': embarked_mode,
            'fare_by_class': {
                1: float(df[df['Pclass'] == 1]['Fare'].median()),
                2: float(df[df['Pclass'] == 2]['Fare'].median()),
                3: float(df[df['Pclass'] == 3]['Fare'].median())
            }
        }

        # Feature engineering on training set
        train_df = df.copy()
        train_df['Title'] = train_df['Name'].apply(extract_title)
        train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
        train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
        train_df['Age'] = train_df['Age'].fillna(age_median)
        train_df['Fare'] = train_df['Fare'].fillna(fare_median)
        train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)

        X = self._build_feature_matrix(train_df)
        y = train_df['Survived']

        # Logistic Regression (matching notebook)
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_scores = cross_val_score(self.lr_model, X, y, cv=5, scoring='accuracy')
        self.lr_cv_score = round(float(lr_scores.mean()) * 100, 2)
        self.lr_model.fit(X, y)

        # Random Forest (ensemble verification)
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        rf_scores = cross_val_score(self.rf_model, X, y, cv=5, scoring='accuracy')
        self.rf_cv_score = round(float(rf_scores.mean()) * 100, 2)
        self.rf_model.fit(X, y)

        self.is_ready = True
        return self

    def _build_feature_matrix(self, df):
        """Construct deterministic feature DataFrame conforming to FEATURE_COLUMNS."""
        matrix = pd.DataFrame(index=df.index)
        matrix['Pclass'] = df['Pclass'].astype(int)
        matrix['Age'] = df['Age'].astype(float)
        matrix['Fare'] = df['Fare'].astype(float)
        matrix['FamilySize'] = df['FamilySize'].astype(int)
        matrix['IsAlone'] = df['IsAlone'].astype(int)
        
        # Sex one-hot (baseline: female)
        matrix['Sex_male'] = (df['Sex'] == 'male').astype(int)
        
        # Embarked one-hot (baseline: C - Cherbourg)
        matrix['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
        matrix['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
        
        # Title one-hot (baseline: Master)
        matrix['Title_Miss'] = (df['Title'] == 'Miss').astype(int)
        matrix['Title_Mr'] = (df['Title'] == 'Mr').astype(int)
        matrix['Title_Mrs'] = (df['Title'] == 'Mrs').astype(int)
        matrix['Title_Rare'] = (df['Title'] == 'Rare').astype(int)
        
        return matrix[FEATURE_COLUMNS]

    def _calculate_fleet_stats(self, df):
        """Precompute fleet survival patterns for data explorer tab."""
        total = len(df)
        survived = int(df['Survived'].sum())
        overall_rate = round(float(survived / total * 100), 1)

        # Survival by Class
        by_class = {}
        for c in [1, 2, 3]:
            sub = df[df['Pclass'] == c]
            rate = round(float(sub['Survived'].mean() * 100), 1) if len(sub) > 0 else 0
            by_class[str(c)] = {'total': len(sub), 'survived': int(sub['Survived'].sum()), 'rate': rate}

        # Survival by Gender
        by_gender = {}
        for g in ['female', 'male']:
            sub = df[df['Sex'] == g]
            rate = round(float(sub['Survived'].mean() * 100), 1) if len(sub) > 0 else 0
            by_gender[g] = {'total': len(sub), 'survived': int(sub['Survived'].sum()), 'rate': rate}

        # Survival by Age Group
        age_bins = [0, 12, 18, 50, 100]
        age_labels = ['Children (0-12)', 'Youth (13-18)', 'Adults (19-50)', 'Seniors (51+)']
        df_age = df.dropna(subset=['Age']).copy()
        df_age['AgeGroup'] = pd.cut(df_age['Age'], bins=age_bins, labels=age_labels)
        by_age = {}
        for label in age_labels:
            sub = df_age[df_age['AgeGroup'] == label]
            rate = round(float(sub['Survived'].mean() * 100), 1) if len(sub) > 0 else 0
            by_age[label] = {'total': len(sub), 'rate': rate}

        # Survival by Embarkation Port
        port_names = {'C': 'Cherbourg (FR)', 'Q': 'Queenstown (IE)', 'S': 'Southampton (UK)'}
        by_port = {}
        for p, name in port_names.items():
            sub = df[df['Embarked'] == p]
            rate = round(float(sub['Survived'].mean() * 100), 1) if len(sub) > 0 else 0
            by_port[name] = {'total': len(sub), 'rate': rate}

        self.stats = {
            'total_passengers': total,
            'total_survived': survived,
            'overall_rate': overall_rate,
            'by_class': by_class,
            'by_gender': by_gender,
            'by_age': by_age,
            'by_port': by_port
        }

    def predict_passenger(self, data):
        """
        Evaluate a single passenger profile and return survival probabilities,
        decisions, and diagnostic explanation factors.
        """
        if not self.is_ready:
            self.train_or_load()

        name = data.get('name', 'Anonymous Passenger').strip()
        sex = data.get('sex', 'male').lower().strip()
        pclass = int(data.get('pclass', 3))
        
        # Courtesy title
        given_title = data.get('title', '').strip()
        if not given_title or given_title == 'Auto':
            title = extract_title(f"{name}")
            if title == 'Mr' and sex == 'female':
                title = 'Miss'
        else:
            title = given_title

        # Numeric fields with fallbacks
        try:
            age = float(data.get('age', self.medians['age']))
        except (ValueError, TypeError):
            age = self.medians['age']

        try:
            sibsp = max(0, int(data.get('sibsp', 0)))
        except (ValueError, TypeError):
            sibsp = 0

        try:
            parch = max(0, int(data.get('parch', 0)))
        except (ValueError, TypeError):
            parch = 0

        try:
            fare = float(data.get('fare', self.medians['fare_by_class'].get(pclass, 15.0)))
        except (ValueError, TypeError):
            fare = self.medians['fare_by_class'].get(pclass, 15.0)

        embarked = data.get('embarked', 'S').upper()
        if embarked not in ['C', 'Q', 'S']:
            embarked = 'S'

        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        # Construct single-row DataFrame
        single_df = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'Fare': fare,
            'Embarked': embarked,
            'FamilySize': family_size,
            'IsAlone': is_alone,
            'Title': title
        }])

        X_single = self._build_feature_matrix(single_df)

        # Logistic Regression predictions
        lr_prob = float(self.lr_model.predict_proba(X_single)[0][1])
        lr_pred = int(self.lr_model.predict(X_single)[0])

        # Random Forest predictions
        rf_prob = float(self.rf_model.predict_proba(X_single)[0][1])
        rf_pred = int(self.rf_model.predict(X_single)[0])

        # Blended survival probability (weighted: 60% LR, 40% RF)
        blended_prob = (lr_prob * 0.6) + (rf_prob * 0.4)
        survived = bool(blended_prob >= 0.5)
        survival_pct = round(blended_prob * 100, 1)

        # Risk Tier
        if survival_pct >= 70.0:
            risk_tier = "High Survival Probability"
            status_theme = "safe" # Emerald / Cyan
        elif survival_pct >= 45.0:
            risk_tier = "Moderate Survival Probability"
            status_theme = "warning" # Amber
        else:
            risk_tier = "High Mortality Risk"
            status_theme = "danger" # Coral / Red

        # Diagnostic Explanation Factors
        factors = self._generate_diagnostic_factors(pclass, sex, age, title, family_size, is_alone, fare, embarked)

        # Generate boarding ticket number
        ticket_code = f"CRUISE-{pclass}{embarked}-{hash(name) % 9000 + 1000}"

        return {
            'passenger': {
                'name': name,
                'title': title,
                'sex': sex.capitalize(),
                'age': age,
                'pclass': pclass,
                'pclass_label': {1: '1st Class (Upper Deck)', 2: '2nd Class (Middle Cabin)', 3: '3rd Class (Lower Berths)'}.get(pclass),
                'fare': round(fare, 2),
                'embarked': embarked,
                'embarked_city': {'C': 'Cherbourg, France', 'Q': 'Queenstown, Ireland', 'S': 'Southampton, England'}.get(embarked),
                'family_size': family_size,
                'is_alone': bool(is_alone),
                'ticket_code': ticket_code
            },
            'prediction': {
                'survived': survived,
                'probability_percent': survival_pct,
                'risk_tier': risk_tier,
                'status_theme': status_theme,
                'models': {
                    'logistic_regression': {
                        'name': 'Logistic Regression (Primary Notebook Model)',
                        'probability': round(lr_prob * 100, 1),
                        'prediction': bool(lr_pred),
                        'accuracy': f"{self.lr_cv_score}% CV"
                    },
                    'random_forest': {
                        'name': 'Random Forest Classifier (Ensemble)',
                        'probability': round(rf_prob * 100, 1),
                        'prediction': bool(rf_pred),
                        'accuracy': f"{self.rf_cv_score}% CV"
                    }
                }
            },
            'factors': factors
        }

    def _generate_diagnostic_factors(self, pclass, sex, age, title, family_size, is_alone, fare, embarked):
        """Analyze individual passenger attributes to produce transparent explanations."""
        factors = []

        # 1. Gender & Protocol
        if sex == 'female':
            factors.append({
                'category': 'Evacuation Protocol',
                'impact': 'positive',
                'title': 'Maritime Protocol Priority',
                'description': 'Maritime emergency guidelines gave women and children earliest lifeboat access.'
            })
        else:
            if age > 14:
                factors.append({
                    'category': 'Evacuation Protocol',
                    'impact': 'negative',
                    'title': 'Strict Boarding Deferral',
                    'description': 'Adult male passengers faced strict boarding deferrals until women and children were accommodated.'
                })

        # 2. Cabin Class & Proximity
        if pclass == 1:
            factors.append({
                'category': 'Accommodations',
                'impact': 'positive',
                'title': 'Promenade Deck Proximity',
                'description': '1st Class suites were situated on top promenade decks directly adjacent to the boat stations.'
            })
        elif pclass == 2:
            factors.append({
                'category': 'Accommodations',
                'impact': 'neutral',
                'title': 'Mid-Ship Location',
                'description': '2nd Class staterooms offered moderate staircase access to open boat decks.'
            })
        else:
            factors.append({
                'category': 'Accommodations',
                'impact': 'negative',
                'title': 'Lower Hull Berths & Bulkheads',
                'description': '3rd Class quarters were situated deep within lower hull sections with complex stairways.'
            })

        # 3. Age Factors
        if age <= 12:
            factors.append({
                'category': 'Demographics',
                'impact': 'positive',
                'title': 'Child Boarding Priority',
                'description': f'At {int(age)} years old, young passengers received priority escort into early emergency craft.'
            })
        elif age >= 60:
            factors.append({
                'category': 'Demographics',
                'impact': 'negative',
                'title': 'Physical Mobility in Crisis',
                'description': f'Senior travelers ({int(age)} years) faced intense physical hurdles ascending icy ladders.'
            })

        # 4. Family & Companionship
        if 2 <= family_size <= 4:
            factors.append({
                'category': 'Companionship',
                'impact': 'positive',
                'title': 'Optimal Family Size',
                'description': f'Traveling in a supportive group of {family_size} provided critical mutual assistance.'
            })
        elif family_size > 4:
            factors.append({
                'category': 'Companionship',
                'impact': 'negative',
                'title': 'Large Group Coordination',
                'description': f'Large families of {family_size} faced extreme difficulty remaining together during evacuation.'
            })
        elif is_alone:
            factors.append({
                'category': 'Companionship',
                'impact': 'neutral',
                'title': 'Solo Traveler',
                'description': 'Solo travelers possessed individual mobility but lacked group search and alert assistance.'
            })

        # 5. Ticket Fare
        if fare > 60:
            factors.append({
                'category': 'Service Tier',
                'impact': 'positive',
                'title': 'VIP Passage & Steward Assistance',
                'description': f'High-tier ticket (£{round(fare, 1)}) ensured dedicated steward assistance to boat stations.'
            })

        return factors

    def get_presets(self):
        """Provide realistic passenger presets for immediate testing."""
        return [
            {
                'id': 'vip_lady',
                'label': 'Lady Eleanor Vance',
                'subtitle': '1st Class VIP Patron',
                'name': 'Vance, Lady. Eleanor Margaret',
                'title': 'Rare',
                'sex': 'female',
                'pclass': 1,
                'age': 34,
                'fare': 120.0,
                'sibsp': 1,
                'parch': 0,
                'embarked': 'C'
            },
            {
                'id': 'working_mechanic',
                'label': 'Arthur Pendelton',
                'subtitle': '3rd Class Young Tradesman',
                'name': 'Pendelton, Mr. Arthur James',
                'title': 'Mr',
                'sex': 'male',
                'pclass': 3,
                'age': 23,
                'fare': 7.85,
                'sibsp': 0,
                'parch': 0,
                'embarked': 'S'
            },
            {
                'id': 'family_child',
                'label': 'Master Leo Davies',
                'subtitle': '2nd Class Child with Family',
                'name': 'Davies, Master. Leo Patrick',
                'title': 'Master',
                'sex': 'male',
                'pclass': 2,
                'age': 6,
                'fare': 32.50,
                'sibsp': 1,
                'parch': 2,
                'embarked': 'S'
            },
            {
                'id': 'independent_nurse',
                'label': 'Miss Clara Higgins',
                'subtitle': '2nd Class Solo Professional',
                'name': 'Higgins, Miss. Clara Maeve',
                'title': 'Miss',
                'sex': 'female',
                'pclass': 2,
                'age': 28,
                'fare': 16.50,
                'sibsp': 0,
                'parch': 0,
                'embarked': 'Q'
            },
            {
                'id': 'wealthy_gentleman',
                'label': 'Archibald Sterling',
                'subtitle': '1st Class Wealthy Merchant',
                'name': 'Sterling, Mr. Archibald Vance',
                'title': 'Mr',
                'sex': 'male',
                'pclass': 1,
                'age': 52,
                'fare': 78.25,
                'sibsp': 0,
                'parch': 0,
                'embarked': 'C'
            }
        ]

# Global engine instance
engine = CruiseModelEngine()
