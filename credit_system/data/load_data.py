from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

def load_german_credit_data():
    """Load and preprocess the German Credit dataset"""
    # Load dataset
    statlog_german_credit_data = fetch_ucirepo(id=144)
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets
    
    # Process data for interpretability
    # Add meaningful feature names for documentation
    feature_names = {
        'Attribute1': 'checking_account_status',
        'Attribute2': 'duration_months',
        'Attribute3': 'credit_history',
        'Attribute4': 'purpose',
        'Attribute5': 'credit_amount',
        'Attribute6': 'savings_account',
        'Attribute7': 'employment_since',
        'Attribute8': 'installment_rate',
        'Attribute9': 'personal_status_sex',
        'Attribute10': 'other_debtors',
        'Attribute11': 'present_residence',
        'Attribute12': 'property',
        'Attribute13': 'age',
        'Attribute14': 'other_installments',
        'Attribute15': 'housing',
        'Attribute16': 'existing_credits',
        'Attribute17': 'job',
        'Attribute18': 'num_dependents',
        'Attribute19': 'telephone',
        'Attribute20': 'foreign_worker'
    }
    
    # Rename columns for better interpretability
    X = X.rename(columns=feature_names)

    attribute_mapping = {
        'checking_account_status': {
            'A11': '< 0 DM',
            'A12': '0 <= ... < 200 DM',
            'A13': '>= 200 DM / salary assignment',
            'A14': 'no checking account'
        },
        'credit_history': {
            'A30': 'no credits / all paid back duly',
            'A31': 'all credits paid back duly',
            'A32': 'existing credits paid back duly',
            'A33': 'delayed payments',
            'A34': 'critical account / other credits'
        },
        'purpose': {
            'A40': 'car (new)',
            'A41': 'car (used)',
            'A42': 'furniture/equipment',
            'A43': 'radio/television',
            'A44': 'domestic appliances',
            'A45': 'repairs',
            'A46': 'education',
            'A47': 'vacation',
            'A48': 'retraining',
            'A49': 'business',
            'A410': 'others'
        },
        'savings_account': {
            'A61': '< 100 DM',
            'A62': '100 <= ... < 500 DM',
            'A63': '500 <= ... < 1000 DM',
            'A64': '>= 1000 DM',
            'A65': 'unknown / no savings'
        },
        'employment_since': {
            'A71': 'unemployed',
            'A72': '< 1 year',
            'A73': '1 <= ... < 4 years',
            'A74': '4 <= ... < 7 years',
            'A75': '>= 7 years'
        },
        'personal_status_sex': {
            'A91': 'male: divorced/separated',
            'A92': 'female: div/sep/married',
            'A93': 'male: single',
            'A94': 'male: married/widowed',
            'A95': 'female: single'
        },
        'other_debtors': {
            'A101': 'none',
            'A102': 'co-applicant',
            'A103': 'guarantor'
        },
        'property': {
            'A121': 'real estate',
            'A122': 'building society savings / life insurance',
            'A123': 'car or other',
            'A124': 'unknown / no property'
        },
        'other_installments': {
            'A141': 'bank',
            'A142': 'stores',
            'A143': 'none'
        },
        'housing': {
            'A151': 'rent',
            'A152': 'own',
            'A153': 'for free'
        },
        'job': {
            'A171': 'unemployed / unskilled non-resident',
            'A172': 'unskilled resident',
            'A173': 'skilled employee / official',
            'A174': 'management / self-employed / high qualification'
        },
        'telephone': {
            'A191': 'none',
            'A192': 'yes, registered'
        },
        'foreign_worker': {
            'A201': 'yes',
            'A202': 'no'
        }
    }

    for col, mapping in attribute_mapping.items():
        if col in X.columns:
            X[col] = X[col].replace(mapping)

    
    # Convert target labels from {1, 2} to {0, 1} where 1 is "bad" (default)
    y = y.replace({1: 0, 2: 1})
    
    # Add clear target name
    y.name = "credit_risk"
    
    # Identify protected attributes
    protected_attributes = ["personal_status_sex", "age", "foreign_worker"]
    
    return X, y, protected_attributes

def prepare_data_splits(X, y, test_size=0.1):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Identify column types
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()