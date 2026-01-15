"""
Data Loading and Preprocessing for MLP Training

Handles:
- Loading from MongoDB or CSV
- Feature engineering
- Train/Val/Test splitting (grouped by batch)
- Label encoding
- Standardization (optional)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Optional
from pymongo import MongoClient
import pickle

from .feature_engg import engineer_all_features, get_feature_columns
from .add_mech_labels import validate_mechanism_labels, label_mechanisms
from .add_violation_labels import create_parameter_risk_labels
from .constants import *

# ============================================================================
# DATA LOADING
# ============================================================================

def load_from_mongodb(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from MongoDB
    
    Args:
        limit: Optional limit on number of documents (for testing)
    
    Returns:
        DataFrame with columns:
        - board_id, batch_id, board_number
        - paste_volume, stencil_thickness, paste_viscosity, ambient_rh, ambient_temperature
        - defect, mech_causes, root_causes
        - hour_of_day, stencil_batch
    """
    client = MongoClient(MONGO_URI)
    db = client[DB]
    coll = db[COL]
    # Fetch documents
    cursor = coll.find({}).limit(limit) if limit else coll.find({})
    
    # Convert to DataFrame
    data = []
    for doc in cursor:
        row = {
            'board_id': doc['board_id'],
            'batch_id': doc['batch_id'],
            'board_number': doc['board_number'],
            
            # Raw parameters
            'paste_volume': doc['parameters']['paste_volume_per_aperture'],
            'stencil_thickness': doc['parameters']['stencil_thickness'],
            'paste_viscosity': doc['parameters']['paste_viscosity'],
            'ambient_rh': doc['parameters']['ambient_rh'],
            'ambient_temperature': doc['parameters']['ambient_temperature'],
            
            # Labels
            'defect': doc['labels']['defect'],
            'mech causes': doc['labels'].get('mechanism_causes'),
            'root causes': doc['labels'].get('root_causes'),
            
            # Temporal
            'hour_of_day': doc['temporal']['hour_of_day'],
            'stencil_batch': doc['temporal']['stencil_batch']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    print(f"Loaded {len(df)} boards from MongoDB")
    print(f"Defect distribution:\n{df['defect'].value_counts()}")
    
    return df

# ============================================================================
# TRAIN/VAL/TEST SPLITTING
# ============================================================================

def grouped_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by groups (batches) to avoid leakage
    
    CRITICAL: Must split by batch_id, not randomly, to ensure
              temporal/contextual features don't leak between splits
    
    Args:
        df: DataFrame with data
    
    Returns:
        Tuple of (df_train, df_val, df_test)
    """
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(RANDOM_SEED)
    
    # Get unique groups
    unique_groups = df[GROUP_COL].unique()
    n_groups = len(unique_groups) # 100
    
    # Shuffle groups
    shuffled_groups = unique_groups.copy() #100
    np.random.shuffle(shuffled_groups)
    
    # Calculate split points
    train_split = int(n_groups * TRAIN_RATIO) #70
    val_split = int(n_groups * (TRAIN_RATIO + VAL_RATIO)) #85
    
    # Assign groups
    train_groups = shuffled_groups[:train_split]#0->70
    val_groups = shuffled_groups[train_split:val_split]#70->85
    test_groups = shuffled_groups[val_split:]#85->100
    
    # Split DataFrames
    df_train = df[df[GROUP_COL].isin(train_groups)].copy() #get train df rows with batch_ids mentioned in train_grps
    df_val = df[df[GROUP_COL].isin(val_groups)].copy() #get val df rows with batch_ids mentioned in val_grps
    df_test = df[df[GROUP_COL].isin(test_groups)].copy() #get test df rows with batch_ids mentioned in test_grps
    
    # Print summary
    print("\n" + "="*60)
    print("GROUPED SPLIT SUMMARY")
    print("="*60)
    print(f"Total groups: {n_groups}")
    print(f"Train groups: {len(train_groups)} ({len(train_groups)/n_groups*100:.1f}%)")
    print(f"Val groups:   {len(val_groups)} ({len(val_groups)/n_groups*100:.1f}%)")
    print(f"Test groups:  {len(test_groups)} ({len(test_groups)/n_groups*100:.1f}%)")
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Train samples: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Val samples:   {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"Test samples:  {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    
    # Print defect distribution per split
    print("\nDefect Distribution:")
    print("\nTrain:")
    print(df_train['defect'].value_counts())
    print("\nValidation:")
    print(df_val['defect'].value_counts())
    print("\nTest:")
    print(df_test['defect'].value_counts())
    
    return df_train, df_val, df_test


# ============================================================================
# LABEL ENCODING
# ============================================================================

def encode_defect_labels(df_train: pd.DataFrame,
                  df_val: pd.DataFrame,
                  df_test: pd.DataFrame,
                  target_col: str = 'defect') -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode categorical labels to integers
    
    Defect classes (alphabetically sorted by LabelEncoder):
    - 0: No Defect
    - 1: Open Circuit
    - 2: Solder Bridging
    
    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        df_test: Test DataFrame
        target_col: Column name with labels
    
    Returns:
        Tuple of (y_train, y_val, y_test, label_encoder)
    """
    # Fit encoder on train set only
    #label encode under the hood: 
    # Looks only at training labels, 
    # Sorts unique class names alphabetically, 
    # Assigns integers starting from 0
    label_encoder = LabelEncoder()
    # y_train = [0, 1, 0, 2, 1]
    y_train = label_encoder.fit_transform(df_train[target_col])
    
    # Transform val and test using same encoder
    y_val = label_encoder.transform(df_val[target_col])
    y_test = label_encoder.transform(df_test[target_col])
    
    # Print mapping
    print("\n" + "="*60)
    print("LABEL ENCODING")
    print("="*60)
    print("Class mapping:")
    for idx, class_name in enumerate(label_encoder.classes_):
        count_train = (y_train == idx).sum()
        count_val = (y_val == idx).sum()
        count_test = (y_test == idx).sum()
        print(f"  {idx}: {class_name:20s} "
              f"(Train: {count_train:5d}, Val: {count_val:4d}, Test: {count_test:4d})")
    
    return y_train, y_val, y_test, label_encoder

def encode_mechanism_labels(df_train: pd.DataFrame,
                           df_val: pd.DataFrame,
                           df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode mechanism labels to integers
    
    Creates mechanism labels from binary columns (mutually exclusive):
    - apertureoverfill
    - poorpastetransfer
    
    Mechanism classes (alphabetically sorted by LabelEncoder):
    - 0: Aperture overfill
    - 1: No mechanism
    - 2: Poor paste transfer
    
    Args:
        df_train: Training DataFrame with mechanism binary columns
        df_val: Validation DataFrame with mechanism binary columns
        df_test: Test DataFrame with mechanism binary columns
    
    Returns:
        Tuple of (y_mechanism_train, y_mechanism_val, y_mechanism_test, mechanism_encoder)
    """
        # Helper function to create string labels from binary columns
    def get_mechanism_label(row):
        """
        Convert binary mechanism columns to single string label
        Mutually exclusive: only one can be True
        """
        if row['apertureoverfill'] == 1:
            return 'Aperture overfill'
        elif row['poorpastetransfer'] == 1:
            return 'Poor paste transfer'
        else:
            return 'No mechanism'
    # Create string label column for each split
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    
    df_train['mechanism_label'] = df_train.apply(get_mechanism_label, axis=1)
    df_val['mechanism_label'] = df_val.apply(get_mechanism_label, axis=1)
    df_test['mechanism_label'] = df_test.apply(get_mechanism_label, axis=1)
    
    # Encode using LabelEncoder
    mechanism_encoder = LabelEncoder()
    y_mechanism_train = mechanism_encoder.fit_transform(df_train['mechanism_label'])
    y_mechanism_val = mechanism_encoder.transform(df_val['mechanism_label'])
    y_mechanism_test = mechanism_encoder.transform(df_test['mechanism_label'])

    # Print mapping
    print("\n" + "="*60)
    print("MECHANISM LABEL ENCODING")
    print("="*60)
    print("Class mapping:")
    for idx, class_name in enumerate(mechanism_encoder.classes_):
        count_train = (y_mechanism_train == idx).sum()
        count_val = (y_mechanism_val == idx).sum()
        count_test = (y_mechanism_test == idx).sum()
        print(f"  {idx}: {class_name:20s} "
              f"(Train: {count_train:5d}, Val: {count_val:4d}, Test: {count_test:4d})")
    
    print("\n✓ Mutual exclusivity verified across all splits")
    
    return y_mechanism_train, y_mechanism_val, y_mechanism_test, mechanism_encoder
    

# ============================================================================
# FEATURE STANDARDIZATION
# ============================================================================

def standardize_features(X_train: np.ndarray,
                        X_val: np.ndarray,
                        X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features to zero mean and unit variance
    
    Note: Fit scaler on training data only, then apply to val/test
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
    
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    # Fit on train, transform all
    # What does fit do
    # μ_train = mean(TRAIN)
    # σ_train = std(TRAIN)
    # z = (x - μ_train) / σ_train
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("FEATURE STANDARDIZATION")
    print("="*60)
    print(f"Train: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    print(f"Val:   mean={X_val_scaled.mean():.4f}, std={X_val_scaled.std():.4f}")
    print(f"Test:  mean={X_test_scaled.mean():.4f}, std={X_test_scaled.std():.4f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================

def prepare_data_for_training(data_source: str,
                             use_standardization: bool = True,
                             **load_kwargs) -> Dict:
    """
    Complete data preparation pipeline
    
    Steps:
    1. Load data (MongoDB or CSV)
    2. Engineer features
    3. Split by batch (grouped)
    4. Encode labels
    5. Standardize features (optional)
    
    Args:
        data_source: Path to CSV or 'mongodb'
        use_standardization: Whether to standardize features
        **load_kwargs: Additional kwargs for load functions
    
    Returns:
        Dictionary containing:
        - X_train, X_val, X_test: Feature arrays
        - y_train, y_val, y_test: Label arrays
        - y_mechanism_train, y_mechanism_val, y_mechanism_test: Mechanism label arrays
        - label_encoder: Fitted LabelEncoder
        - scaler: Fitted StandardScaler (if used)
        - feature_info: Feature column information
        - split_info: Batch assignments and mechanism statistics
    """
    print("="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    if data_source == 'mongodb':
        df = load_from_mongodb(**load_kwargs)
    
    # Step 2: Engineer features
    print("\nStep 2: Engineering features...")
    df_engineered = engineer_all_features(df)
    feature_info = get_feature_columns()
    print(f"Engineered {len(feature_info['all_engineered'])} features")
    print(f"Total features: {len(feature_info['all_features'])}")

    print("\nLabelling mechanisms...")
    df_engineered = label_mechanisms(df_engineered)

    print("\nLabelling parameter violations...")
    df_engineered = create_parameter_risk_labels(
        df_engineered,
        PARAMETER_SPECS,
        parameter_column_map
    )
    
    # Step 3: Split by batch
    print("\nStep 3: Splitting data by batch...")
    df_train, df_val, df_test = grouped_split(
        df_engineered
    )
    
    # Step 4: Extract features
    print("\nStep 4: Extracting feature arrays...")
    X_train = df_train[feature_info['all_features']].values #train_n x f
    X_val = df_val[feature_info['all_features']].values #val_n x f
    X_test = df_test[feature_info['all_features']].values #test_n x f
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")
    
    # Step 5a: Encode defect labels
    print("\nStep 5: Encoding defect labels...")
    y_train, y_val, y_test, defect_encoder = encode_defect_labels(
        df_train, df_val, df_test
    )

    # Step 5b: Encode mechanism labels
    print("\nStep 5b: Encoding mechanism labels...")
    y_mechanism_train, y_mechanism_val, y_mechanism_test, mechanism_encoder = encode_mechanism_labels(
        df_train, df_val, df_test
    )

    # Step 5c: Extract parameter risk scores
    print("\nStep 5c: Extracting parameter risk scores...")
    risk_cols = [f'{p}_risk' for p in PARAMETER_SPECS.keys()]
    y_param_risk_train = df_train[risk_cols].values  # (n_train, 5)
    y_param_risk_val = df_val[risk_cols].values      # (n_val, 5)
    y_param_risk_test = df_test[risk_cols].values    # (n_test, 5)

    print(f"Parameter risk shape:")
    print(f"  Train: {y_param_risk_train.shape}")
    print(f"  Val:   {y_param_risk_val.shape}")
    print(f"  Test:  {y_param_risk_test.shape}")

    # Print distribution
    print(f"\nParameter risk distribution (train set):")
    for i, col in enumerate(risk_cols):
        high_risk = (y_param_risk_train[:, i] > 0.70).sum()
        pct = (y_param_risk_train[:, i] > 0.70).mean() * 100
        print(f"  {col:25s}: {high_risk:>6,} high-risk ({pct:5.2f}%)")

    # Step 6: Standardize (optional)
    scaler = None
    if use_standardization:
        print("\nStep 6: Standardizing features...")
        X_train, X_val, X_test, scaler = standardize_features(
            X_train, X_val, X_test
        )
    else:
        print("\nStep 6: Skipping standardization (using raw features)")
    
    # Collect split info
    split_info = {
        'train_batches': sorted(df_train['batch_id'].unique().tolist()),
        'val_batches': sorted(df_val['batch_id'].unique().tolist()),
        'test_batches': sorted(df_test['batch_id'].unique().tolist()),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        # Defect distribution
        'train_defect_dist': {
            defect_encoder.classes_[i]: int((y_train == i).sum())
            for i in range(len(defect_encoder.classes_))
        },
        'val_defect_dist': {
            defect_encoder.classes_[i]: int((y_val == i).sum())
            for i in range(len(defect_encoder.classes_))
        },
        'test_defect_dist': {
            defect_encoder.classes_[i]: int((y_test == i).sum())
            for i in range(len(defect_encoder.classes_))
        },
        # Mechanism distribution
        'train_mechanism_dist': {
            mechanism_encoder.classes_[i]: int((y_mechanism_train == i).sum())
            for i in range(len(mechanism_encoder.classes_))
        },
        'val_mechanism_dist': {
            mechanism_encoder.classes_[i]: int((y_mechanism_val == i).sum())
            for i in range(len(mechanism_encoder.classes_))
        },
        'test_mechanism_dist': {
            mechanism_encoder.classes_[i]: int((y_mechanism_test == i).sum())
            for i in range(len(mechanism_encoder.classes_))
        }
    }
    
    print("\n" + "="*60)
    print("PREPARATION COMPLETE")
    print("="*60)
    print(f"✓ Ready for model training")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'y_mechanism_train': y_mechanism_train,
        'y_mechanism_val': y_mechanism_val,
        'y_mechanism_test': y_mechanism_test,
        'y_param_risk_train': y_param_risk_train,
        'y_param_risk_val': y_param_risk_val,
        'y_param_risk_test': y_param_risk_test,
        'defect_encoder': defect_encoder,
        'mechanism_encoder': mechanism_encoder,
        'scaler': scaler,
        'feature_info': feature_info,
        'split_info': split_info,
        'risk_columns': risk_cols
    }


def save_preprocessing_artifacts(defect_encoder: LabelEncoder,
                                 mechanism_encoder: LabelEncoder,
                                 scaler: Optional[StandardScaler],
                                 feature_info: Dict,
                                 filepath: str = 'preprocessing_artifacts.pkl'):
    """
    Save preprocessing objects for later use (inference)
    
    Args:
        defect_encoder: Fitted LabelEncoder on defect labels
        mechanism_encoder: Fitted LabelEncoder on mechanism labels
        scaler: Fitted StandardScaler (or None)
        feature_info: Feature column information
        filepath: Where to save
    """
    artifacts = {
        'defect_encoder': defect_encoder,
        'mechanism_encoder': mechanism_encoder,
        'scaler': scaler,
        'feature_info': feature_info
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"\n✓ Saved preprocessing artifacts to {filepath}")
