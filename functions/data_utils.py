"""
Data utility functions for loading, cleaning, and preprocessing data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import pickle

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format")

def basic_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Data types:\n{df.dtypes}")

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        test_size (float): Proportion of test set (default: 0.2)
        val_size (float): Proportion of validation set (default: 0.2)
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_split_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def identify_column_types(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify numeric and categorical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Tuple: (numeric_cols, categorical_cols)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)
    
    return numeric_cols, categorical_cols

def handle_na(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, 
              numeric_cols: list, categorical_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Handle missing values using imputation.
    
    Args:
        X_train, X_val, X_test: Training, validation, and test sets
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Tuple: Cleaned X_train, X_val, X_test, numeric_imputer, categorical_imputer
    """
    numeric_imputer = None
    categorical_imputer = None
    
    # Handle numeric columns if they exist
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy='median')
        
        X_train_numeric = pd.DataFrame(
            numeric_imputer.fit_transform(X_train[numeric_cols]),
            columns=numeric_cols,
            index=X_train.index
        )
        
        X_val_numeric = pd.DataFrame(
            numeric_imputer.transform(X_val[numeric_cols]),
            columns=numeric_cols,
            index=X_val.index
        )
        
        X_test_numeric = pd.DataFrame(
            numeric_imputer.transform(X_test[numeric_cols]),
            columns=numeric_cols,
            index=X_test.index
        )
    else:
        # Create empty DataFrames with correct indices if no numeric columns
        X_train_numeric = pd.DataFrame(index=X_train.index)
        X_val_numeric = pd.DataFrame(index=X_val.index)
        X_test_numeric = pd.DataFrame(index=X_test.index)
    
    # Handle categorical columns if they exist
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        X_train_categorical = pd.DataFrame(
            categorical_imputer.fit_transform(X_train[categorical_cols]),
            columns=categorical_cols,
            index=X_train.index
        )
        
        X_val_categorical = pd.DataFrame(
            categorical_imputer.transform(X_val[categorical_cols]),
            columns=categorical_cols,
            index=X_val.index
        )
        
        X_test_categorical = pd.DataFrame(
            categorical_imputer.transform(X_test[categorical_cols]),
            columns=categorical_cols,
            index=X_test.index
        )
    else:
        # Create empty DataFrames with correct indices if no categorical columns
        X_train_categorical = pd.DataFrame(index=X_train.index)
        X_val_categorical = pd.DataFrame(index=X_val.index)
        X_test_categorical = pd.DataFrame(index=X_test.index)
    
    # Combine back together
    X_train_clean = pd.concat([X_train_numeric, X_train_categorical], axis=1)
    X_val_clean = pd.concat([X_val_numeric, X_val_categorical], axis=1)
    X_test_clean = pd.concat([X_test_numeric, X_test_categorical], axis=1)
    
    print(f"Missing values after imputation:")
    print(f"Training: {X_train_clean.isnull().sum().sum()}")
    print(f"Validation: {X_val_clean.isnull().sum().sum()}")
    print(f"Test: {X_test_clean.isnull().sum().sum()}")
    
    return X_train_clean, X_val_clean, X_test_clean, numeric_imputer, categorical_imputer

def encode(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, 
           categorical_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical variables using One-Hot Encoding without dropping columns.
    This ensures compatibility with future data that will have original column structure.
    
    Args:
        X_train, X_val, X_test: Training, validation, and test sets
        categorical_cols: List of categorical column names
        
    Returns:
        Tuple: Encoded X_train, X_val, X_test and the encoder
    """
    categorical_encoder = None
    
    if categorical_cols:
        # Use drop=None to keep all columns for future data compatibility
        categorical_encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
        
        X_train_categorical_encoded = pd.DataFrame(
            categorical_encoder.fit_transform(X_train[categorical_cols]),
            columns=categorical_encoder.get_feature_names_out(categorical_cols),
            index=X_train.index
        )
        
        X_val_categorical_encoded = pd.DataFrame(
            categorical_encoder.transform(X_val[categorical_cols]),
            columns=categorical_encoder.get_feature_names_out(categorical_cols),
            index=X_val.index
        )
        
        X_test_categorical_encoded = pd.DataFrame(
            categorical_encoder.transform(X_test[categorical_cols]),
            columns=categorical_encoder.get_feature_names_out(categorical_cols),
            index=X_test.index
        )
        
        print(f"Encoded categorical features shape:")
        print(f"Training: {X_train_categorical_encoded.shape}")
        print(f"Validation: {X_val_categorical_encoded.shape}")
        print(f"Test: {X_test_categorical_encoded.shape}")
    else:
        # Create empty DataFrames with correct indices if no categorical columns
        X_train_categorical_encoded = pd.DataFrame(index=X_train.index)
        X_val_categorical_encoded = pd.DataFrame(index=X_val.index)
        X_test_categorical_encoded = pd.DataFrame(index=X_test.index)
        
        print("No categorical columns to encode.")
    
    return X_train_categorical_encoded, X_val_categorical_encoded, X_test_categorical_encoded, categorical_encoder


def preprocess_pipeline(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        
    Returns:
        Tuple: Processed X_train, X_val, X_test, y_train, y_val, y_test, and fitted transformers
    """
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col)
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(X_train)
    
    # Handle missing values
    X_train_clean, X_val_clean, X_test_clean, numeric_imputer, categorical_imputer = handle_na(
        X_train, X_val, X_test, numeric_cols, categorical_cols
    )
    
    # Encode categorical variables (only if categorical columns exist)
    if categorical_cols:
        X_train_cat_encoded, X_val_cat_encoded, X_test_cat_encoded, categorical_encoder = encode(
            X_train_clean, X_val_clean, X_test_clean, categorical_cols
        )
    else:
        X_train_cat_encoded = pd.DataFrame(index=X_train_clean.index)
        X_val_cat_encoded = pd.DataFrame(index=X_val_clean.index)
        X_test_cat_encoded = pd.DataFrame(index=X_test_clean.index)
        categorical_encoder = None
    
    # Scale numeric features (only if numeric columns exist)
    if numeric_cols:
        X_train_num_scaled, X_val_num_scaled, X_test_num_scaled, numeric_scaler = scale(
            X_train_clean[numeric_cols], X_val_clean[numeric_cols], X_test_clean[numeric_cols], numeric_cols
        )
    else:
        X_train_num_scaled = pd.DataFrame(index=X_train_clean.index)
        X_val_num_scaled = pd.DataFrame(index=X_val_clean.index)
        X_test_num_scaled = pd.DataFrame(index=X_test_clean.index)
        numeric_scaler = None
    
    # Combine final datasets
    X_train_final = pd.concat([X_train_num_scaled, X_train_cat_encoded], axis=1)
    X_val_final = pd.concat([X_val_num_scaled, X_val_cat_encoded], axis=1)
    X_test_final = pd.concat([X_test_num_scaled, X_test_cat_encoded], axis=1)
    
    print(f"\nFinal processed dataset shapes:")
    print(f"Training: {X_train_final.shape}")
    print(f"Validation: {X_val_final.shape}")
    print(f"Test: {X_test_final.shape}")
    
    # Store fitted transformers
    transformers = {
        'numeric_imputer': numeric_imputer,
        'categorical_imputer': categorical_imputer,
        'categorical_encoder': categorical_encoder,
        'numeric_scaler': numeric_scaler,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    
    return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, transformers

def transform_new_data(X_new: pd.DataFrame, transformers: dict) -> pd.DataFrame:
    """
    Apply fitted transformers to new data for prediction.
    
    Args:
        X_new (pd.DataFrame): New data with original structure (unprocessed)
        transformers (dict): Dictionary containing fitted transformers from preprocess_pipeline
        
    Returns:
        pd.DataFrame: Processed data ready for prediction
    """
    print("Applying transformations to new data...")
    
    numeric_cols = transformers['numeric_cols']
    categorical_cols = transformers['categorical_cols']
    
    # Handle missing values
    X_new_processed = X_new.copy()
    
    if numeric_cols and transformers['numeric_imputer'] is not None:
        X_new_numeric = pd.DataFrame(
            transformers['numeric_imputer'].transform(X_new[numeric_cols]),
            columns=numeric_cols,
            index=X_new.index
        )
    else:
        X_new_numeric = pd.DataFrame(index=X_new.index)
    
    if categorical_cols and transformers['categorical_imputer'] is not None:
        X_new_categorical = pd.DataFrame(
            transformers['categorical_imputer'].transform(X_new[categorical_cols]),
            columns=categorical_cols,
            index=X_new.index
        )
    else:
        X_new_categorical = pd.DataFrame(index=X_new.index)
    
    # Encode categorical variables
    if categorical_cols and transformers['categorical_encoder'] is not None:
        X_new_categorical_encoded = pd.DataFrame(
            transformers['categorical_encoder'].transform(X_new_categorical),
            columns=transformers['categorical_encoder'].get_feature_names_out(categorical_cols),
            index=X_new.index
        )
    else:
        X_new_categorical_encoded = pd.DataFrame(index=X_new.index)
    
    # Scale numeric features
    if numeric_cols and transformers['numeric_scaler'] is not None:
        X_new_numeric_scaled = pd.DataFrame(
            transformers['numeric_scaler'].transform(X_new_numeric),
            columns=numeric_cols,
            index=X_new.index
        )
    else:
        X_new_numeric_scaled = pd.DataFrame(index=X_new.index)
    
    # Combine final dataset
    X_new_final = pd.concat([X_new_numeric_scaled, X_new_categorical_encoded], axis=1)
    
    print(f"New data processed successfully!")
    print(f"Original shape: {X_new.shape}")
    print(f"Processed shape: {X_new_final.shape}")
    
    return X_new_final

def save_transformers(transformers: dict, filepath: str) -> None:
    """
    Save fitted transformers to file for future use.
    
    Args:
        transformers (dict): Dictionary containing fitted transformers
        filepath (str): Path to save the transformers
    """
    
    with open(filepath, 'wb') as f:
        pickle.dump(transformers, f)
    
    print(f"Transformers saved to {filepath}")

def load_transformers(filepath: str) -> dict:
    """
    Load fitted transformers from file.
    
    Args:
        filepath (str): Path to load the transformers from
        
    Returns:
        dict: Dictionary containing fitted transformers
    """
    
    with open(filepath, 'rb') as f:
        transformers = pickle.load(f)
    
    print(f"Transformers loaded from {filepath}")
    return transformers

def save_datasets(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, 
                  y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, 
                  folder_path: str = '../data/datasets') -> None:
    """
    Save processed datasets to CSV files.
    
    Args:
        X_train, X_val, X_test: Feature datasets
        y_train, y_val, y_test: Target datasets
        folder_path (str): Path to save the datasets
    """
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Save feature datasets
    X_train.to_csv(f'{folder_path}/X_train.csv', index=False)
    X_val.to_csv(f'{folder_path}/X_val.csv', index=False)
    X_test.to_csv(f'{folder_path}/X_test.csv', index=False)
    
    # Save target datasets
    y_train.to_csv(f'{folder_path}/y_train.csv', index=False, header=['target'])
    y_val.to_csv(f'{folder_path}/y_val.csv', index=False, header=['target'])
    y_test.to_csv(f'{folder_path}/y_test.csv', index=False, header=['target'])
    
    print(f"Datasets saved successfully to {folder_path}/")
    print(f"Files saved:")
    print(f"  - X_train.csv: {X_train.shape}")
    print(f"  - X_val.csv: {X_val.shape}")
    print(f"  - X_test.csv: {X_test.shape}")
    print(f"  - y_train.csv: {y_train.shape}")
    print(f"  - y_val.csv: {y_val.shape}")
    print(f"  - y_test.csv: {y_test.shape}")

def load_datasets(folder_path: str = '../data/processed') -> tuple:
    """
    Load previously saved datasets from CSV files.
    
    Args:
        folder_path (str): Path to load the datasets from
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    import os
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    # Load feature datasets
    X_train = pd.read_csv(f'{folder_path}/X_train.csv')
    X_val = pd.read_csv(f'{folder_path}/X_val.csv')
    X_test = pd.read_csv(f'{folder_path}/X_test.csv')
    
    # Load target datasets
    y_train = pd.read_csv(f'{folder_path}/y_train.csv')['target']
    y_val = pd.read_csv(f'{folder_path}/y_val.csv')['target']
    y_test = pd.read_csv(f'{folder_path}/y_test.csv')['target']
    
    print(f"Datasets loaded successfully from {folder_path}/")
    print(f"Loaded datasets:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_val: {X_val.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_val: {y_val.shape}")
    print(f"  - y_test: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_data_overview(df: pd.DataFrame, target_col: str = None, figsize: tuple = (15, 10)) -> None:
    """
    Create comprehensive overview plots for the dataset with dark theme and pastel colors.
    Automatically adapts to categorical and numeric variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name (optional)
        figsize (tuple): Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set dark theme with pastel colors
    plt.style.use('dark_background')
    colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD1DC', '#E1BAFF', '#C9FFE1', '#FFE1BA']
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from features if specified
    if target_col:
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
    
    # Calculate subplot layout
    total_plots = len(numeric_cols) + len(categorical_cols)
    if target_col:
        total_plots += 1  # Add target distribution plot
    
    if total_plots == 0:
        print("No columns to plot")
        return
    
    cols = 3
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, facecolor='#2E2E2E')
    fig.suptitle('Data Overview', fontsize=16, color='white', y=0.98)
    
    # Flatten axes for easy indexing
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot target distribution first if specified
    if target_col and target_col in df.columns:
        ax = axes[plot_idx]
        if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
            # Categorical target
            value_counts = df[target_col].value_counts()
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                         color=colors[:len(value_counts)])
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        else:
            # Numeric target
            ax.hist(df[target_col].dropna(), bins=20, color=colors[0], alpha=0.7, edgecolor='black')
        
        ax.set_title(f'{target_col}', fontsize=12, color='white', pad=10)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plot_idx += 1
    
    # Plot numeric columns
    for i, col in enumerate(numeric_cols):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        # Check if it's actually categorical (few unique values)
        if df[col].nunique() < 10:
            # Treat as categorical
            value_counts = df[col].value_counts().sort_index()
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                         color=colors[i % len(colors)])
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index)
        else:
            # True numeric - histogram
            ax.hist(df[col].dropna(), bins=20, color=colors[i % len(colors)], 
                   alpha=0.7, edgecolor='black')
        
        ax.set_title(col, fontsize=10, color='white', pad=5)
        ax.tick_params(colors='white', labelsize=8)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plot_idx += 1
    
    # Plot categorical columns
    for i, col in enumerate(categorical_cols):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        value_counts = df[col].value_counts().head(10)  # Top 10 categories
        
        bars = ax.bar(range(len(value_counts)), value_counts.values, 
                     color=colors[(i + len(numeric_cols)) % len(colors)])
        
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
        ax.set_title(col, fontsize=10, color='white', pad=5)
        ax.tick_params(colors='white', labelsize=8)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, target_col: str = None, figsize: tuple = (10, 8)) -> None:
    """
    Plot correlation matrix for numeric columns with dark theme.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name (optional)
        figsize (tuple): Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation matrix")
        return
    
    # Set dark theme
    plt.style.use('dark_background')
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, facecolor='#2E2E2E')
    
    # Custom colormap with pastel colors
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                annot=True, fmt='.2f', ax=ax)
    
    ax.set_title('Correlation Matrix', fontsize=14, color='white', pad=20)
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.show()

def analyze_and_create_validation_schema(df: pd.DataFrame, target_column: str, schema_path: str = '../models/validation_schema.json') -> dict:
    """
    Analizza il dataset e crea uno schema di validazione automatico.
    
    Args:
        df (pd.DataFrame): Dataset da analizzare
        target_column (str): Nome della colonna target
        schema_path (str): Percorso dove salvare lo schema JSON
        
    Returns:
        dict: Schema di validazione generato
    """
    import json
    import os
    
    print("📊 ANALISI PER VALIDAZIONE DATI")
    print("="*60)

    # Identificazione colonne numeriche e categoriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Rimuoviamo il target dalle feature
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    print(f"\n🔢 FEATURE NUMERICHE ({len(numeric_cols)}):")
    print("-" * 40)
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        dtype = df[col].dtype
        has_nulls = df[col].isnull().any()
        print(f"{col:<30} | Min: {min_val:>8.2f} | Max: {max_val:>8.2f} | Type: {dtype} | Nulls: {has_nulls}")

    print(f"\n📝 FEATURE CATEGORICHE ({len(categorical_cols)}):")
    print("-" * 40)
    for col in categorical_cols:
        unique_vals = df[col].dropna().unique().tolist()
        n_unique = len(unique_vals)
        has_nulls = df[col].isnull().any()
        print(f"{col:<30} | Categories ({n_unique}): {unique_vals} | Nulls: {has_nulls}")

    print(f"\n🎯 TARGET VARIABLE:")
    print("-" * 40)
    target_vals = df[target_column].dropna().unique().tolist()
    print(f"{target_column:<30} | Classes: {target_vals}")

    # Creare dizionario per la validazione automatica
    validation_schema = {
        'numeric_constraints': {},
        'categorical_constraints': {},
        'target_classes': target_vals
    }

    for col in numeric_cols:
        validation_schema['numeric_constraints'][col] = {
            'min_value': float(df[col].min()),
            'max_value': float(df[col].max()),
            'type': str(df[col].dtype),
            'nullable': int(df[col].isnull().any())  # Convertire bool a int
        }

    for col in categorical_cols:
        validation_schema['categorical_constraints'][col] = {
            'allowed_values': df[col].dropna().unique().tolist(),
            'nullable': int(df[col].isnull().any())  # Convertire bool a int
        }

    print(f"\n✅ Schema di validazione creato!")
    print(f"   - {len(numeric_cols)} feature numeriche")
    print(f"   - {len(categorical_cols)} feature categoriche") 
    print(f"   - {len(target_vals)} classi target")

    # Mostra un esempio del dizionario
    print(f"\n📋 ESEMPIO SCHEMA:")
    print("-" * 40)
    print("Numeric constraints (primi 3):")
    for i, (col, constraints) in enumerate(list(validation_schema['numeric_constraints'].items())[:3]):
        print(f"  {col}: {constraints}")
    print("Categorical constraints (primi 2):")
    for i, (col, constraints) in enumerate(list(validation_schema['categorical_constraints'].items())[:2]):
        print(f"  {col}: {constraints}")

    # Salviamo lo schema per usarlo in produzione
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
    with open(schema_path, 'w') as f:
        json.dump(validation_schema, f, indent=2)
    print(f"\n💾 Schema salvato in: {schema_path}")
    
    return validation_schema


def validate_data_entry(entry: dict, schema: dict) -> dict:
    """
    Valida una singola entry usando lo schema di validazione.
    
    Args:
        entry (dict): Dizionario con i dati da validare
        schema (dict): Schema di validazione generato automaticamente
    
    Returns:
        dict: Risultato della validazione con errori (se presenti)
    """
    errors = []
    warnings = []
    
    # Validazione feature numeriche
    for col, constraints in schema['numeric_constraints'].items():
        if col in entry:
            value = entry[col]
            if value is not None:
                if value < constraints['min_value']:
                    errors.append(f"{col}: valore {value} < minimo {constraints['min_value']}")
                if value > constraints['max_value']:
                    errors.append(f"{col}: valore {value} > massimo {constraints['max_value']}")
        else:
            if not constraints['nullable']:
                warnings.append(f"{col}: campo mancante (richiesto)")
    
    # Validazione feature categoriche
    for col, constraints in schema['categorical_constraints'].items():
        if col in entry:
            value = entry[col]
            if value is not None and value not in constraints['allowed_values']:
                errors.append(f"{col}: '{value}' non è tra i valori ammessi {constraints['allowed_values']}")
        else:
            if not constraints['nullable']:
                warnings.append(f"{col}: campo mancante (richiesto)")
    
    # Validazione target (se presente)
    if 'diagnosis' in entry:
        target_value = entry['diagnosis']
        if target_value not in schema['target_classes']:
            errors.append(f"diagnosis: '{target_value}' non è tra le classi ammesse {schema['target_classes']}")
    
    # Calcola completeness solo per le feature (escludendo il target)
    feature_fields = list(schema['numeric_constraints'].keys()) + list(schema['categorical_constraints'].keys())
    present_features = len([k for k in feature_fields if k in entry and entry[k] is not None])
    total_features = len(feature_fields)
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'completeness': (present_features / total_features) * 100 if total_features > 0 else 0
    }

def create_preprocessing_pipeline(X_data: pd.DataFrame) -> dict:
    """
    Create and fit preprocessing transformers on a given dataset.
    
    This function creates the same transformers as preprocess_pipeline but only fits them
    on the provided data without splitting. Used for retraining transformers on combined
    train+validation data.
    
    Args:
        X_data (pd.DataFrame): Input features to fit transformers on
        
    Returns:
        dict: Dictionary containing fitted transformers with fit_transform capability
    """
    print(f"Creating preprocessing pipeline for data shape: {X_data.shape}")
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(X_data)
    
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
    
    # Initialize transformers
    numeric_imputer = None
    categorical_imputer = None
    categorical_encoder = None
    numeric_scaler = None
    
    # Create and fit numeric imputer
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy='median')
        numeric_imputer.fit(X_data[numeric_cols])
        print(f"✅ Fitted numeric imputer on {len(numeric_cols)} columns")
    
    # Create and fit categorical imputer
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        categorical_imputer.fit(X_data[categorical_cols])
        print(f"✅ Fitted categorical imputer on {len(categorical_cols)} columns")
        
        # Fit categorical encoder on imputed data
        X_cat_imputed = pd.DataFrame(
            categorical_imputer.transform(X_data[categorical_cols]),
            columns=categorical_cols
        )
        categorical_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        categorical_encoder.fit(X_cat_imputed)
        print(f"✅ Fitted categorical encoder on {len(categorical_cols)} columns")
    
    # Create and fit numeric scaler on imputed data
    if numeric_cols:
        X_num_imputed = pd.DataFrame(
            numeric_imputer.transform(X_data[numeric_cols]),
            columns=numeric_cols
        )
        numeric_scaler = StandardScaler()
        numeric_scaler.fit(X_num_imputed)
        print(f"✅ Fitted numeric scaler on {len(numeric_cols)} columns")
    
    # Create transformers dictionary with fit_transform capability
    transformers = {
        'numeric_imputer': numeric_imputer,
        'categorical_imputer': categorical_imputer,
        'categorical_encoder': categorical_encoder,
        'numeric_scaler': numeric_scaler,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    
    # Add fit_transform method to the transformers object
    def fit_transform_data(X):
        """Apply all transformations to input data and return numpy array"""
        print(f"Applying transformations to data shape: {X.shape}")
        
        numeric_cols = transformers['numeric_cols']
        categorical_cols = transformers['categorical_cols']
        
        # Handle missing values for numeric columns
        if numeric_cols and transformers['numeric_imputer'] is not None:
            X_numeric_imputed = pd.DataFrame(
                transformers['numeric_imputer'].transform(X[numeric_cols]),
                columns=numeric_cols,
                index=X.index
            )
        else:
            X_numeric_imputed = pd.DataFrame(index=X.index)
        
        # Handle missing values for categorical columns
        if categorical_cols and transformers['categorical_imputer'] is not None:
            X_categorical_imputed = pd.DataFrame(
                transformers['categorical_imputer'].transform(X[categorical_cols]),
                columns=categorical_cols,
                index=X.index
            )
        else:
            X_categorical_imputed = pd.DataFrame(index=X.index)
        
        # Encode categorical variables
        if categorical_cols and transformers['categorical_encoder'] is not None:
            X_categorical_encoded = pd.DataFrame(
                transformers['categorical_encoder'].transform(X_categorical_imputed),
                columns=transformers['categorical_encoder'].get_feature_names_out(categorical_cols),
                index=X.index
            )
        else:
            X_categorical_encoded = pd.DataFrame(index=X.index)
        
        # Scale numeric features
        if numeric_cols and transformers['numeric_scaler'] is not None:
            X_numeric_scaled = pd.DataFrame(
                transformers['numeric_scaler'].transform(X_numeric_imputed),
                columns=numeric_cols,
                index=X.index
            )
        else:
            X_numeric_scaled = pd.DataFrame(index=X.index)
        
        # Combine final dataset
        X_final = pd.concat([X_numeric_scaled, X_categorical_encoded], axis=1)
        print(f"Transformed data shape: {X_final.shape}")
        
        return X_final.values  # Return numpy array
    
    def transform_data(X):
        """Apply all transformations to input data and return numpy array"""
        return fit_transform_data(X)
    
    # Add methods to transformers dict
    transformers['fit_transform'] = fit_transform_data
    transformers['transform'] = transform_data
    
    print(f"✅ Preprocessing pipeline created successfully!")
    print(f"   - Numeric transformers: {'✅' if numeric_cols else '❌'}")
    print(f"   - Categorical transformers: {'✅' if categorical_cols else '❌'}")
    
    return transformers

def load_test_set_for_api(splitted_path: str = '../data/splitted') -> Dict:
    """
    Load the original test set from splitted data for API testing.
    
    Args:
        splitted_path: Path to the splitted data directory
        
    Returns:
        Dict containing X_test, y_test, features, and n_samples
    """
    X_test_path = os.path.join(splitted_path, 'X_test_raw.csv')
    y_test_path = os.path.join(splitted_path, 'y_test_raw.csv')
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Original test set not found in {splitted_path}")
    
    # Load test data
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)['target']
    
    print(f"✅ Loaded original test set from splitted data:")
    print(f"   - Samples: {len(X_test)}")
    print(f"   - Features: {len(X_test.columns)}")
    print(f"   - Feature names: {X_test.columns[:5].tolist()}{'...' if len(X_test.columns) > 5 else ''}")
    
    return {
        'X_test': X_test,
        'y_test': y_test,
        'features': X_test.columns.tolist(),
        'n_samples': len(X_test)
    }

def get_random_test_samples(splitted_path: str = '../data/splitted', n_samples: int = 1, random_state: int = 42) -> Dict:
    """
    Get random samples from the original test set for API testing.
    
    Args:
        splitted_path: Path to the splitted data directory
        n_samples: Number of random samples to return
        random_state: Random seed for reproducibility
        
    Returns:
        Dict containing sampled X_test, y_test, and metadata
    """
    test_data = load_test_set_for_api(splitted_path)
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get random indices
    total_samples = test_data['n_samples']
    random_indices = np.random.choice(total_samples, size=min(n_samples, total_samples), replace=False)
    
    # Extract random samples
    X_samples = test_data['X_test'].iloc[random_indices]
    y_samples = test_data['y_test'].iloc[random_indices]
    
    print(f"🎲 Randomly selected {len(random_indices)} sample(s) from test set:")
    print(f"   - Indices: {random_indices.tolist()}")
    print(f"   - True labels: {y_samples.tolist()}")
    
    return {
        'X_test': X_samples,
        'y_test': y_samples,
        'indices': random_indices,
        'features': test_data['features'],
        'n_samples': len(random_indices)
    }

def load_original_dataset_split(data_path: str = '../data/origin/depression.csv', target_col: str = 'diagnosis', 
                               test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42,
                               save_to_splitted: bool = True, splitted_path: str = '../data/splitted') -> tuple:
    """
    Load the original dataset and split it into train, validation, and test sets.
    This returns the RAW data before any preprocessing and optionally saves the splits.
    
    Args:
        data_path (str): Path to the original dataset
        target_col (str): Name of the target column
        test_size (float): Proportion of test set
        val_size (float): Proportion of validation set from remaining data
        random_state (int): Random state for reproducibility
        save_to_splitted (bool): Whether to save split datasets to splitted folder
        splitted_path (str): Path to save splitted datasets
        
    Returns:
        tuple: X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw
    """
    # Load original dataset
    df = load_data(data_path)
    
    # Split into features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # First split: separate test set
    X_temp, X_test_raw, y_temp, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Original dataset split successfully:")
    print(f"  - X_train_raw: {X_train_raw.shape}")
    print(f"  - X_val_raw: {X_val_raw.shape}")
    print(f"  - X_test_raw: {X_test_raw.shape}")
    print(f"  - y_train_raw: {y_train_raw.shape}")
    print(f"  - y_val_raw: {y_val_raw.shape}")
    print(f"  - y_test_raw: {y_test_raw.shape}")
    
    # Optionally save the split datasets
    if save_to_splitted:
        save_splitted_datasets(X_train_raw, X_val_raw, X_test_raw, 
                              y_train_raw, y_val_raw, y_test_raw, splitted_path)
    
    return X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw

def save_splitted_datasets(X_train_raw, X_val_raw, X_test_raw, 
                          y_train_raw, y_val_raw, y_test_raw, folder_path: str = '../data/splitted'):
    """
    Save the split RAW datasets to CSV files.
    
    Args:
        X_train_raw, X_val_raw, X_test_raw: Feature datasets
        y_train_raw, y_val_raw, y_test_raw: Target datasets
        folder_path (str): Path to save the datasets
    """
    import os
    
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Save feature datasets
    X_train_raw.to_csv(f'{folder_path}/X_train_raw.csv', index=False)
    X_val_raw.to_csv(f'{folder_path}/X_val_raw.csv', index=False)
    X_test_raw.to_csv(f'{folder_path}/X_test_raw.csv', index=False)
    
    # Save target datasets (preserving original column names)
    pd.DataFrame({y_train_raw.name or 'target': y_train_raw}).to_csv(f'{folder_path}/y_train_raw.csv', index=False)
    pd.DataFrame({y_val_raw.name or 'target': y_val_raw}).to_csv(f'{folder_path}/y_val_raw.csv', index=False)
    pd.DataFrame({y_test_raw.name or 'target': y_test_raw}).to_csv(f'{folder_path}/y_test_raw.csv', index=False)
    
    print(f"📁 Split RAW datasets saved to {folder_path}/")
    print(f"  - X_train_raw.csv: {X_train_raw.shape}")
    print(f"  - X_val_raw.csv: {X_val_raw.shape}")
    print(f"  - X_test_raw.csv: {X_test_raw.shape}")
    print(f"  - y_train_raw.csv: {y_train_raw.shape}")
    print(f"  - y_val_raw.csv: {y_val_raw.shape}")
    print(f"  - y_test_raw.csv: {y_test_raw.shape}")

def load_splitted_datasets(folder_path: str = '../data/splitted') -> tuple:
    """
    Load previously saved split RAW datasets from CSV files.
    
    Args:
        folder_path (str): Path to load the datasets from
        
    Returns:
        tuple: X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw
    """
    import os
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    # Load feature datasets
    X_train_raw = pd.read_csv(f'{folder_path}/X_train_raw.csv')
    X_val_raw = pd.read_csv(f'{folder_path}/X_val_raw.csv')
    X_test_raw = pd.read_csv(f'{folder_path}/X_test_raw.csv')
    
    # Load target datasets
    y_train_raw = pd.read_csv(f'{folder_path}/y_train_raw.csv').squeeze()
    y_val_raw = pd.read_csv(f'{folder_path}/y_val_raw.csv').squeeze()
    y_test_raw = pd.read_csv(f'{folder_path}/y_test_raw.csv').squeeze()
    
    print(f"Split RAW datasets loaded successfully from {folder_path}/")
    print(f"Loaded datasets:")
    print(f"  - X_train_raw: {X_train_raw.shape}")
    print(f"  - X_val_raw: {X_val_raw.shape}")
    print(f"  - X_test_raw: {X_test_raw.shape}")
    print(f"  - y_train_raw: {y_train_raw.shape}")
    print(f"  - y_val_raw: {y_val_raw.shape}")
    print(f"  - y_test_raw: {y_test_raw.shape}")
    
    return X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw

def preprocess_pipeline_train_val(X_train_raw, X_val_raw, y_train_raw, y_val_raw):
    """
    Process only train and validation data (not test) for grid search.
    
    Args:
        X_train_raw, X_val_raw: Raw feature datasets
        y_train_raw, y_val_raw: Raw target datasets
        
    Returns:
        tuple: X_train_processed, X_val_processed, y_train_processed, y_val_processed, transformers
    """
    print("Processing train and validation data only...")
    
    # Combine train and val to fit transformers
    X_combined = pd.concat([X_train_raw, X_val_raw], axis=0, ignore_index=True)
    
    # Create and fit transformers on combined train+val data
    transformers = create_preprocessing_pipeline(X_combined)
    
    # Transform train and val separately
    X_train_processed = pd.DataFrame(
        transformers['transform'](X_train_raw),
        columns=[f'feature_{i}' for i in range(transformers['transform'](X_train_raw).shape[1])]
    )
    X_val_processed = pd.DataFrame(
        transformers['transform'](X_val_raw),
        columns=[f'feature_{i}' for i in range(transformers['transform'](X_val_raw).shape[1])]
    )
    
    # Process targets (just convert to pandas Series with consistent names)
    y_train_processed = pd.Series(y_train_raw.values, name='target')
    y_val_processed = pd.Series(y_val_raw.values, name='target')
    
    print(f"✅ Train+Val data processed:")
    print(f"  X_train_processed: {X_train_processed.shape}")
    print(f"  X_val_processed: {X_val_processed.shape}")
    
    return X_train_processed, X_val_processed, y_train_processed, y_val_processed, transformers

def load_model_and_transformers(models_path: str = '../models'):
    """
    Load trained model and transformers from the models directory.
    
    Args:
        models_path (str): Path to the models directory
        
    Returns:
        tuple: (model, transformers_dict)
    """
    import joblib
    import os
    
    try:
        # Load model
        model_path = os.path.join(models_path, 'final_model.pkl')
        model = joblib.load(model_path)
        
        # Load transformers
        transformers_path = os.path.join(models_path, 'transformers.pkl')
        transformers = load_transformers(transformers_path)
        
        return model, transformers
        
    except Exception as e:
        print(f"Error loading model and transformers: {e}")
        return None, None

def clean_data_for_api(data):
    """
    Clean data for API by replacing NaN values and ensuring JSON compatibility.
    
    Args:
        data: Input data (dict, list, or pandas object)
        
    Returns:
        Cleaned data compatible with JSON serialization
    """
    import pandas as pd
    import numpy as np
    
    if isinstance(data, dict):
        cleaned_data = {}
        for key, value in data.items():
            if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                cleaned_data[key] = 0  # Replace NaN with 0
            elif isinstance(value, (np.integer, np.floating)):
                cleaned_data[key] = float(value)  # Convert numpy types to Python types
            else:
                cleaned_data[key] = value
        return cleaned_data
    
    elif isinstance(data, list):
        return [clean_data_for_api(item) for item in data]
    
    elif isinstance(data, pd.DataFrame):
        # Fill NaN values and convert to dict
        df_cleaned = data.fillna(0)
        return df_cleaned.to_dict(orient='records')
    
    elif pd.isna(data) or (isinstance(data, float) and np.isnan(data)):
        return 0
    
    else:
        return data


