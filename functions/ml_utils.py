"""
Machine Learning utility functions for model training, evaluation, and grid search.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# Machine Learning imports
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# Import models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import pickle
import joblib
import os

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def detect_problem_type(y_train: pd.Series, target_type: str = 'auto') -> str:
    """
    Automatically detect the problem type based on target variable.
    
    Args:
        y_train: Training target variable
        target_type: 'auto', 'classification', 'regression'
        
    Returns:
        str: 'binary_classification', 'multiclass_classification', or 'regression'
    """
    if target_type != 'auto':
        if target_type == 'classification':
            unique_vals = y_train.nunique()
            return 'binary_classification' if unique_vals == 2 else 'multiclass_classification'
        else:
            return 'regression'
    
    # Auto detection
    is_numeric = pd.api.types.is_numeric_dtype(y_train)
    unique_vals = y_train.nunique()
    
    if not is_numeric or unique_vals <= 10:
        # Categorical or few unique values -> Classification
        return 'binary_classification' if unique_vals == 2 else 'multiclass_classification'
    else:
        # Many unique numeric values -> Regression
        return 'regression'

def encode_target(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, 
                  problem_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode target variables if they are categorical.
    
    Returns:
        Encoded targets and the encoder (None if no encoding needed)
    """
    if 'classification' in problem_type and not pd.api.types.is_numeric_dtype(y_train):
        print("Encoding categorical target variable...")
        label_encoder = LabelEncoder()
        
        # Fit on all target data to ensure consistent encoding
        all_targets = pd.concat([y_train, y_val, y_test])
        label_encoder.fit(all_targets)
        
        y_train_encoded = label_encoder.transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
        
        print(f"Target encoding mapping:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name} -> {i}")
            
        return y_train_encoded, y_val_encoded, y_test_encoded, label_encoder
    else:
        print("No target encoding needed.")
        return y_train.values, y_val.values, y_test.values, None

def get_models_config(problem_type: str, random_state: int = 42) -> Dict[str, Dict]:
    """
    Get appropriate models and parameter grids based on problem type.
    
    Returns:
        dict: Models and their parameter grids
    """
    if problem_type == 'binary_classification':
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
        
    elif problem_type == 'multiclass_classification':
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'KNeighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            }
        }
        
    else:  # regression
        models = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
    
    return models

def get_scoring_metric(problem_type: str) -> str:
    """Get appropriate scoring metric for the problem type."""
    if problem_type == 'binary_classification':
        return 'f1'
    elif problem_type == 'multiclass_classification':
        return 'f1_macro'
    else:  # regression
        return 'neg_mean_squared_error'

def get_cv_strategy(problem_type: str, y_train: np.ndarray, cv_folds: int = 5, 
                   random_state: int = 42):
    """Get appropriate cross-validation strategy."""
    if 'classification' in problem_type:
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

def grid_search(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, 
                y_val: np.ndarray, models_config: Dict, scoring_metric: str, 
                cv_strategy, problem_type: str, n_jobs: int = -1) -> Tuple[Dict, str, Any]:
    """
    Perform grid search for all models and return results.
    
    Returns:
        Tuple: (results_dict, best_model_name, best_model)
    """
    results = {}
    best_score = -np.inf if 'neg_' not in scoring_metric else np.inf
    best_model_name = None
    best_model = None
    
    print("Starting Grid Search...")
    print("=" * 50)
    
    for model_name, config in models_config.items():
        print(f"\n🔍 Training {model_name}...")
        
        # Prepare data for grid search (train only)
        if len(config['params']) > 0:
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                scoring=scoring_metric,
                cv=cv_strategy,
                n_jobs=n_jobs,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model and score
            best_model_instance = grid_search.best_estimator_
            train_score = grid_search.best_score_
            best_params = grid_search.best_params_
        else:
            # No hyperparameters to tune (e.g., LinearRegression)
            model_instance = config['model']
            model_instance.fit(X_train, y_train)
            
            # Calculate CV score manually
            cv_scores = cross_val_score(model_instance, X_train, y_train, 
                                      cv=cv_strategy, scoring=scoring_metric)
            train_score = cv_scores.mean()
            best_model_instance = model_instance
            best_params = {}
        
        # Evaluate on validation set
        val_predictions = best_model_instance.predict(X_val)
        
        # Calculate validation score
        if 'classification' in problem_type:
            if problem_type == 'binary_classification':
                val_score = f1_score(y_val, val_predictions)
            else:  # multiclass
                val_score = f1_score(y_val, val_predictions, average='macro')
        else:  # regression
            val_score = -mean_squared_error(y_val, val_predictions)  # Negative for consistency
        
        # Store results
        results[model_name] = {
            'model': best_model_instance,
            'params': best_params,
            'train_score': train_score,
            'val_score': val_score,
            'val_predictions': val_predictions
        }
        
        print(f"✅ {model_name} completed:")
        print(f"   Best params: {best_params}")
        print(f"   Train score (CV): {train_score:.4f}")
        print(f"   Validation score: {val_score:.4f}")
        
        # Update best model
        comparison_score = val_score
        if (('neg_' not in scoring_metric and comparison_score > best_score) or 
            ('neg_' in scoring_metric and comparison_score > best_score)):
            best_score = comparison_score
            best_model_name = model_name
            best_model = best_model_instance
    
    print("\n" + "=" * 50)
    print(f"🏆 Best model: {best_model_name}")
    print(f"🏆 Best validation score: {best_score:.4f}")
    
    return results, best_model_name, best_model

def plot_model_comparison(grid_results: Dict, best_model_name: str) -> None:
    """Plot comparison of model performances with dark theme."""
    # Create results summary
    results_df = pd.DataFrame({
        'Model': list(grid_results.keys()),
        'Train_Score': [results['train_score'] for results in grid_results.values()],
        'Validation_Score': [results['val_score'] for results in grid_results.values()]
    })

    print("Model Comparison Results:")
    print("=" * 40)
    print(results_df.round(4))

    # Plot results comparison
    plt.figure(figsize=(12, 6), facecolor='#2E2E2E')
    colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD1DC']

    x_pos = np.arange(len(results_df))
    width = 0.35

    plt.bar(x_pos - width/2, results_df['Train_Score'], width, 
            label='Train Score (CV)', color=colors[0], alpha=0.8)
    plt.bar(x_pos + width/2, results_df['Validation_Score'], width, 
            label='Validation Score', color=colors[1], alpha=0.8)

    plt.xlabel('Models', color='white')
    plt.ylabel('Score', color='white')
    plt.title('Model Performance Comparison', color='white', pad=20)
    plt.xticks(x_pos, results_df['Model'], rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Highlight best model
    best_idx = results_df['Model'].tolist().index(best_model_name)
    plt.axvline(x=best_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)

    plt.tight_layout()
    plt.show()

def plot_learning_curves(model, X_train: pd.DataFrame, y_train: np.ndarray, 
                         scoring_metric: str, cv_strategy, problem_type: str) -> None:
    """
    Plot learning curves to analyze model performance with different training set sizes.
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes,
        cv=cv_strategy,
        scoring=scoring_metric,
        n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6), facecolor='#2E2E2E')
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size', color='white')
    plt.ylabel('Score', color='white')
    plt.title('Learning Curves', color='white', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance_advanced(final_model: Any, X_train: pd.DataFrame, 
                                   top_n: int = 15) -> None:
    """
    Advanced feature importance visualization with better formatting.
    """
    if hasattr(final_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8), facecolor='#2E2E2E')
        top_features = feature_importance.head(top_n)
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=colors, alpha=0.8)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(top_features.iterrows()):
            plt.text(row['importance'] + max(top_features['importance']) * 0.01, 
                    i, f'{row["importance"]:.3f}', 
                    va='center', color='white', fontsize=10)
        
        plt.yticks(range(len(top_features)), top_features['feature'], color='white')
        plt.xlabel('Feature Importance', color='white', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', color='white', pad=20, fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
        
    elif hasattr(final_model, 'coef_'):
        # For linear models
        if len(final_model.coef_.shape) == 1:  # Binary classification or regression
            coefficients = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': final_model.coef_,
                'abs_coefficient': np.abs(final_model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            # Plot coefficients
            plt.figure(figsize=(12, 8), facecolor='#2E2E2E')
            top_coefs = coefficients.head(top_n)
            
            # Color based on positive/negative
            colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in top_coefs['coefficient']]
            
            bars = plt.barh(range(len(top_coefs)), top_coefs['coefficient'], 
                           color=colors, alpha=0.8)
            
            # Add value labels
            for i, (idx, row) in enumerate(top_coefs.iterrows()):
                plt.text(row['coefficient'] + (max(abs(top_coefs['coefficient'])) * 0.01 * 
                        (1 if row['coefficient'] > 0 else -1)), 
                        i, f'{row["coefficient"]:.3f}', 
                        va='center', color='white', fontsize=10)
            
            plt.yticks(range(len(top_coefs)), top_coefs['feature'], color='white')
            plt.xlabel('Coefficient Value', color='white', fontsize=12)
            plt.title(f'Top {top_n} Model Coefficients', color='white', pad=20, fontsize=14)
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.axvline(x=0, color='white', linestyle='-', alpha=0.5)
            plt.tight_layout()
            plt.show()
            
            return coefficients
    
    return None

def create_model_summary_report(ml_results: Dict, save_path: str = None) -> pd.DataFrame:
    """
    Create a comprehensive summary report of the ML pipeline results.
    """
    report_data = {
        'Metric': [],
        'Value': [],
        'Description': []
    }
    
    # Basic info
    report_data['Metric'].extend(['Problem Type', 'Best Model', 'Features Count'])
    report_data['Value'].extend([
        ml_results['problem_type'],
        ml_results['best_model_name'],
        len(ml_results['grid_results'][ml_results['best_model_name']]['model'].feature_names_in_) 
        if hasattr(ml_results['grid_results'][ml_results['best_model_name']]['model'], 'feature_names_in_') 
        else 'N/A'
    ])
    report_data['Description'].extend([
        'Type of ML problem detected',
        'Model with best validation performance',
        'Number of input features'
    ])
    
    # Model performance
    test_results = ml_results['test_results']
    if 'classification' in ml_results['problem_type']:
        report_data['Metric'].extend(['Test Accuracy', 'Test F1-Score', 'Test Precision', 'Test Recall'])
        report_data['Value'].extend([
            f"{test_results['accuracy']:.4f}",
            f"{test_results['f1_score']:.4f}",
            f"{test_results['precision']:.4f}",
            f"{test_results['recall']:.4f}"
        ])
        report_data['Description'].extend([
            'Final accuracy on test set',
            'Final F1-score on test set',
            'Final precision on test set',
            'Final recall on test set'
        ])
    else:
        report_data['Metric'].extend(['Test R²', 'Test MAE', 'Test MSE'])
        report_data['Value'].extend([
            f"{test_results['r2_score']:.4f}",
            f"{test_results['mae']:.4f}",
            f"{test_results['mse']:.4f}"
        ])
        report_data['Description'].extend([
            'Final R² score on test set',
            'Final Mean Absolute Error on test set',
            'Final Mean Squared Error on test set'
        ])
    
    # Best parameters
    best_params = ml_results['best_params']
    for param, value in best_params.items():
        report_data['Metric'].append(f'Best {param}')
        report_data['Value'].append(str(value))
        report_data['Description'].append(f'Optimal {param} parameter')
    
    # Create DataFrame
    summary_df = pd.DataFrame(report_data)
    
    # Save if path provided
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"📄 Summary report saved to: {save_path}")
    
    return summary_df

def train_final_model(best_model_name: str, best_params: Dict, problem_type: str, 
                      X_train: pd.DataFrame, X_val: pd.DataFrame, 
                      y_train: np.ndarray, y_val: np.ndarray, 
                      random_state: int = 42) -> Any:
    """
    Retrain the best model on combined train+validation data.
    
    Returns:
        Trained final model
    """
    # Combine train and validation data for final training
    X_train_final = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_final = np.concatenate([y_train, y_val])

    print(f"Final training data shape:")
    print(f"X_train_final: {X_train_final.shape}")
    print(f"y_train_final: {y_train_final.shape}")

    print(f"\nRetraining {best_model_name} with best parameters:")
    print(f"Best parameters: {best_params}")

    # Create fresh model instance with best parameters
    if best_model_name == 'RandomForest':
        if 'classification' in problem_type:
            final_model = RandomForestClassifier(random_state=random_state, **best_params)
        else:
            final_model = RandomForestRegressor(random_state=random_state, **best_params)
    elif best_model_name == 'LogisticRegression':
        final_model = LogisticRegression(random_state=random_state, max_iter=1000, **best_params)
    elif best_model_name == 'GradientBoosting':
        if 'classification' in problem_type:
            final_model = GradientBoostingClassifier(random_state=random_state, **best_params)
        else:
            final_model = GradientBoostingRegressor(random_state=random_state, **best_params)
    elif best_model_name == 'LinearRegression':
        final_model = LinearRegression(**best_params)
    elif best_model_name == 'KNeighbors':
        if 'classification' in problem_type:
            final_model = KNeighborsClassifier(**best_params)
        else:
            final_model = KNeighborsRegressor(**best_params)

    # Train final model
    print("Training final model on combined data...")
    final_model.fit(X_train_final, y_train_final)
    
    return final_model

def evaluate_model(final_model: Any, X_test: pd.DataFrame, y_test: np.ndarray, 
                   problem_type: str, label_encoder: LabelEncoder = None) -> Dict:
    """
    Evaluate the final model on test set.
    
    Returns:
        Dictionary with evaluation metrics
    """
    test_predictions = final_model.predict(X_test)

    print("\n🎯 Final Model Performance on Test Set:")
    print("=" * 40)

    results = {}

    if 'classification' in problem_type:
        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions, average='macro')
        recall = recall_score(y_test, test_predictions, average='macro')
        f1 = f1_score(y_test, test_predictions, average='macro')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1-score (macro): {f1:.4f}")
        
        # Classification report
        if label_encoder:
            class_names = label_encoder.classes_
        else:
            class_names = None
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, test_predictions, target_names=class_names))
        
    else:  # regression
        mse = mean_squared_error(y_test, test_predictions)
        mae = mean_absolute_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        
        results = {
            'mse': mse,
            'mae': mae,
            'r2_score': r2
        }
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
    
    return results

def save_model_artifacts(final_model: Any, best_model_name: str, problem_type: str, 
                        best_params: Dict, X_train: pd.DataFrame, grid_results: Dict,
                        test_results: Dict, label_encoder: LabelEncoder,
                        models_path: str = '../models') -> None:
    """
    Save the trained model and all necessary artifacts for production use.
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_path, exist_ok=True)

    # Save the final model
    model_path = f"{models_path}/final_model.pkl"
    joblib.dump(final_model, model_path)

    # Save model metadata
    model_metadata = {
        'model_name': best_model_name,
        'model_type': type(final_model).__name__,
        'problem_type': problem_type,
        'best_params': best_params,
        'features': X_train.columns.tolist(),
        'target_encoder': label_encoder,
        'training_score': grid_results[best_model_name]['train_score'],
        'validation_score': grid_results[best_model_name]['val_score'],
        'test_performance': test_results,
        'grid_search_results': grid_results
    }

    metadata_path = f"{models_path}/model_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(model_metadata, f)

    print("✅ Model and artifacts saved successfully!")
    print(f"📁 Final model: {model_path}")
    print(f"📁 Model metadata: {metadata_path}")

    print(f"\n📊 Model Summary:")
    print(f"   Model: {best_model_name}")
    print(f"   Problem type: {problem_type}")
    print(f"   Features: {len(X_train.columns)}")

def ml_pipeline(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
                target_type: str = 'auto', cv_folds: int = 5, 
                random_state: int = 42, n_jobs: int = -1,
                models_path: str = '../models') -> Dict:
    """
    Complete machine learning pipeline from problem detection to model saving.
    
    Returns:
        Dictionary with all results and artifacts
    """
    print("🚀 Starting ML Pipeline...")
    print("=" * 50)
    
    # 1. Problem type detection
    problem_type = detect_problem_type(y_train, target_type)
    print(f"✅ Detected problem type: {problem_type}")
    
    # 2. Target encoding
    y_train_proc, y_val_proc, y_test_proc, label_encoder = encode_target(
        y_train, y_val, y_test, problem_type
    )
    
    # 3. Get models configuration
    models_config = get_models_config(problem_type, random_state)
    scoring_metric = get_scoring_metric(problem_type)
    cv_strategy = get_cv_strategy(problem_type, y_train_proc, cv_folds, random_state)
    
    print(f"✅ Models to evaluate: {list(models_config.keys())}")
    print(f"✅ Scoring metric: {scoring_metric}")
    
    # 4. Grid search
    grid_results, best_model_name, best_model = grid_search(
        X_train, y_train_proc, X_val, y_val_proc, 
        models_config, scoring_metric, cv_strategy, problem_type, n_jobs
    )
    
    # 5. Plot comparison
    plot_model_comparison(grid_results, best_model_name)
    
    # 6. Train final model
    final_model = train_final_model(
        best_model_name, grid_results[best_model_name]['params'], 
        problem_type, X_train, X_val, y_train_proc, y_val_proc, random_state
    )
    
    # 7. Evaluate on test set
    test_results = evaluate_model(final_model, X_test, y_test_proc, problem_type, label_encoder)
    
    # 8. Save artifacts
    save_model_artifacts(
        final_model, best_model_name, problem_type, 
        grid_results[best_model_name]['params'], X_train, 
        grid_results, test_results, label_encoder, models_path
    )
    
    print("\n🎉 ML Pipeline completed successfully!")
    
    return {
        'final_model': final_model,
        'problem_type': problem_type,
        'best_model_name': best_model_name,
        'best_params': grid_results[best_model_name]['params'],
        'grid_results': grid_results,
        'test_results': test_results,
        'label_encoder': label_encoder
    }

def create_test_set_analysis(final_model: Any, X_test: pd.DataFrame, y_test: np.ndarray, 
                           problem_type: str, label_encoder: LabelEncoder = None) -> None:
    """
    Create comprehensive test set analysis with confusion matrix or regression plots.
    """
    test_predictions = final_model.predict(X_test)
    
    print("🎯 Test Set Performance Analysis:")
    print("=" * 40)
    
    if 'classification' in problem_type:
        # Classification analysis
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Prepare targets for evaluation
        if label_encoder is not None:
            y_test_encoded = label_encoder.transform(pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test)
            class_names = label_encoder.classes_
        else:
            y_test_encoded = y_test if isinstance(y_test, np.ndarray) else y_test.values
            class_names = None
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_encoded, test_predictions)
        
        plt.figure(figsize=(10, 8), facecolor='#2E2E2E')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Test Set', color='white', fontsize=14, pad=20)
        plt.xlabel('Predicted', color='white')
        plt.ylabel('Actual', color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.tight_layout()
        plt.show()
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print("=" * 50)
        print(classification_report(y_test_encoded, test_predictions, 
                                  target_names=class_names))
        
    else:
        # Regression analysis
        y_test_values = y_test if isinstance(y_test, np.ndarray) else y_test.values
        residuals = y_test_values - test_predictions
        
        plt.figure(figsize=(15, 5), facecolor='#2E2E2E')
        
        # Residuals plot
        plt.subplot(1, 3, 1)
        plt.scatter(test_predictions, residuals, alpha=0.6, color='#FFB3BA')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Predicted Values', color='white')
        plt.ylabel('Residuals', color='white')
        plt.title('Residuals Plot', color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.grid(True, alpha=0.3)
        
        # Actual vs Predicted
        plt.subplot(1, 3, 2)
        plt.scatter(y_test_values, test_predictions, alpha=0.6, color='#BAFFC9')
        plt.plot([y_test_values.min(), y_test_values.max()], 
                 [y_test_values.min(), y_test_values.max()], 'r--', alpha=0.7)
        plt.xlabel('Actual Values', color='white')
        plt.ylabel('Predicted Values', color='white')
        plt.title('Actual vs Predicted', color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.grid(True, alpha=0.3)
        
        # Residuals distribution
        plt.subplot(1, 3, 3)
        plt.hist(residuals, bins=20, alpha=0.7, color='#BAE1FF', edgecolor='black')
        plt.xlabel('Residuals', color='white')
        plt.ylabel('Frequency', color='white')
        plt.title('Residuals Distribution', color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 Residuals Statistics:")
        print(f"Mean Residual: {residuals.mean():.4f}")
        print(f"Std Residual: {residuals.std():.4f}")
        print(f"Min Residual: {residuals.min():.4f}")
        print(f"Max Residual: {residuals.max():.4f}")

def show_feature_importance_detailed(final_model: Any, feature_names: list, top_n: int = 15) -> None:
    """
    Show detailed feature importance analysis with advanced visualization.
    """
    print("🔍 Feature Importance Analysis:")
    print("=" * 40)
    
    if hasattr(final_model, 'feature_importances_'):
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"📊 Top {min(top_n, len(feature_importance))} Most Important Features:")
        display(feature_importance.head(top_n).round(4))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8), facecolor='#2E2E2E')
        top_features = feature_importance.head(top_n)
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=colors, alpha=0.8)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(top_features.iterrows()):
            plt.text(row['importance'] + max(top_features['importance']) * 0.01, 
                    i, f'{row["importance"]:.3f}', 
                    va='center', color='white', fontsize=10)
        
        plt.yticks(range(len(top_features)), top_features['feature'], color='white')
        plt.xlabel('Feature Importance', color='white', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', color='white', pad=20, fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        
    elif hasattr(final_model, 'coef_'):
        # For linear models
        if len(final_model.coef_.shape) == 1:  # Binary classification or regression
            coefficients = pd.DataFrame({
                'feature': feature_names,
                'coefficient': final_model.coef_,
                'abs_coefficient': np.abs(final_model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            print(f"📊 Top {min(top_n, len(coefficients))} Feature Coefficients (by absolute value):")
            display(coefficients.head(top_n).round(4))
            
            # Plot coefficients
            plt.figure(figsize=(12, 8), facecolor='#2E2E2E')
            top_coef = coefficients.head(top_n)
            
            colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in top_coef['coefficient']]
            
            bars = plt.barh(range(len(top_coef)), top_coef['coefficient'], 
                           color=colors, alpha=0.8)
            
            # Add value labels
            for i, (idx, row) in enumerate(top_coef.iterrows()):
                plt.text(row['coefficient'] + (max(abs(top_coef['coefficient'])) * 0.01 * 
                        (1 if row['coefficient'] > 0 else -1)), 
                        i, f'{row["coefficient"]:.3f}', 
                        va='center', color='white', fontsize=10)
            
            plt.yticks(range(len(top_coef)), top_coef['feature'], color='white')
            plt.xlabel('Coefficient Value', color='white', fontsize=12)
            plt.title(f'Top {top_n} Model Coefficients', color='white', pad=20, fontsize=14)
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.axvline(x=0, color='white', linestyle='-', alpha=0.5)
            plt.tight_layout()
            plt.show()
    else:
        print("ℹ️ Feature importance not available for this model type.")
        print(f"Model type: {type(final_model).__name__}")

def print_final_summary(problem_type: str, best_model_name: str, best_params: Dict, 
                       test_results: Dict, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                       X_test: pd.DataFrame, models_path: str) -> None:
    """
    Print a comprehensive final summary of the ML pipeline.
    """
    print("🎉 FINAL ML PIPELINE SUMMARY")
    print("=" * 50)
    print(f"📊 Problem Type: {problem_type}")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📈 Features: {X_train.shape[1]}")
    print(f"📦 Dataset Split:")
    print(f"   - Training: {X_train.shape[0]} samples")
    print(f"   - Validation: {X_val.shape[0]} samples") 
    print(f"   - Test: {X_test.shape[0]} samples")
    print(f"   - Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} samples")

    print(f"\n📈 Best Parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    print(f"\n📊 Final Test Performance:")
    for metric, value in test_results.items():
        if value is not None:
            print(f"   {metric}: {value:.4f}")

    print(f"\n💾 Saved Artifacts:")
    print(f"   📁 Model: {models_path}/final_model.pkl")
    print(f"   📁 Metadata: {models_path}/model_metadata.pkl")
    
    print("\n✅ Pipeline completed successfully!")
    print("🚀 Model ready for production deployment!")
