import itertools
import os
import numpy as np
import pandas as pd
from SLP import Perceptron
from adaline import Adaline
import util
from sklearn.model_selection import train_test_split
from helper import *
# Load the data
df = pd.read_csv(util.get_data_path())

# Define features and classes
features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'Origin_Biscoe', 'Origin_Dream', 'Origin_Torgersen']
classes = ['Adelie', 'Chinstrap', 'Gentoo']

def scale_features(X):
    """Scale features to have zero mean and unit variance"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero for constant features
    std = np.where(std == 0, 1, std)
    return (X - mean) / std, mean, std

def prepare_data(df, class_pair, feature_pair):
    """
    Prepare data for binary classification with selected features
    """
    # Filter data for the two classes
    df_filtered = df[df['Species'].isin(class_pair)].copy()
    
    # Create binary labels (1 for first class, -1 for second class)
    df_filtered['binary_label'] = df_filtered['Species'].apply(lambda x: 1 if x == class_pair[0] else -1)
    
    # Select features and convert boolean columns to integers (0/1)
    X = df_filtered[list(feature_pair)].copy()
    
    # Convert boolean columns to integers
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
        
    X_values = X.values.astype(float)

    y = df_filtered['binary_label']
    X_train, X_test , y_train, y_test = train_test_split(X_values,y,test_size=0.4,stratify=y, random_state= 42)
    return X_train,X_test,y_train,y_test

def apply_scaling(X, mean, std):
    """Apply scaling to new data using precomputed mean and std"""
    return (X - mean) / std

def save_weights(model, path, feature_names=None, scaling_params=None):
    """
    Save model weights to file
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as file:
            if model.weights is not None:
                file.write("Weights:\n")
                for i, weight in enumerate(model.weights):
                    feature_name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature_{i}"
                    file.write(f"  {feature_name}: {weight:.6f}\n")
            
            if hasattr(model, 'bias'):
                file.write(f"Bias: {model.bias:.6f}\n")
                
            if scaling_params:
                file.write("\nScaling Parameters:\n")
                file.write(f"Means: {[f'{m:.6f}' for m in scaling_params['mean']]}\n")
                file.write(f"Std: {[f'{s:.6f}' for s in scaling_params['std']]}\n")
                
        return True
    except Exception as e:
        print(f"Error saving weights to {path}: {e}")
        return False

def train_and_save_models():
    """
    Train models on all combinations and save weights in structured directories
    """
    # Generate all combinations of 2 classes from 3
    class_combinations = list(itertools.combinations(classes, 2))
    
    # Generate all combinations of 2 features from available features
    feature_combinations = list(itertools.combinations(features, 2))
    
    # Create models with smaller learning rates for stability
    models = {
        'SLP': Perceptron(learning_rate=0.01, max_iterations=1000, use_bias=True),
        'Adaline': Adaline(learning_rate=0.001, max_iterations=2000, use_bias=True, acceptable_error=0.01)  # Smaller LR for Adaline
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
        for class_pair in class_combinations:
            print(f"  Classes: {class_pair}")
            
            for feature_pair in feature_combinations:
                feature_names = [f.replace('Origin_', '') for f in feature_pair]
                print(f"    Features: {feature_names}")
                

                
                X_train,X_test,y_train,y_test = prepare_data(df, class_pair, feature_pair)

                X_train, mean, std = scale_features(X_train)

                scaling_params = {'mean': mean, 'std': std}

                
                # Train model
                try:
                    model.train(X_train, y_train)
                    
                    # Check for NaN values in weights
                    if model.weights is not None and np.any(np.isnan(model.weights)):
                        print(f"      [FAILED] NaN values in weights")
                        continue
                    
                    # Create directory path
                    dir_path = f"{model_name}/class_{class_pair[0]}_{class_pair[1]}/feature_{feature_names[0]}_{feature_names[1]}"
                    
                    # Save weights
                    weights_path = os.path.join(dir_path, "weights.txt")
                    success = save_weights(model, weights_path, feature_names, scaling_params)
                    
                    if success:
                        # Store results
                        results.append({
                            'model': model_name,
                            'classes': class_pair,
                            'features': feature_pair,
                            'weights_path': weights_path,
                            'weights': model.weights.copy() if model.weights is not None else None,
                            'bias': model.bias if hasattr(model, 'bias') else None,
                            'scaling_params': scaling_params
                        })
                        
                        print(f"      [SUCCESS] Weights saved to: {weights_path}")
                    else:
                        print(f"      [FAILED] Could not save weights to: {weights_path}")
                    
                except Exception as e:
                    print(f"      [ERROR] Training failed: {str(e)}")
    
    return results

def evaluate_models(results, df):
    """
    Evaluate all trained models and print performance metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    for result in results:
        model_name = result['model']
        class_pair = result['classes']
        feature_pair = result['features']
        scaling_params = result.get('scaling_params')
        
       
        X_train,X_test,y_train,y_test = prepare_data(df, class_pair, feature_pair)
        X_test = apply_scaling(X_test, scaling_params['mean'], scaling_params['std'])

        
        # Recreate model for prediction
        if model_name == 'SLP':
            model = Perceptron()
        else:
            model = Adaline()
        
        model.weights = result['weights']
        if hasattr(model, 'bias'):
            model.bias = result['bias']
        
        cm, accuracy = test_classifier(model,X_test,y_test)

        # plot_confusion_matrix(cm)

        # Simplify feature names for display
        feature_names = [f.replace('Origin_', '') for f in feature_pair]
        
        print(f"{model_name:8} | Classes {class_pair} | Features {feature_names}")
        print(f"  Accuracy: {accuracy:.4f}")
        if result['weights'] is not None:
            print(f"  Weights: {[f'{w:.4f}' for w in result['weights']]}")
        print(f"  Bias: {result['bias']:.4f}")
        print(f"  Path: {result['weights_path']}")
        print()

def plot_model(index= 0, results= []):

    assert(index<len(results))
    # Get the first result
    first_result = results[index]

    # Extract relevant values
    weights = first_result['weights']
    bias = first_result['bias']
    class_pair = first_result['classes']
    feature_pair = first_result['features']
    scaling_params = first_result['scaling_params']
    X_train,X_test,y_train,y_test = prepare_data(df,class_pair,feature_pair)
    X_train = apply_scaling(X_train,scaling_params['mean'],scaling_params['std'])
    # Now call your plot function
    plot_decision_boundary(X_train, y_train, weights, bias, feature_pair, class_pair)



if __name__ == "__main__":
    # Train all models and save weights
    results = train_and_save_models()
    for i in range(len(results)):
        plot_model(i,results)
    # Evaluate and print summary
    evaluate_models(results, df)
    
    print("\nTraining completed! All weights have been saved in the structured directories.")
    
    # Print summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total combinations trained: {len(results)}")
    
    models_summary = {}
    for result in results:
        model_name = result['model']
        if model_name not in models_summary:
            models_summary[model_name] = 0
        models_summary[model_name] += 1
    
    for model_name, count in models_summary.items():
        print(f"{model_name}: {count} combinations")
    
    total_combinations = 2 * len(list(itertools.combinations(classes, 2))) * len(list(itertools.combinations(features, 2)))
    print(f"Total possible combinations: {total_combinations}")
    print(f"Success rate: {len(results)}/{total_combinations} ({len(results)/total_combinations*100:.1f}%)")