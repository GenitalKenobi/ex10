import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.datasets import load_iris
@click.group()
def cli():
 """Machine Learning Command Line Workflows"""
 pass
@click.command()
@click.option('--test_size', default=0.2, help='Test set size (default: 0.2)')
@click.option('--n_estimators', default=100, help='Number of estimators in RandomForest (default: 100)')
@click.option('--max_depth', default=None, help='Maximum depth of the tree(default: None)')
@click.option('--model_output', default='model.pkl', help='Output filename for the trained model (default: model.pkl)')
def train(test_size, n_estimators, max_depth, model_output):
 """Train a machine learning model on the Iris dataset."""
 # Ensure max_depth is an integer or None
 max_depth = int(max_depth) if max_depth is not None else None
 # Load the Iris dataset
 iris = load_iris()
 X = iris.data
 y = iris.target
 # Split data into training and testing sets
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
random_state=42)
 # Train a model (Random Forest)
 model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
 model.fit(X_train, y_train)
 # Save the trained model
 joblib.dump(model, model_output)
 print(f"Model trained and saved to {model_output}")
@click.command()
@click.option('--output_file', default='output_predictions.csv', help='Output filename for predictions (default: output_predictions.csv)')
@click.option('--model_path', default = 'model.pkl', help = 'Input filename for the model (default: model.pkl)')
def predict(output_file, model_path):
 """Make predictions using a trained model on the Iris dataset."""
 # Load the trained model
 try:
  model = joblib.load(model_path)
 except FileNotFoundError:
  print(f"Error: Trained model '{model_path}' not found. Please train a model first using the 'train' command.")
  return
 # Load the Iris dataset
 iris = load_iris()
 X = iris.data
 # Make predictions
 predictions = model.predict(X)
 # Save predictions to an output file
 pd.DataFrame({'prediction': predictions}).to_csv(output_file, index=False)
 print(f"Predictions saved to {output_file}")
cli.add_command(train)
cli.add_command(predict)
if __name__ == "__main__":
 cli()