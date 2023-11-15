import click
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split

@click.group()
def cli():
    """Machine Learning CLI Workflow"""
    pass
@click.command()
@click.option('--test_size', default=0.2, help='Test size (default: 0.2)')
@click.option('--n_estimators', default=100, help='Number of trees in classifier (default: 100)')
def train(test_size, n_estimators):
    """Train the Random Forest model on the Iris Dataset"""
    df = load_iris()
    X = df.data
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.joblib')
    print("Model saved as model.joblib")

@click.command()
@click.option('--sl', help='Sepal Length')
@click.option('--sw', help='Sepal Width')
@click.option('--pl', help='Petal Length')
@click.option('--pw', help='Petal Width')
def predict(sl, sw, pl, pw):
    model = joblib.load('model.joblib')
    classes = {0:'Iris Setosa', 1:'Iris Versicolour', 2:'Iris Virginica'}
    print("Prediction:", classes[model.predict([[sl, sw, pl, pw]])[0]])

cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli()