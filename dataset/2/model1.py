import pandas as pd
import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

class ModelProcessor:
    def __init__(self, data_path, target_variable):
        self.data = pd.read_csv(data_path)
        self.target_variable = target_variable

    def prepare_data(self):
        X = self.data.drop(columns=[self.target_variable])
        y = self.data[self.target_variable]
        X = pd.get_dummies(X, drop_first=True)
        return X, y

    def train_and_evaluate(self):
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = SGDClassifier(
            loss='log_loss',
            max_iter=2000,
            tol=1e-3,
            penalty='l2',
            alpha=0.0001,
            random_state=42
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        
        print(f"Accuracy: {accuracy}")
        return accuracy

def main():
    parser = argparse.ArgumentParser(description='Custom Model for Dataset 2')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--target', type=str, required=True, help='Target variable name')
    args = parser.parse_args()
    
    processor = ModelProcessor(args.data, args.target)
    processor.train_and_evaluate()

if __name__ == "__main__":
    main()
