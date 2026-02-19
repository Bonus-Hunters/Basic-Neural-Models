import utils.util
from nn_models.MLP import MLP
from sklearn.metrics import accuracy_score
import numpy as np

X_train, X_test, y_train, y_test = utils.util.get_data_multiclass()

# Use simpler architecture and proper parameters
mlp = MLP(neurons_num=[2,2,2],  # Simpler architecture
          learning_rate=0.01, 
          epochs=500, 
          activation="sigmoid", 
          use_bias=True
          )  

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print("Predictions:", y_pred)
print("True labels:", y_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")