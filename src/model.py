import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import joblib
import mlflow
import mlflow.sklearn


class SalesModel:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'forest': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        self.best_model = None
        self.best_name = None

        mlflow.set_experiment("Sales_Forcasting")
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train models and pick the best one"""
        results = {}
        
        for name, model in self.models.items():
            mlflow.end_run()
            with mlflow.start_run(run_name=f"{name}_model"):
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    pred = np.maximum(pred, 0)  # No negative sales
                    
                    mae = mean_absolute_error(y_test, pred)
                    mse = mean_squared_error(y_test, pred)
                    r2=r2_score(y_test, pred)
                    results[name] = {
                        'MAE': mae,
                        'MSE': mse,
                        'R²': r2
                    }

                    # Log common parameters for ALL models
                    mlflow.log_param("model_type", name)
                    mlflow.log_param("train_samples", len(X_train))
                    mlflow.log_param("test_samples", len(X_test))
                    
                    # Log common metrics for ALL models
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("r2", r2)
                    
                    # Log model for ALL models
                    mlflow.sklearn.log_model(model, f"{name}_model")

                    # Log model-specific parameters
                    if name == 'forest':
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("random_state", model.random_state)
                    elif name == 'linear':
                        mlflow.log_param("fit_intercept", model.fit_intercept)
                    
                    print(f"{name}: MAE = {mae}")
                    print(f"{name}: MSE = {mse}")
                    print(f"{name}: R² = {r2}")

                except Exception as e:
                    print(f"Error training {name}: {e}")
                    mlflow.log_param("error", str(e))
                
        # Pick best model
        self.best_name = min(results, key=lambda x: results[x]['MAE'])
        self.best_model = self.models[self.best_name]
        
        # Log best model info
        mlflow.end_run()
        
        with mlflow.start_run(run_name="best_model_summary"):
            mlflow.log_param("best_model", self.best_name)
            mlflow.log_metric("best_mae", results[self.best_name]['MAE'])
            mlflow.sklearn.log_model(self.best_model, "best_model")

            # end
            mlflow.end_run()

        print("Training complete!")
        print(f"Best model: {self.best_name}")
        return results
    
    def predict(self, X):
        """Make predictions"""
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        pred = self.best_model.predict(X)
        return np.maximum(pred, 0)
    
    def save(self):
        """Save the best model"""
        try:
            joblib.dump(self.best_model, './models/best_model.pkl')
            print("Model saved!")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise