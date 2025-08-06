import pandas as pd
from process import *
import mlflow
from model import SalesModel

def run_pipeline():
    """Main forecasting pipeline"""
    print("=== Sales Forecasting Pipeline ===\n")
    mlflow.end_run() 
    # Start main pipeline run
    with mlflow.start_run(run_name="sales_forecasting_pipeline"):

        # 1. Load and process data
        print("1. Loading data...")
        df = load_data()
        print(df.shape)
        daily_sales = create_daily_sales(df)

        # Log data info
        mlflow.log_param("total_days", len(daily_sales))
        mlflow.log_param("date_start", str(daily_sales['date'].min()))
        mlflow.log_param("date_end", str(daily_sales['date'].max()))
        mlflow.log_metric("avg_daily_sales", daily_sales['total_sales'].mean())
        
        print(daily_sales.shape)
        print(f"   Data: {len(daily_sales)} days")
        print(f"   Range: {daily_sales['date'].min()} to {daily_sales['date'].max()}")
        print(f"   Avg daily sales: ${daily_sales['total_sales'].mean():,.0f}")
        
        # 2. Create features
        print("\n2. Creating features...")
        df_features = add_features(daily_sales)
        save_daily_sales(df_features)
        
        # 3. Prepare training data
        print("\n3. Preparing training data...")
        feature_cols = get_features()
        print("feature columns-------",feature_cols)
        
        # Remove rows with missing data
        df_clean = df_features.dropna()
        print("cleaned data columns-----------",df_clean.columns)
        
        # Split: last 30 days for testing
        split_date = df_clean['date'].max() - pd.DateOffset(days=30)
        train_data = df_clean[df_clean['date'] < split_date]
        test_data = df_clean[df_clean['date'] >= split_date]
        
        X_train = train_data[feature_cols]
        y_train = train_data['total_sales']
        X_test = test_data[feature_cols]
        y_test = test_data['total_sales']

        # Log training info
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("train_days", len(X_train))
        mlflow.log_param("test_days", len(X_test))

        print(f"   Train: {len(X_train)} days")
        print(f"   Test: {len(X_test)} days")
        
        # 4. Train models
        print("\n4. Training models...")
        model = SalesModel()
        results = model.train(X_train, y_train, X_test, y_test)

        # 🔧 CHANGE 1: Fixed pipeline results logging
        # OLD: mlflow.log_metric("pipeline_best_mae", min(results.values()))
        # NEW: Extract MAE values properly from nested dictionary
        best_mae = min(results[model]['MAE'] for model in results)
        mlflow.log_metric("pipeline_best_mae", best_mae)
        
        # 5. Save model
        print("\n5. Saving model...")
        model.save()
        
        # 6. Make future predictions
        print("\n6. Making future predictions...")
        # This retrieves and safely copies the last row of the DataFrame df_clean
        last_row = df_clean.iloc[-1].copy()
        print(f"   Last row: {last_row}")
        predictions = []
        
        for i in range(7):  # Next 7 days
            # compute the next day's date based on the 'date' value in the last row of your DataFrame.
            next_date = last_row['date'] + pd.DateOffset(days=1)
            
            # Update features
            last_row['date'] = next_date
            last_row['year'] = next_date.year
            last_row['month'] = next_date.month
            last_row['day'] = next_date.day
            last_row['dayofweek'] = next_date.dayofweek
            last_row['is_weekend'] = 1 if next_date.dayofweek >= 5 else 0
            
            # Predict
            X_pred = last_row[feature_cols].to_frame().T # 1d to 2d -> acceptable by ML models
            # print("X_pred of 0",X_pred[0])
            
            pred = model.predict(X_pred)[0]
            print("predict ", model.predict(X_pred)[0])
            
            predictions.append({
                'date': next_date,
                'predicted_sales': pred
            })
            
            # Update for next iteration
            last_row['sales_yesterday'] = pred

        # Save predictions
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv('predictions.csv', index=False)
        mlflow.log_artifact('predictions.csv')

        # Log prediction summary
        avg_prediction = pred_df['predicted_sales'].mean()
        mlflow.log_metric("avg_7day_prediction", avg_prediction)

        # Show predictions
        print("\nNext 7 days forecast:")
        for pred in predictions:
            date_str = pred['date'].strftime('%Y-%m-%d %A')
            sales = pred['predicted_sales']
            print(f"   {date_str}: ${sales:,.0f}")
        
        print("\n=== Pipeline Complete! ===")

        # ending mlflow
        mlflow.end_run()

if __name__ == "__main__":
    run_pipeline()