# Integrated model training and comparison
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tcn import TCN
from imblearn.combine import SMOTEENN
from tensorflow.keras.layers import Bidirectional,BatchNormalization
from xgboost import XGBClassifier
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ---------------------- Data Preprocessing ----------------------
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Membership Start Date'] = pd.to_datetime(df['Membership Start Date'], dayfirst=True)
    df['Membership End Date'] = pd.to_datetime(df['Membership End Date'], dayfirst=True)
    df['Subscription Plan'] = df['Subscription Plan'].fillna('Basic')
    df['Renewal Status'] = df['Renewal Status'].fillna('Manual')
    df['Engagement Level'] = df['Engagement Level'].fillna('Medium')
    df['Subscription Renewed'] = df['Subscription Renewed'].fillna('Not Renewed')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2}).fillna(-1).astype(int)
    df['Subscription Plan'] = df['Subscription Plan'].astype('category').cat.codes
    df['Renewal Status'] = df['Renewal Status'].map({'Manual': 0, 'Auto-Renew': 1})
    df['Engagement Level'] = df['Engagement Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['Subscription Renewed'] = df['Subscription Renewed'].map({'Not Renewed': 0, 'Renewed': 1})
    df['Membership Duration'] = (df['Membership End Date'] - df['Membership Start Date']).dt.days
    df['Start Month'] = df['Membership Start Date'].dt.month
    df['End Month'] = df['Membership End Date'].dt.month
    df['Membership Age'] = (pd.to_datetime('today') - df['Membership End Date']).dt.days
    return df

# ---------------------- Sequence Generator ----------------------
def create_sequences(df, prediction_gap):
    prediction_delta = {'1M': timedelta(days=30), '3M': timedelta(days=90), '6M': timedelta(days=180), '12M': timedelta(days=365)}[prediction_gap]
    user_groups = df.groupby('User ID')
    sequences, labels, idx = [], [], []
    # idx = df.index.values
    for _, group in user_groups:
        group = group.sort_values(by='Membership Start Date')
        if len(group) < 3:
            continue
        user_sequence = []
        for i in range(len(group)):
            record = group.iloc[i]
            features = [record['Age'], int(record['Gender']), int(record['Subscription Plan']), int(record['Renewal Status']), int(record['Engagement Level']), record['Membership Duration'], record['Start Month'], record['End Month'], record['Membership Age']]
            user_sequence.append(features)
            target_start = record['Membership End Date']
            target_end = target_start + prediction_delta
            future_renewals = group[(group['Membership Start Date'] > target_start) & (group['Membership Start Date'] <= target_end)]
            label = 0
            if len(future_renewals) > 0:
                label = int(future_renewals['Subscription Renewed'].any())
            if len(user_sequence) >= 3:
                sequences.append(user_sequence[-3:])
                labels.append(label)
                idx.append(record.name)
    print(f"Sequences length: {len(sequences)}")
    print(f"Labels length: {len(labels)}")
    print(f"Idx length: {len(idx)}")
    return np.array(sequences), np.array(labels), np.array(idx)

# ---------------------- Model Trainers ----------------------
def train_lstm(X, y, idx):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X.reshape(X.shape[0], -1), y)
    X_res = X_res.reshape(X_res.shape[0], X.shape[1], X.shape[2])
    
    # Reshape X_res back to the original 3D shape
    X_res = X_res.reshape(X_res.shape[0], X.shape[1], X.shape[2])

    # Ensure idx has the correct number of repetitions
    repeats = X_res.shape[0] // X.shape[0]
    remainder = X_res.shape[0] % X.shape[0]

    # Tile idx to match the number of samples in X_res
    idx_res = np.tile(idx, repeats)

    # If there is a remainder, repeat the first few elements of idx to match the exact number of samples
    if remainder > 0:
        idx_res = np.concatenate([idx_res, idx[:remainder]])

    # Flatten idx_res to match the shape of X_res
    idx_res = idx_res.flatten()

    # Assert to check if lengths match
    assert len(X_res) == len(idx_res), "Length of X_res and idx_res doesn't match!"
    
    print(f"Shape of X_res: {X_res.shape}")
    print(f"Shape of y_res: {y_res.shape}")
    print(f"Length of idx_res: {len(idx_res)}")
    
    # indices = np.arange(len(X))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_res, y_res, idx_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights_dict,
        callbacks=[early_stopping],
        verbose=0
    )

    preds = model.predict(X_test).flatten()
    preds_binary = (preds > 0.5).astype(int)
    
    print("\nModel Weights Summary:")
    for layer in model.layers:
        weights = layer.get_weights()
        print(f"\nLayer: {layer.name}")
        for w in weights:
            print(f"Weight shape: {w.shape}")

    # Forecast (Validation Predictions)
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = np.round(y_pred_prob).clip(0,1)

    # Residuals
    residuals = y_test - y_pred_prob

    # print("\nResidual Diagnostics:")
    # plt.figure(figsize=(10, 4))
    # plt.plot(residuals)
    # plt.title('Residuals over Validation Set')
    # plt.show()

    # Forecast Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nForecast Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    print("LSTM Predictions:", y_pred_prob[:10])  
    
    return accuracy_score(y_test, preds_binary), log_loss(y_test, preds), history, X_test, y_test, model, idx_test


def train_tcn(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X.reshape(X.shape[0], -1), y)
    X_res = X_res.reshape(X_res.shape[0], X.shape[1], X.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    model = Sequential([Input(shape=(X.shape[1], X.shape[2])), TCN(), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    preds = model.predict(X_test).flatten()
    preds_binary = (preds > 0.5).astype(int)
    print("\nModel Summary:")
    model.summary()

    # Forecast
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = np.round(y_pred_prob).clip(0,1)

    # Residuals
    residuals = y_test - y_pred_prob

    # print("\nResidual Diagnostics:")
    # plt.figure(figsize=(10, 4))
    # plt.plot(residuals)
    # plt.title('Residuals over Validation Set')
    # plt.show()

    # Forecast Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nForecast Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    print("TCN Predictions:", y_pred_prob[:10])
    
    return accuracy_score(y_test, preds_binary), log_loss(y_test, preds), history


def train_xgb(X, y):
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], -1)

    # Handle class imbalance with SMOTE-ENN instead of normal SMOTE
    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model with better hyperparameters
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        learning_rate=0.001,
        n_estimators=100,
        max_depth=15,
        random_state=42,
        reg_lambda=100
    )

    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(X_train.shape[1])],
        'Importance': model.feature_importances_
    })
    print("\nMain Parameters (Feature Importances):")
    print(importance_df.sort_values(by='Importance', ascending=False))

    # Calculate residuals
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    residuals = y_test - y_pred_prob

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_prob))
    mae = mean_absolute_error(y_test, y_pred_prob)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    return accuracy_score(y_test, preds), log_loss(y_test, proba)


# ---------------------- Visualization ----------------------
def plot_model_comparisons(gap_metrics):
    gaps = list(gap_metrics.keys())
    models = list(next(iter(gap_metrics.values())).keys())
    colors = ['b', 'g', 'orange']
    plt.figure(figsize=(10, 5))
    for model, color in zip(models, colors):
        accs = [gap_metrics[gap][model]['accuracy'] for gap in gaps]
        plt.plot(gaps, accs, marker='o', label=model, color=color)
    plt.title('Validation Accuracy Across Models & Gaps')
    plt.xlabel('Prediction Gap')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_comparison_accuracy.png')
    plt.close()

def plot_accuracy_per_epoch(histories, gap):
    plt.figure(figsize=(8, 4))
    for model_name, hist in histories.items():
        if hist is not None:
            plt.plot(hist.history['accuracy'], label=f'{model_name}')
    plt.title(f'Accuracy over Epochs ({gap})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'accuracy_over_epochs_{gap}.png')
    plt.close()
    
def plot_lstm_forecast(train_data, test_data, future_predictions, test_dates, future_dates):

    plt.figure(figsize=(10, 6))

    # Plot Training Data
    plt.plot(train_data.index, train_data['Subscription Renewed'], label='Training Data', color='blue')

    # Plot Test Data
    plt.plot(test_dates, test_data['Subscription Renewed'], label='Test Data', color='green')

    # Plot Forecasted Future Data
    plt.plot(future_dates, future_predictions, label='Forecasted Future', color='red', linestyle='--')

    plt.title("LSTM Model - Training, Test, and Forecasted Data (1M Gap)")
    plt.xlabel("Date")
    plt.ylabel("Subscription Renewed (1=Renewed, 0=Not Renewed)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    csv_file = 'TS_OTTDataset_Cleaned.csv'
    gap_metrics = {gap: {'LSTM': {}, 'TCN': {}, 'XGBoost': {}} for gap in ['1M', '3M', '6M', '12M']}
    df = load_and_prepare_data(csv_file)

    for gap in ['1M', '3M', '6M', '12M']:
        print(f"\n🔁 Training models for prediction gap: {gap}")
        X, y, idx = create_sequences(df, gap)

        histories = {}

        acc, loss, hist, X_test, y_test, model, idx_test = train_lstm(X, y, idx)
        gap_metrics[gap]['LSTM'] = {'accuracy': acc, 'loss': loss}
        histories['LSTM'] = hist

        acc, loss, hist = train_tcn(X, y)
        gap_metrics[gap]['TCN'] = {'accuracy': acc, 'loss': loss}
        histories['TCN'] = hist

        acc, loss = train_xgb(X, y)
        gap_metrics[gap]['XGBoost'] = {'accuracy': acc, 'loss': loss}
        histories['XGBoost'] = None

        plot_accuracy_per_epoch(histories, gap)
        
        # Forecast and plot future predictions for LSTM
        # future_predictions = []
        # current_input = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])

        # for _ in range(12):  # Predict 12 months ahead
        #     pred = model.predict(current_input, verbose=0)
        #     future_predictions.append(pred[0])
        #     current_input = np.roll(current_input, shift=-1, axis=1)
        #     current_input[0, -1, 0] = pred  # Update only first feature (assumed target)

        # future_predictions = np.round(np.array(future_predictions)).astype(int)

        # test_data_df = df.iloc[idx_test]
        # test_dates = pd.to_datetime(test_data_df['Membership End Date'])

        # future_dates = pd.date_range(start=test_dates.max(), periods=13, freq='M')[1:]

        # plot_lstm_forecast(test_data_df, test_data_df, future_predictions, test_dates, future_dates)

    plot_model_comparisons(gap_metrics)
    print("✅ Model training and comparison complete. Accuracy plots saved.")
