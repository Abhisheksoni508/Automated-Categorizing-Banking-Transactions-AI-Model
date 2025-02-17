import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('C:/Users/Abhishek Soni/Downloads/personal_transactions.csv')

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data['YearMonth'] = data['Date'].dt.to_period('M')

# Pivot data to include individual debit and credit features per category
pivot_data = data.pivot_table(
    index='YearMonth',
    columns=['Category', 'Transaction Type'],
    values='Amount',    
    aggfunc='sum'
).fillna(0)

# Flatten MultiIndex columns (Category, Transaction Type) to single level
pivot_data.columns = ['_'.join(col).strip() for col in pivot_data.columns.values]

# Reset index for modeling
pivot_data.reset_index(inplace=True)

# Prepare features and targets
features = pivot_data.iloc[:, 1:].copy()  # Exclude 'YearMonth' for modeling
scaler_features = MinMaxScaler()
scaled_features = scaler_features.fit_transform(features)

# Prepare time-series data (sequence of past months)
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Past 'sequence_length' months
        y.append(data[i + sequence_length])  # Next month's values
    return np.array(X), np.array(y)

# Create sequences for features
sequence_length = 12
X, y = create_sequences(scaled_features, sequence_length)

# Train-test split
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshape X to be a 2D array for MLP (we remove the sequence dimension)
X_train_flattened = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test_flattened = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

# Configurations to test
configs = [
  #  {'dropout_rate': 0.3, 'learning_rate': 0.001, 'neurons': [128, 64]},
  #  {'dropout_rate': 0.3, 'learning_rate': 0.01, 'neurons': [128, 64]},
    {'dropout_rate': 0.25, 'learning_rate': 0.0002, 'neurons': [256, 256, 128]}
]


# Experimenting with different configurations
def experiment_with_config(config):
    # Build the MLP model with the current configuration
    model = Sequential()
    
    # Input layer
    model.add(Dense(config['neurons'][0], activation='relu', input_dim=X_train_flattened.shape[1]))
    model.add(Dropout(config['dropout_rate']))
    
    # Hidden layers
    for neurons in config['neurons'][1:]:
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(config['dropout_rate']))
    
    # Output layer
    model.add(Dense(y_train.shape[1], activation='linear'))  # Output layer (same number of units as target variables)
    
    # Compile the model with the specified learning rate
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    # Train the model
    history = model.fit(
        X_train_flattened, y_train,
        validation_data=(X_test_flattened, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Visualize Loss Over Epochs
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Model Loss Over Epochs (Dropout: {config['dropout_rate']}, LR: {config['learning_rate']})")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test_flattened, y_test, verbose=0)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}\n")
    
    return model  # Return the trained model

# Run experiments for each configuration and select the final model
final_model = None
for config in configs:
    print(f"\nExperimenting with Configuration: {config}")
    final_model = experiment_with_config(config)

# Define User Interaction Function with unique output style
categories = features.columns  # All individual categories and transaction types

def predict_future_spending(year, month, categories, sequence_length, model, scaler_features):
    # Use the most recent sequence for prediction
    last_sequence = scaled_features[-sequence_length:]
    last_sequence_flattened = last_sequence.reshape((1, last_sequence.shape[0] * last_sequence.shape[1]))
    future_prediction = model.predict(last_sequence_flattened)
    # Rescale predictions back to the original scale
    future_prediction_rescaled = scaler_features.inverse_transform(future_prediction)
    # Clip negative values to 0
    future_prediction_rescaled[future_prediction_rescaled < 0] = 0

    total_debit = 0
    total_credit = 0

    # New styled output
    print(f"\n*** Predicted Spending Overview for {month:02d}/{year} ***\n")
    print("Detailed Category Breakdown:")
    print("-" * 50)

    for category, predicted_value in zip(categories, future_prediction_rescaled[0]):
        if "_debit" in category:
            total_debit += predicted_value
            print(f"▶️  {category.replace('_debit', '').title()} (Debit) : £{predicted_value:.2f}")
        elif "_credit" in category:
            total_credit += predicted_value
            print(f"▶️  {category.replace('_credit', '').title()} (Credit) : £{predicted_value:.2f}")

    print("-" * 50)
    print("\nSummary Overview:")
    print(f"💸 Total Debited Amount    : £{total_debit:.2f}")
    print(f"💰 Total Credited Amount   : £{total_credit:.2f}")
    print("-" * 50)
    print("Thank you for using the Spending Prediction Service!")
    
    return future_prediction_rescaled[0]

# User Interaction with debugging
while True:
    user_input = input("Enter the month and year for prediction (MM/YYYY) or 'exit' to quit: ")
    
    # Debugging: print the raw input
    print(f"Received input: {user_input}")
    
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    try:
        month, year = map(int, user_input.split('/'))
        
        if 1 <= month <= 12:
            print(f"Predicting spending for {month:02d}/{year}...")
            if final_model is not None:
                predict_future_spending(year, month, categories, sequence_length, final_model, scaler_features)
            else:
                print("No trained model found. Please ensure training has completed successfully.")
        else:
            print("Invalid month. Please enter a valid MM/YYYY format.")
    except ValueError:
        print("Invalid input. Please enter a valid MM/YYYY format.")