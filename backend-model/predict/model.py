import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Load and preprocess the data
def load_data(filepath):
    print("[INFO] Loading data...")
    df = pd.read_excel(filepath, engine='openpyxl')
    df['Visit Date'] = pd.to_datetime(df['Visit Date'])
    df['Date'] = df['Visit Date'].dt.date  # remove time component
    return df

# Aggregate visits and household size by day and agency
def aggregate_visits(df):
    print("[INFO] Aggregating daily visit counts and household sizes...")
    grouped = df.groupby(['Date', 'Visited Agency']).agg({
        'Household Size': ['count', 'mean']
    }).reset_index()
    grouped.columns = ['Date', 'Visited Agency', 'Visit Count', 'Avg Household Size']
    return grouped

# Feature engineering
def add_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.weekday
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Add cyclical encoding for day of week, month, etc.
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365)
    
    # Add more features to help with oscillation patterns
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Week_sin'] = np.sin(2 * np.pi * df['Week']/52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week']/52)
    
    return df

# Add lagged features
def add_lagged_features(df, lags=[1, 2, 3, 7, 14, 28]):
    for lag in lags:
        df[f'Visit_Count_Lag_{lag}'] = df.groupby('Visited Agency')['Visit Count'].shift(lag)
    
    # Add rolling means to capture trends at different time scales
    for window in [7, 14, 30]:
        df[f'Rolling_Mean_{window}d'] = df.groupby('Visited Agency')['Visit Count'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
    
    # Add features for rate of change (momentum)
    df['Visit_Change'] = df.groupby('Visited Agency')['Visit Count'].diff()
    
    df.fillna(0, inplace=True)
    return df

# Prepare features and labels
def prepare_data(df):
    print("[INFO] Preparing features and labels...")
    df = add_date_features(df)
    df = add_lagged_features(df)
    
    # Use MinMaxScaler instead of StandardScaler for time series
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # One-hot encode agency
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    agency_encoded = enc.fit_transform(df[['Visited Agency']])
    agency_df = pd.DataFrame(agency_encoded, columns=enc.get_feature_names_out())
    
    # List of numerical features, excluding target and date
    numerical_features = [col for col in df.columns if col not in 
                         ['Date', 'Visited Agency', 'Visit Count']]
    
    numerical_df = df[numerical_features]
    numerical_scaled = scaler.fit_transform(numerical_df)
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_features)
    
    # Combine features
    features = pd.concat([numerical_scaled_df, agency_df], axis=1)
    features.index = df.index # keep the index to avoid errors later
    labels = df['Visit Count'].values
    
    print(f"[DEBUG] Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    return features, labels, enc, scaler

# Build TensorFlow model
def build_model(input_shape):
    print("[INFO] Building model with advanced architecture...")
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    
    # Initial dense layers
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Add residual connections to help with complex patterns
    for units in [128, 64]:
        residual = x
        x = tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        # Add residual connection if shapes match
        if residual.shape[-1] == x.shape[-1]:
            x = tf.keras.layers.add([x, residual])
        else:
            # Use a projection to match dimensions
            projection = tf.keras.layers.Dense(units, activation='linear')(residual)
            x = tf.keras.layers.add([x, projection])
    
    # Final regression output
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use a fixed learning rate instead of a schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model

def train_model(filepath, model_dir='models'):
    df = load_data(filepath)
    daily_data = aggregate_visits(df)
    features, labels, encoder, scaler = prepare_data(daily_data)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = build_model(X_train.shape[1])

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "food_bank_model.keras")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_best_only=True,
        monitor='val_loss'
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=20,
        restore_best_weights=True,
        monitor='val_loss'
    )

    # Add learning rate reduction callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,
        patience=5, 
        min_lr=0.00001,
        verbose=1
    )

    print("[INFO] Starting training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=1000,  # More epochs with early stopping
        batch_size=32,  # Adjust batch size
        validation_split=0.2,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr]
    )

    # Save the encoder and scaler
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[INFO] Model, encoder, and scaler saved to {model_dir}")
    return model, encoder, scaler, df

# Add a load_saved_model function
def load_saved_model(model_dir='models'):
    model_path = os.path.join(model_dir, "food_bank_model.keras")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"[ERROR] Model files not found in {model_dir}")

    print(f"[INFO] Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"[INFO] Loading encoder from {encoder_path}")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    
    print(f"[INFO] Loading scaler from {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, encoder, scaler

# Prediction
def predict(model, encoder, scaler, date_str, agency_name, avg_household_size):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    day = date.weekday()
    month = date.month
    year = date.year
    dayofyear = date.timetuple().tm_yday
    
    # Calculate quarter manually
    quarter = (month - 1) // 3 + 1
    
    is_weekend = 1 if day in [5, 6] else 0
    
    # For debugging, get the feature names from the scaler
    feature_names = scaler.feature_names_in_
    print(f"[DEBUG] Scaler feature names: {feature_names}")
    
    # Create a dictionary with all required features
    input_dict = {}
    
    # Initialize all features that might be in the scaler's feature set
    all_features = {
        'DayOfWeek': day,
        'Month': month,
        'Year': year,
        'DayOfYear': dayofyear,
        'Quarter': quarter,
        'IsWeekend': is_weekend,
        'Avg Household Size': avg_household_size,
        'DayOfWeek_sin': np.sin(2 * np.pi * day/7),
        'DayOfWeek_cos': np.cos(2 * np.pi * day/7),
        'Month_sin': np.sin(2 * np.pi * month/12),
        'Month_cos': np.cos(2 * np.pi * month/12),
        'DayOfYear_sin': np.sin(2 * np.pi * dayofyear/365),
        'DayOfYear_cos': np.cos(2 * np.pi * dayofyear/365),
        'Week': int(date.strftime('%V')),
        'Week_sin': np.sin(2 * np.pi * int(date.strftime('%V'))/52),
        'Week_cos': np.cos(2 * np.pi * int(date.strftime('%V'))/52)
    }
    
    # Add lagged features
    for lag in [1, 2, 3, 7, 14, 28]:
        all_features[f'Visit_Count_Lag_{lag}'] = 0
    
    # Add rolling mean features
    for window in [7, 14, 30]:
        all_features[f'Rolling_Mean_{window}d'] = 0
    
    # Add momentum feature
    all_features['Visit_Change'] = 0
    
    # Only use the feature names that were used during training
    for feature in feature_names:
        if feature in all_features:
            input_dict[feature] = [all_features[feature]]
        else:
            # If a feature is missing, add a placeholder (0)
            print(f"[WARNING] Feature '{feature}' not found in input, using 0")
            input_dict[feature] = [0]
    
    # Create dataframe with exact same columns and order as used in training
    input_df = pd.DataFrame(input_dict)
    
    # Scale the features using the scaler
    numerical_scaled = scaler.transform(input_df)
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=feature_names)
    
    # Transform agency name using the encoder
    agency_df = pd.DataFrame({'Visited Agency': [agency_name]})
    agency_encoded = encoder.transform(agency_df)
    agency_columns = encoder.get_feature_names_out(['Visited Agency'])
    agency_df_encoded = pd.DataFrame(agency_encoded, columns=agency_columns)
    
    # Combine features
    features = pd.concat([numerical_scaled_df, agency_df_encoded], axis=1)
    
    # Debug info
    print(f"[DEBUG] Prediction features: {features.shape}, Model input shape: {model.input_shape}")
    
    # Make prediction
    prediction = model.predict(features, verbose=0)[0][0]
    return max(0, round(prediction))  # Apply a 20% boost to predictions

def plot_agency_predictions(model, encoder, scaler, df, agency_name):
    """
    Plot the actual visit counts against the model predictions for a specific agency.
    """
    # First aggregate the data if not already aggregated
    if 'Visit Count' not in df.columns:
        df = aggregate_visits(df)
    
    # Filter data for the specified agency
    agency_data = df[df['Visited Agency'] == agency_name].copy()
    
    if len(agency_data) == 0:
        print(f"[ERROR] No data found for agency '{agency_name}'")
        return
    
    # Sort by date
    agency_data = add_date_features(agency_data)
    agency_data = agency_data.sort_values('Date')
    agency_data = add_lagged_features(agency_data)
    
    # Make predictions for each date
    predictions = []
    for idx, row in agency_data.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        avg_household_size = row['Avg Household Size'] if 'Avg Household Size' in row else row.get('Household Size', 2)
        pred = predict(model, encoder, scaler, date_str, agency_name, avg_household_size)
        predictions.append(pred)
    
    agency_data['Predicted'] = predictions
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(agency_data['Date'], agency_data['Visit Count'], 'o-', label='Actual Visits')
    plt.plot(agency_data['Date'], agency_data['Predicted'], 'r--', label='Predicted Visits')
    
    plt.title(f'Actual vs Predicted Visits: {agency_name}')
    plt.xlabel('Date')
    plt.ylabel('Number of Visits')
    plt.legend()
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    # Make y-axis show integers only
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show or save the plot
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plot_path = f"plots/{agency_name.replace(' ', '_')}_predictions.png"
    plt.savefig(plot_path)
    print(f"[INFO] Plot saved to {plot_path}")
    
    plt.show()

# Main code
if __name__ == "__main__":
    data_file = "data/2024visits1.xlsx"
    model_dir = 'models'
    agency = "Churches on the Hill Food Bank"
    date = "2024-04-10"
    
    # Change this line - use 'Avg Household Size' which is the column name after aggregation
    household_size_column = 'Avg Household Size'

    try:
        model, encoder, scaler = load_saved_model(model_dir)
        df = load_data(data_file)
        daily_data = aggregate_visits(df)
    except FileNotFoundError:
        print("[INFO] No saved model found, training new model...")
        model, encoder, scaler, df = train_model(data_file, model_dir)
        daily_data = aggregate_visits(df)
    
    # Get average household size from past data for that agency (fallback = 2)
    avg_household_size = daily_data[daily_data['Visited Agency'] == agency][household_size_column].mean()
    if np.isnan(avg_household_size):
        avg_household_size = 2

    predicted_visits = predict(model, encoder, scaler, date, agency, avg_household_size)
    print(f"[RESULT] Predicted visits on {date} at {agency}: {predicted_visits}")
    
    # Plot the predictions for this agency
    plot_agency_predictions(model, encoder, scaler, df, agency)
s