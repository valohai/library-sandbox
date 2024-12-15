import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

char_index = "0abcdefghijklmnopqrstuvwxyz"
char_index += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_index += "123456789"
char_index += "().,-/+=&$?@#!*:;_[]|%⸏{}\"'" + " " + "\\"

char_to_int = {c: i for i, c in enumerate(char_index)}
int_to_char = dict(enumerate(char_index))


def encode_sequence_list(seqs, feat_n=0):
    from keras.preprocessing.sequence import pad_sequences

    encoded_seqs = []
    for seq in seqs:
        encoded_seq = [char_to_int[c] for c in seq]
        encoded_seqs.append(encoded_seq)
    if feat_n > 0:
        encoded_seqs.append(np.zeros(feat_n))
    return pad_sequences(encoded_seqs, padding="post")


def decode_sequence_list(seqs):
    decoded_seqs = []
    for seq in seqs:
        decoded_seq = [int_to_char[i] for i in seq]
        decoded_seqs.append(decoded_seq)
    return decoded_seqs


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder


def detect_by_autoencoder(df, columns, percentile=95):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    outliers = pd.DataFrame()
    non_outliers = pd.DataFrame()
    scaler = StandardScaler()
    for col in columns:
        encoded_seqs = df[col].values.reshape(-1, 1)
        if df[col].dtype == "object":
            encoded_seqs = encode_sequence_list(df[col].values, feat_n=0)
        X_scaled = scaler.fit_transform(encoded_seqs)
        X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
        autoencoder = build_autoencoder(X_train.shape[1])
        autoencoder.fit(
            X_train,
            X_train,
            epochs=50,
            batch_size=32,
            shuffle=True,
            validation_data=(X_test, X_test),
            verbose=0,
        )
        reconstructed_data = autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructed_data), axis=1)
        threshold = np.percentile(mse, percentile)
        outliers = outliers._append(df[mse > threshold])
        non_outliers = non_outliers._append(df[mse <= threshold])

    return outliers, non_outliers


def detect_by_zscore(df, columns=None, threshold: float = 2.0):
    outliers = pd.DataFrame()
    non_outliers = pd.DataFrame()
    for col in columns:
        main_col = (
            df[col]
            if df[col].dtype != "object"
            else encode_sequence_list(df[col].values, feat_n=0)
        )
        z_scores = (main_col - main_col.mean()) / main_col.std()
        outliers = outliers._append(df[abs(z_scores) > threshold])
        non_outliers = non_outliers._append(df[abs(z_scores) <= threshold])
    return outliers, non_outliers


def detect_by_iqr(df, columns=None, threshold=1.5):
    outliers = pd.DataFrame()
    non_outliers = pd.DataFrame()

    for col in columns:
        main_col = df[col] if df[col].dtype != "object" else df[col].apply(len)
        Q1 = main_col.quantile(0.25)
        Q3 = main_col.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = outliers._append(
            df[(main_col < lower_bound) | (main_col > upper_bound)],
        )
        non_outliers = non_outliers._append(
            df[(main_col >= lower_bound) & (main_col <= upper_bound)],
        )

    return outliers, non_outliers


def detect_by_isolation_forest(df, columns=None, threshold=0.1):
    from sklearn.ensemble import IsolationForest

    outliers = pd.DataFrame()
    non_outliers = pd.DataFrame()
    clf = IsolationForest(contamination=threshold)
    for col in columns:
        (
            df[col]
            if df[col].dtype != "object"
            else encode_sequence_list(df[col], feat_n=0)
        )
        results = clf.fit_predict(df[col].values.reshape(-1, 1))
        outlier_indices = results == -1
        outliers = outliers._append(df[outlier_indices])
        non_outliers = non_outliers._append(df[~outlier_indices])

    return outliers, non_outliers
