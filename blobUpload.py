import pickle
from azure.storage.blob import BlobServiceClient
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import logging
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.optimizers import Adam
from config import connection_string, container_name, upload_directory, file_to_upload, input_data_set
import joblib



def uploadtoblob():
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    for folder_name, subfolders, filenames in os.walk(upload_directory):
        for filename in filenames:
            local_path = os.path.join(folder_name, filename)
            blob_name = os.path.relpath(local_path, upload_directory)
            blob_name = blob_name.replace(os.sep, '/')
            blob_name = os.path.join(upload_directory, blob_name)
            with open(local_path, "rb") as data:
                blob_client = container_client.get_blob_client(blob_name)
                blob_client.upload_blob(data, overwrite=True)
    with open(file_to_upload, "rb") as data:
        blob_client = container_client.get_blob_client(file_to_upload)
        blob_client.upload_blob(data, overwrite=True)

def getinputdatablob():
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs()
    for blob in blobs:
        if blob.name == input_data_set:
            blob_client = container_client.get_blob_client(blob)
            local_path = os.path.join(".", blob.name)
            with open(local_path, "wb") as file:
                file.write(blob_client.download_blob().readall())
    return input_data_set

def load_dataset(filename):
    try:
        # filename = getinputdatablob()
        df = pd.read_csv(filename)
        df['Maturity'] = df['Maturity'] / 365
        df['Spot Price'] = df['Spot Price'] / df['Strike Price']
        df['Call_Premium'] = df['Call_Premium'] / df['Strike Price']
        X = df.drop('Call_Premium', axis=1)
        Y = df['Call_Premium']
        return X, Y
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while loading the dataset: %s", str(e))

def scale_data(X_train, X_test):
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while scaling the data: %s", str(e))


def upload_model_to_blob(model, scaler):
    try:
        # Serialize the model and scaler objects to bytes
        model_bytes = pickle.dumps(model)
        scaler_bytes = pickle.dumps(scaler)

        # Create BlobServiceClient and ContainerClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Upload the directory
        for folder_name, subfolders, filenames in os.walk(upload_directory):
            for filename in filenames:
                local_path = os.path.join(folder_name, filename)
                blob_name = os.path.relpath(local_path, upload_directory)
                blob_name = blob_name.replace(os.sep, '/')  # Replace the path separator with '/'

                # Append the main parent folder name
                blob_name = os.path.join(upload_directory, blob_name)

                with open(local_path, "rb") as data:
                    blob_client = container_client.get_blob_client(blob_name)
                    blob_client.upload_blob(data, overwrite=True)
        # Upload the file
        with open(file_to_upload, "rb") as data:
            blob_client = container_client.get_blob_client(file_to_upload)
            blob_client.upload_blob(data, overwrite=True)

    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while uploading the model and scaler to Blob Storage: %s", str(e))


def save_model(model, scaler, model_filename, scaler_filename):
    try:
        # Save the model and scaler
        model.save(model_filename, save_format='tf')
        joblib.dump(scaler, scaler_filename)
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while saving the model and scaler: %s", str(e))


def build_model(input_dim):
    try:
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim))
        model.add(Activation('elu'))
        model.add(Dropout(0.3))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64))
        model.add(Activation('elu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)
        return model
    except Exception as e:
        print('error', str(e))
        logging.error("An error occurred while building the model: %s", str(e))
