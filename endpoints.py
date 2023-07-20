from flask import Blueprint, jsonify,current_app
import traceback
from sklearn.model_selection import train_test_split
from blobUpload import load_dataset, build_model,scale_data,getinputdatablob,uploadtoblob,save_model
import logging as logger
import config


modeltraining_routes = Blueprint('modeltraining_routes', __name__)
@modeltraining_routes.route('/', methods=['GET'])
def test():
    return "APP is running"

@modeltraining_routes.route('/train', methods=['POST'])
def train_model():
    try:
        dataset_filename = getinputdatablob()
        X, Y = load_dataset(dataset_filename)
        logger.info('Input Dataset loaded')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        logger.info('Scale data is complete')
        # Build the model
        model = build_model(X_train_scaled.shape[1])
        logger.info('Model build is complete')
        # Train the model
        num_epochs = 100
        batch_size = 64
        logger.info('Training Started')

        model.fit(X_train_scaled, Y_train, batch_size=batch_size, epochs=num_epochs,
                  validation_split=0.1, verbose=2)
        logger.info('Training complete')

        # Evaluate model on test data
        test_loss = model.evaluate(X_test_scaled, Y_test)
        test_accuracy = 100 - test_loss * 100
        print("Test Accuracy: {:.2f}%".format(test_accuracy))
        logger.info("Test Accuracy: {:.2f}%".format(test_accuracy))

        model_filename = config.upload_directory
        scaler_filename = config.file_to_upload
        save_model(model, scaler, model_filename, scaler_filename)

        uploadtoblob(model, scaler)
        logger.info('Model and scalars saved')

        return jsonify({'message': 'Model trained and saved successfully.'})
    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during model training.'})
