from flask import Flask
from endpoints import modeltraining_routes

app = Flask(__name__)
app.register_blueprint(modeltraining_routes)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)

