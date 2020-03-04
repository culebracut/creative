from flask import Flask
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/hello/<name>")
def hello_there(name):
    foo =jsonify(name)
    return(foo)