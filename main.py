from flask import Flask, render_template, url_for, request
from cnn import predict

app = Flask(__name__)



@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    name = request.form['name']
    predict(name)
    return render_template('final.html', name=name)

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500
    
if __name__ == '__main__':
    app.run()
