from flask import Flask, render_template, url_for, request
from cnn import predict

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    name = request.form['name']
    res= predict(name)
    return render_template('final.html', res= res)

    
if __name__ == '__main__':
    app.run()
