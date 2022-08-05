from flask import Flask, request, render_template
import datetime
from flask_bootstrap import Bootstrap


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def hello():
    return render_template('index.html', utc_dt=datetime.datetime.utcnow())

@app.route('/about/')
def about():
    return render_template('about.html', utc_dt=datetime.datetime.utcnow())

if __name__ == "__main__":
    app.run()