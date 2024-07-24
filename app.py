from flask import Flask, render_template, request, jsonify
from main import model, vectorizer, knowledge_base, get_best_response

app = Flask(__name__)

botname = 'Kepweng'

@app.route("/")
def home():
    return render_template("index.html", botname=botname)

@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    answer = get_best_response(user_input, vectorizer, model, knowledge_base)
    return jsonify(answer)

if __name__ == '__main__':
    app.run()
