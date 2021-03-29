"""
Usage
-----
CUDA_VISIBLE_DEVICES=0 python app.py
"""

import os
from time import time

from CodemixedNLP.cli import ModelCreator
from flask import Flask, render_template, request
from flask_cors import CORS

# TOKENIZE = True
TOPK = 1
PRELOADED_MODELS = {}
CURR_MODEL_KEYWORD = None
CURR_MODEL = (None, None, None)  # representing (vocabs, model, get_predictions_function_signature)

LOGS_PATH = "./logs"
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
opfile = open(os.path.join(LOGS_PATH, str(time()) + ".logs.txt"), "w")


def load_model(task_name):
    global PRELOADED_MODELS, CURR_MODEL_KEYWORD, CURR_MODEL

    if task_name not in PRELOADED_MODELS:
        print(f"loading model for {task_name}")
        pretrained = ModelCreator().from_pretrained(task_name)
        vocabs, model = pretrained.vocabs, pretrained.model
        get_predictions = pretrained.get_predictions
        PRELOADED_MODELS.update({task_name: (vocabs, model, get_predictions)})

    CURR_MODEL_KEYWORD = task_name
    CURR_MODEL = PRELOADED_MODELS[task_name]

    return


def preload_models():
    print("pre-loading models")
    global PRELOADED_MODELS, CURR_MODEL_KEYWORD

    for task_name in ["sentiment", "aggression", "lid"]:
        load_model(task_name)

    print("\n")
    for k in PRELOADED_MODELS.keys():
        print(f"preloaded model for {k}")
    print("\n")

    print(f"set CURR_MODEL_KEYWORD to {CURR_MODEL_KEYWORD}")

    return


def save_query(text):
    global opfile
    opfile.write(text)
    opfile.flush()
    return


# Define the app
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default


@app.route('/')
@app.route('/home', methods=['POST'])
def home():
    return render_template('home.html')


@app.route('/loaded', methods=['POST'])
def loaded():
    print(request.form)
    print(request.form["checkers"])
    load_model(request.form["checkers"])
    return render_template('loaded.html')


@app.route('/reset', methods=['POST'])
def reset():
    return render_template('loaded.html')


@app.route('/predict', methods=['POST'])
def predict():
    global CURR_MODEL, CURR_MODEL_KEYWORD, TOPK
    print(CURR_MODEL_KEYWORD)
    if request.method == 'POST':
        print("#################")
        print(request.form)
        print(request.form.keys())
        message = request.form['hidden-message']
        message = message.strip("\n").strip("\r")
        if message == "":
            return render_template('loaded.html')
        if TOPK == 1:
            vocabs, model, get_predictions = CURR_MODEL
            label_vocab = None if (vocabs is None or len(vocabs) == 0) else vocabs["label_vocab"]
            prediction = get_predictions(message)[0]
            print(message)
            print(prediction)
            save_query(CURR_MODEL_KEYWORD + "\t" + message + "\t" + prediction + "\n")

            if CURR_MODEL_KEYWORD in ["lid", ]:
                prediction = " ".join(['++--' + f'{x}' + '/' + f'{y}' + '-+-' if y == "en"
                                       else '--++' + f'{x}' + '/' + f'{y}' + '-+-' if y == "hi"
                else '+-+' + f'{x}' + '/' + f'{y}' + '-+-'
                                       for x, y in zip(message.split(" "), prediction.split(" "))])
            elif CURR_MODEL_KEYWORD in ["pos", "ner", ]:
                # In case of ner or pos models, the prediction can be returned in the format
                # token1\POS token2\POS2 token3\POS3 .... tokenN\POSN
                prediction = " ".join(
                    [f'{x}' + '/' + f'{y}' for x, y in zip(message.split(" "), prediction.split(" "))])

            return render_template('result.html', prediction=prediction, message=message)
        else:
            raise NotImplementedError("please keep TOPK=1")
    return render_template('home.html')


if __name__ == "__main__":
    print("*** Flask Server ***")
    preload_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
