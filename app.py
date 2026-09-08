from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load trained Perceptron model
with open("perceptron.pkl", "rb") as file:
    model = pickle.load(file)


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Perceptron Prediction</title>

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, Helvetica, sans-serif;
        }

        body {
            min-height: 100vh;
            background:
                radial-gradient(circle at top left, #273469, transparent 35%),
                radial-gradient(circle at bottom right, #1b998b, transparent 30%),
                #0f172a;

            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px;
            color: white;
        }

        .container {
            width: 100%;
            max-width: 1000px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 25px;
            overflow: hidden;
            box-shadow: 0 25px 60px rgba(0,0,0,0.35);
        }

        .left {
            padding: 50px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .badge {
            display: inline-block;
            width: fit-content;
            padding: 8px 15px;
            border-radius: 30px;
            background: rgba(45,212,191,0.15);
            border: 1px solid rgba(45,212,191,0.4);
            color: #5eead4;
            font-size: 13px;
            margin-bottom: 20px;
        }

        h1 {
            font-size: 44px;
            line-height: 1.1;
            margin-bottom: 18px;
        }

        .highlight {
            color: #5eead4;
        }

        .description {
            color: #cbd5e1;
            line-height: 1.7;
            font-size: 16px;
            margin-bottom: 25px;
        }

        .features {
            list-style: none;
        }

        .features li {
            margin: 13px 0;
            color: #e2e8f0;
        }

        .features li::before {
            content: "✓";
            color: #5eead4;
            font-weight: bold;
            margin-right: 10px;
        }

        .right {
            background: rgba(15,23,42,0.65);
            padding: 45px;
            display: flex;
            justify-content: center;
            flex-direction: column;
        }

        .card-title {
            font-size: 26px;
            margin-bottom: 8px;
        }

        .card-subtitle {
            color: #94a3b8;
            margin-bottom: 30px;
        }

        .input-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #cbd5e1;
            font-size: 14px;
        }

        input {
            width: 100%;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #334155;
            background: #1e293b;
            color: white;
            outline: none;
            font-size: 16px;
            transition: 0.3s;
        }

        input:focus {
            border-color: #5eead4;
            box-shadow: 0 0 0 3px rgba(94,234,212,0.1);
        }

        button {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #14b8a6, #06b6d4);
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
            margin-top: 5px;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(20,184,166,0.3);
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            background: rgba(94,234,212,0.1);
            border: 1px solid rgba(94,234,212,0.3);
        }

        .result-title {
            color: #94a3b8;
            font-size: 13px;
            margin-bottom: 8px;
        }

        .prediction {
            font-size: 28px;
            font-weight: bold;
            color: #5eead4;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            border-radius: 12px;
            background: rgba(239,68,68,0.12);
            border: 1px solid rgba(239,68,68,0.3);
            color: #fca5a5;
        }

        .footer {
            margin-top: 25px;
            text-align: center;
            color: #64748b;
            font-size: 12px;
        }

        @media(max-width: 800px) {
            .container {
                grid-template-columns: 1fr;
            }

            .left {
                padding: 35px;
            }

            .right {
                padding: 35px;
            }

            h1 {
                font-size: 35px;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <div class="left">

        <div class="badge">
            MACHINE LEARNING MODEL
        </div>

        <h1>
            Perceptron
            <span class="highlight">Prediction</span>
        </h1>

        <p class="description">
            Enter CGPA and Resume Score to predict whether
            a candidate will be placed or not placed.
        </p>

        <ul class="features">
            <li>Scikit-learn Perceptron model</li>
            <li>Real-time prediction</li>
            <li>Simple interactive interface</li>
            <li>Responsive design</li>
        </ul>

    </div>


    <div class="right">

        <h2 class="card-title">
            Make a Prediction
        </h2>

        <p class="card-subtitle">
            Enter the candidate details below.
        </p>

        <form method="POST">

            <div class="input-group">
                <label for="feature1">
                    CGPA
                </label>

                <input
                    type="number"
                    step="any"
                    id="feature1"
                    name="feature1"
                    placeholder="Enter CGPA"
                    required
                >
            </div>


            <div class="input-group">
                <label for="feature2">
                    Resume Score
                </label>

                <input
                    type="number"
                    step="any"
                    id="feature2"
                    name="feature2"
                    placeholder="Enter Resume Score"
                    required
                >
            </div>


            <button type="submit">
                Predict Result →
            </button>

        </form>


        {% if prediction is not none %}

        <div class="result">

            <div class="result-title">
                MODEL PREDICTION
            </div>

            <div class="prediction">
                {{ prediction }}
            </div>

        </div>

        {% endif %}


        {% if error %}

        <div class="error">
            {{ error }}
        </div>

        {% endif %}


        <div class="footer">
            Powered by Python • Flask • Scikit-learn
        </div>

    </div>

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            # Feature 1 = CGPA
            cgpa = float(request.form["feature1"])

            # Feature 2 = Resume Score
            resume_score = float(request.form["feature2"])

            # Model expects:
            # [CGPA, Resume Score]
            input_data = np.array([[cgpa, resume_score]])

            # Get model prediction
            result = model.predict(input_data)[0]

            # Convert 0/1 into meaningful output
            if result == 1:
                prediction = "Placed"
            else:
                prediction = "Not Placed"

        except Exception as e:
            error = f"Prediction error: {str(e)}"

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

