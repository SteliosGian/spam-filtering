# Spam Filtering with TensorFlow

[![Build Status](https://www.travis-ci.com/SteliosGian/spam-filtering.svg?branch=master)](https://www.travis-ci.com/SteliosGian/spam-filtering)
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#how-to-train">How To Train</a></li>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#notes">Notes</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This projects predicts whether a message is either Spam or Not Spam using a <a href="https://en.wikipedia.org/wiki/Long_short-term_memory" target="_blank">Long Short-term Memory Network</a>(LSTM).
<br>
The predictions are served through an API which is run in a Docker container.

The dataset for this project is taken from <a href="https://www.kaggle.com/uciml/sms-spam-collection-dataset" target="_blank">Kaggle</a>. Download it and place it in the "src/main/data/" directory.



### Built With

* [Docker](https://www.docker.com/)
* [TensorFlow](https://www.tensorflow.org/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [MLflow](https://mlflow.org/)
* [Travis CI](https://www.travis-ci.com/)


## Getting Started
This projects starts a server where the API is running.

To start, move to the docker directory by running

```Bash
cd docker/
```

After that, run the following commands. First, we build the docker image using docker-compose.
```Bash
docker compose build
```

And then, run the application inside the docker container.
```Bash
docker compose up
```

After that, the API will be available at http://0.0.0.0:8000/predict 

In the form, you can type a message an it will predict if the message is a spam or not.


### How To Train
The LSTM model is trained using the train.py file located at "src/main/train.py".

Additionally, if the MLflow server is up and running, the metrics and hyperparameters are tracked and saved.

To train the model, follow these steps:

```Bash
cd src
python3 main/train.py --source main/data/spam.csv --pipe_path main/trained_pipe --model_path main/trained_models
```

The "source", "pipe_path", and "model_path" are mandatory arguments to train the model. There are additional optional arguments to specify the hyperparameters to be used. These can be found in the "train.py" file.


### Prerequisites
Install the required Python libraries from the "requirements.txt" file.


### Notes


## Roadmap
<ul>
  <li>Containerize the application &#9745; </li>
  <li>Set up CI/CD pipeline &#9745; </li>
  <li>Deploy on Heroku</li>
</ul>

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-white.svg?
[linkedin-url]: https://linkedin.com/in/stelios-giannikis
