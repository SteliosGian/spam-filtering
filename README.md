# Spam Filtering with TensorFlow

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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#notes">Notes</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Predicting whether a message is spam or ham using Deep Learning methods.


### Built With

* [Docker](https://www.docker.com/)
* [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/)
* [TensorFlow](https://www.tensorflow.org/)
* [FastAPI](https://fastapi.tiangolo.com/)


## Getting Started
To set up the server, clone the repo and run the following commands
```Bash
cd src/main/
uvicorn API:app --reload
```

After that, the API will be available at http://127.0.0.1:8000/predict 

In the form, you can type a message an it will predict if the message is a spam or not.


### Prerequisites
Go to the directory where the Pipfile is located (/src) and type "pipenv install".


### Notes


## Roadmap
<ul>
  <li>Containerize the application</li>
  <li>Set up CI/CD pipeline</li>
  <li>Deploy on Heroku</li>
</ul>

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-white.svg?
[linkedin-url]: https://linkedin.com/in/stelios-giannikis