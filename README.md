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
* [TensorFlow](https://www.tensorflow.org/)
* [FastAPI](https://fastapi.tiangolo.com/)


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


### Prerequisites
Install the required Python libraries from the "requirements.txt" file.


### Notes


## Roadmap
<ul>
  <li>Containerize the application</li>
  <li>Set up CI/CD pipeline</li>
  <li>Deploy on Heroku</li>
</ul>

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-white.svg?
[linkedin-url]: https://linkedin.com/in/stelios-giannikis