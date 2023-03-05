# PEMFC-Dash-App
Web-interface for the PEMFC-Model core module
([github.com/zbt-tools/pemfc-core](https://github.com/zbt-tools/pemfc-core))

![127 0 0 1_8050_ (3)](https://user-images.githubusercontent.com/94350939/222960749-90e7bfba-92a6-46de-80dd-64a1801340f2.png)


### Local installation using Anaconda

Prerequisites: git, Anaconda ([anaconda.org](https://www.anaconda.org))

- Clone this repository
- Navigate into your local directory of this repo with Anaconda prompt
- Create new conda environment via: \
  ```conda env create```
- Activate conda environment: \
  ```conda activate dash_app```
- Run: \
  ```python app.py```
- Open browser and paste specified local url

### Local installation using pip

Prerequisites: git, Python (>=3.9.7), pip

- Clone this repository
- Navigate into your local directory of this repo with console
- Install dependencies via pip:
  - Windows: \
    ```pip install -r requirements\requirements.txt```
  - Linux: \
    ```pip install -r requirements/requirements.txt```
- Run: \
  ```python app.py```
- Open browser and paste specified local url

### Create web-host with gunicorn in docker container

Prerequisites: docker

- Clone (git required) or download this repository
- Navigate into your local directory of this repo with console
- Create docker container: \
  ```docker build --tag pemfc-dash-app .```
- Run docker container: \
  ```docker run --rm -p 9090:8080 -e PORT=8080 pemfc-dash-app```
- Open browser and paste specified local url

### Specific hosting on Ubuntu with uWSGI

Prerequisites: Ubuntu, git

- follow instructions in ubuntu_specific/Manual.md

