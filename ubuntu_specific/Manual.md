# How to host pemfc-dash-app on Ubuntu 20.04 with wsgi
###Steps

- Create app folder
- Clone repository https://github.com/ZBT-Tools/pemfc-dash-app
- Install: sudo apt install python3.8-venv
- Install: pip install wheel
- Create environment python3 -m venv your_env_name
- Activate your_env_name
- Install packages: pip install -r requirements/requirements.txt
- check if dash app runs standalone: python3 main.py 
- Install and prepare uWSGI
  - sudo apt-get install build-essential python-dev
  - sudo apt install uwsgi-core
  - sudo apt install  uwsgi-plugin-python3
  - Move wsgi.py from ubuntu_specific to working dir (pemfc-dash-app)
  - Move index.ini file from ubuntu_specific to working dir (pemfc-dash-app)
  - Modify virtualenv path in index.ini
  - restart 
- run server
  - uwsgi index.ini
- Set up server to start at system start and connect it to NGINX with following steps:
  - https://carpiero.medium.com/host-a-dashboard-using-python-dash-and-linux-in-your-own-linux-server-85d891e960bc
