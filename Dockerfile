FROM python:3.10 AS build
RUN python3 -m venv /venv

# example of a development library package that needs to be installed
# RUN apt-get -qy update && apt-get -qy install libldap2-dev && \
#     rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# install requirements separately to prevent pip from downloading and
# installing pypi dependencies every time a file in your project changes
ADD ./requirements /project/requirements
ARG REQS=requirements
RUN /venv/bin/pip install -r project/requirements/$REQS.txt

# install the project, basically copying its code, into the virtualenv.
# this assumes the project has a functional setup.py
# ADD . ./project
# WORKDIR /project
ADD . /project
RUN /venv/bin/pip install /project

# this won't have any effect on our production image, is only meant for
# if we want to run commands like pytest in the build image
# WORKDIR /project


# the second, production stage can be much more lightweight:
FROM python:3.10-slim AS production
COPY --from=build /venv /venv

ENV PATH="/venv/bin:$PATH"

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# install runtime libraries (different from development libraries!)
# RUN apt-get -qy update && apt-get -qy install libldap-2.4-2 && \
#     rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# remember to run python from the virtualenv
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:server
