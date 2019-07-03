FROM python:slim-stretch
MAINTAINER Ivana Milovanovic <milovanovichivana@gmail.com>
LABEL Description "Perfect Shot App"

# Update the OS & install packages
RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev python python-dev python-pip python3 python3-dev python3-pip

# Copy the files into the container
COPY . /usr/src/perfect_shot/

# Set the working directory
WORKDIR /usr/src/perfect_shot/flask_app

# Install the dependencies
RUN pip3 install Flask numpy gevent pillow scipy matplotlib opencv-python scikit-learn pandas imutils dlib

# Expose the port
EXPOSE 5000

# Run the app
# CMD [ "bash" , "scripts/flask_app.sh"]
CMD [ "python3" , "app.py"]
