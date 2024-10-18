#!/bin/bash

# Install necessary Python packages
pip3 install tensorflow
pip3 install dronekit
pip3 install opencv-python
pip3 install mavproxy

# Add Coral Edge TPU repository to the system sources
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Add the GPG key for the Coral Edge TPU repository
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update package lists
sudo apt-get update

# Install Coral Edge TPU libraries and Python bindings
sudo apt-get install libedgetpu1-std
sudo apt-get install python3-pycoral
