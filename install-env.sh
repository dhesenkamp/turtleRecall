#!/bin/bash


# check for M1 chip, abort if not detected
if [[ `uname -m` == 'arm64' ]]; then
  echo Cannot detect M1 chip. Please make sure you are running this script on an Apple Silicon device
  exit 1
fi

echo M1 chip detected. Proceeding with creation of Tensorflow environment.

# install XCode command line tools
xcode select --install

# install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# create environment with python==3.9
conda create --name turtleRecall python=3.9
conda activate turtleRecall

# install Tensorflow dependencies from Apple
conda install -c apple tensorflow-deps

# install Tensorflow for MacOS
pip install tensorflow-macos

# install Tensorflow Metal for GPU support
pip install tensorflow-metal