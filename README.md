# Future Frame Prediction

## Overview

This repository contains code and resources for future frame prediction in video sequences. The goal of future frame prediction is to generate future frames of a video based on the past frames. This technique is useful in various applications such as video compression, anomaly detection, and autonomous driving.

## Project Structure

The repository is structured as follows:

- `simvp-gsta/` - Contains the SimVP model for frame prediction.
- `u-net/` - Contains the U-Net model for frame segmentation.

## Models

### SimVP (Simple Video Prediction)

SimVP is a straightforward yet effective model for future frame prediction. It is designed to balance performance and computational efficiency.

### U-Net

U-Net is a convolutional neural network that has been adapted for video frame segmentation. It uses a symmetric encoder-decoder structure with skip connections to preserve spatial information.

## Usage

### Installation

To run the models in this repository, you'll need Python and the required dependencies. You can install the dependencies using the following command:

```sh
pip install -r requirements.txt 
