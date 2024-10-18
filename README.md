# Rescue Drone with Autonomous Grid Survey and GSM Communication

This project is focused on building an autonomous rescue drone that surveys disaster areas based on predefined boundaries provided via telemetry. The drone detects people using a camera and a TensorFlow Lite model, sending their location to a rescue team using a GSM 900A MINI module. After completing the grid survey, the drone returns to the launch point.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Hardware Components](#hardware-components)
- [Software Setup](#software-setup)
- [Machine Learning Model](#machine-learning-model)
- [Usage Instructions](#usage-instructions)
- [Future Improvements](#future-improvements)

## Overview
The rescue drone autonomously surveys a disaster area based on GPS boundaries provided via telemetry. When the drone detects a person during the grid survey, it sends their coordinates via SMS using the GSM 900A MINI module. Once the grid survey is complete, the drone returns to its original launch point.

## Features
- **Autonomous Grid Survey**: Survey boundaries are defined through telemetry, and the drone autonomously navigates the area in a grid pattern.
- **Person Detection**: A TensorFlow Lite model detects people in real time using a camera.
- **Real-time Alerts**: The GSM 900A MINI module sends SMS alerts with the detected person’s coordinates.
- **Return to Launch (RTL)**: The drone automatically returns to the launch point upon completing the grid survey.

## Hardware Components
- **Orange Cube Plus Flight Controller**
- **HERE+ GPS for precise positioning**
- **Raspberry Pi 4B+ for onboard processing**
- **Google Coral USB Accelerator for edge AI**
- **SIM900A MINI GSM Module for communication**
- **Camera for real-time person detection**

## Software Setup
1. **Flight Controller Setup**:
   - Configure the Orange Cube Plus using Mission Planner.
   - Define the survey boundaries using telemetry input.
2. **Raspberry Pi Setup**:
   - Install Raspberry Pi OS and necessary libraries.
   - Clone this repository and install required dependencies:
   - Note:- Make Sure that git for windows installed
     ```bash
     git clone https://github.com/ShivaTsavatapalli5/AIDrone.git
     cd AIDrone
     ./requirements.sh
     ```
   - Ensure TensorFlow Lite and Google Coral USB are properly set up for person detection.
3. **GSM Module Setup**:
   - Configure the SIM900A GSM module to send SMS alerts with GPS coordinates.

## Machine Learning Model
- **Model**: TensorFlow Lite (edgetpu.tflite) trained for person detection.
- **Hardware Accelerator**: Google Coral USB for fast inference.
- **Input**: Live camera feed from the drone.
- **Output**: SMS alert with GPS coordinates of detected person.

## Usage Instructions
1. **Power on the Drone**: Ensure that all hardware components are powered and connected.
2. **Input Survey Boundaries**: Use telemetry to send the GPS boundaries for the survey.
3. **Start the Survey**: The drone will autonomously perform the grid survey.
4. **Detection Alerts**: When a person is detected, the drone will send an SMS with their location via the GSM module.
5. **Return to Launch**: After completing the grid survey, the drone will return to its launch point.

## Future Improvements
- **Enhanced Obstacle Avoidance**: Add more sensors for better navigation.
- **Live Video Streaming**: Implement real-time video feed to monitor the drone’s survey.
- **Extended Flight Time**: Optimize power consumption to extend flight duration.
- **LoRa** : Upgrade to LoRa communication in order to get rid off mobile network usage
