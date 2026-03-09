#!/bin/bash
 echo "Se porneste detect_boxes.py"

python3 ~/hailo/detect_boxes.py

echo "Se porneste Hailo"

cd ~/hailo-rpi5-examples || exit

source setup_env.sh

python3 basic_pipelines/detection_simple.py --input usb

echo"Program finalizat"
