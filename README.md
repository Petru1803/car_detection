# car_detection

git clone https://github.com/hailo-ai/hailo-rpi5-examples.git

cd hailo-rpi5-examples

./install.sh

source setup_env.sh (de fiecare data cand se deschide un terminal nou)

//camera usb
python basic_pipelines/detection.py --input usb

//camera rpi
python basic_pipelines/detection.py --input rpi

Pentru detectarea numarului de masini care trec printr-o anumita locatie se ruleaza detect_boxes.py intr-un nou terminal:
python3 detect_boxes.py --left 100 --top 100 --width 640 --height 480 (variabilele se seteaza in functie de pozitia AI-ului pe ecran)

