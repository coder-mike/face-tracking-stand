Following this tutorial:

https://core-electronics.com.au/guides/face-recognition-with-raspberry-pi-and-opencv/

```sh
ssh michael@192.168.86.27
```

Password is "michael".

```sh
source ./servo_env/bin/activate

# Installs
sudo apt update
sudo apt full-upgrade
pip install opencv-python
pip install imutils
sudo apt install cmake
pip install face-recognition # Takes a long time (10 to 30 minutes)
sudo apt-get install python3-smbus
sudo apt-get install i2c-tools
sudo pip install adafruit-circuitpython-servokit

# Training
python face_recognition_example/model_training.py

# Running
python main.py
```
