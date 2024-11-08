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

# Training (this produces a file named "encodings.pickle")
python face_recognition_example/model_training.py

# Running
python main.py
```


## Remote dev

Yes, you can use the **Remote - SSH** extension in Visual Studio Code to directly edit files on your Raspberry Pi. Here's how to set it up:

1. **Install the Remote - SSH Extension**:
   - Open VS Code.
   - Go to the Extensions view by clicking on the square icon on the sidebar or pressing `Ctrl+Shift+X`.
   - Search for `Remote - SSH` and install it.

2. **Add Your Raspberry Pi to SSH Hosts**:
   - Open the Command Palette with `Ctrl+Shift+P`.
   - Type `Remote-SSH: Add New SSH Host...` and select it.
   - Enter your SSH connection string, for example:
     ```bash
     ssh pi@<RASPBERRY_PI_IP_ADDRESS>
     ```
   - Choose the SSH configuration file to update (usually it's `C:\Users\<YourUsername>\.ssh\config` on Windows).

3. **Configure SSH Key Authentication (Optional but Recommended)**:
   - Generate an SSH key pair on your desktop machine if you haven't already:
     ```bash
     ssh-keygen
     ```
   - Copy your public key to the Raspberry Pi:
     ```bash
     ssh-copy-id pi@<RASPBERRY_PI_IP_ADDRESS>
     ```

4. **Connect to Your Raspberry Pi**:
   - Open the Command Palette with `Ctrl+Shift+P`.
   - Type `Remote-SSH: Connect to Host...` and select your Raspberry Pi from the list.
   - A new window will open connected to your Raspberry Pi.

5. **Open Your Project Folder**:
   - Once connected, open the folder containing your code on the Raspberry Pi.

Now you can edit files directly on your Raspberry Pi from VS Code without needing to sync through Git. You can also run and debug your code remotely.










I want to optimize this a bit. I want to have separate processing paths for a "full scan" vs "small changes". The full scan mode is what it currently does. The "small changes" mode will only do the face detection around each region where a face was previously detected. Then the resizing will