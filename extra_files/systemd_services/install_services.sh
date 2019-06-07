#!/bin/bash

# The file is copied instead of a symlink for security issues. The Github repository is not particuarly secure, and this way the systemd service (which runs as root) won't automatically update.

# have the xbox remote driver run at startup
cp xboxdrv.service /etc/systemd/system/xboxdrv.service
systemctl enable xboxdrv.service
systemctl restart xboxdrv.service


# run drive code at startup if the USB stop button or logitech USB controller is plugged in
cp drive_at_startup.service /etc/systemd/system/drive_at_startup.service
systemctl enable drive_at_startup.service

# blnking to arduino commands
cp blinking.service /etc/systemd/system/blinking.service
systemctl enable blinking.service

# let the rock user run ```sudo /sbin/modprobe pcspkr``` without a password. This is required to make beeps with the industrial motherboard's built in speaker (we love legacy hardware)
cp spkr_sudoers /etc/sudoers.d/.

sudo apt install -y beep
