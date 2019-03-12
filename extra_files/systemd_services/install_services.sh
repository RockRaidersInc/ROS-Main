#!/bin/bash

# The file is copied instead of a symlink for security issues. The Github repository is not particuarly secure, and this way the systemd service (which runs as root) won't automatically update.
cp xboxdrv.service /etc/systemd/system/xboxdrv.service
systemctl enable xboxdrv.service
systemctl restart xboxdrv.service
