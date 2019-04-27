#!/bin/bash

systemctl disable xboxdrv.service
rm /etc/systemd/system/xboxdrv.service

systemctl disable drive_at_startup.service
rm /etc/systemd/system/drive_at_startup.service

