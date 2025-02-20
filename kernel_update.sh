#!/usr/bin/env bash
# Update the kernel patch

pushd $HOME
git clone https://github.com/Soheil-ab/Orca.git
pushd Orca/linux

sudo dpkg -i linux-header*
sudo dpkg -i linux-image*

# Update default grub
# Replace GRUB_DEFAULT=0 with GRUB_DEFAULT='1>2'
sudo sed -i "s/GRUB_DEFAULT=0/GRUB_DEFAULT='1>2'/g" /etc/default/grub

sudo update-grub
sudo reboot