#!/bin/bash
echo -1 > /sys/module/usbcore/parameters/autosuspend
#echo -1 > /sys/kernel/debug/tegra_hdmi/hotplug
#echo 4 > /sys/class/graphics/fb0/blank

cd /home/ubuntu/vision-stronghold/
./scripts/camerasettings.sh

echo "Vision program starting..."

until ./bin/Vision2016 ; do
#    if [ $? -eq 1 ]; then
#        echo "Vision program exited with code $? (safe shutdown via signal). Closing wrapper."
#        exit 1
#    fi
    echo "Vision program crashed with code $?. Respawning..."
    sleep 1
done
