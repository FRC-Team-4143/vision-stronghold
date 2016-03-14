#!/bin/sh
echo "RUN THIS AS ROOT"

# Debug with eclipse
echo N > /sys/kernel/debug/gk20a.0/timeouts_enabled

# CPU
echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
echo 1 > /sys/devices/system/cpu/cpu0/online
echo 1 > /sys/devices/system/cpu/cpu1/online
echo 1 > /sys/devices/system/cpu/cpu2/online
echo 1 > /sys/devices/system/cpu/cpu3/online
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# GPU
echo 852000000 > /sys/kernel/debug/clock/override.gbus/rate
echo 1 > /sys/kernel/debug/clock/override.gbus/state

echo "GPU clock :" 
cat /sys/kernel/debug/clock/gbus/rate

# Memory
echo 924000000 > /sys/kernel/debug/clock/override.emc/rate
echo 1 > /sys/kernel/debug/clock/override.emc/state

echo "Memory freq:"
cat /sys/kernel/debug/clock/emc/rate

#disable auto-suspend and USB3.0
sed -i 's/usb_port_owner_info=0/usb_port_owner_info=2/' /boot/extlinux/extlinux.conf
echo -1 > /sys/module/usbcore/parameters/autosuspend
sh -c 'for dev in /sys/bus/usb/devices/*/power/autosuspend; do echo -1 >$dev; done'
