#!/bin/sh
echo 'bright'
uvcdynctrl --set='Brightness' 5
echo 'contrast'
uvcdynctrl --set='Contrast' 10
echo 'saturation'
uvcdynctrl --set='Saturation' 200
echo 'white'
uvcdynctrl --set='White Balance Temperature, Auto' 0
echo 'exposure auto'
uvcdynctrl --set='Exposure, Auto' 1
echo 'exposure'
#uvcdynctrl --set='Exposure (Absolute)' 5  # very dark
uvcdynctrl --set='Exposure (Absolute)' 10  # full sunlight
#uvcdynctrl --set='Exposure (Absolute)' 20  # afternoon
