#!/bin/sh
echo 'bright'
uvcdynctrl -d video0 --set='Brightness' 5
uvcdynctrl -d video1 --set='Brightness' 5
uvcdynctrl -d video2 --set='Brightness' 5
echo 'contrast'
uvcdynctrl -d video0 --set='Contrast' 10
uvcdynctrl -d video1 --set='Contrast' 10
uvcdynctrl -d video2 --set='Contrast' 10
echo 'saturation'
uvcdynctrl -d video0 --set='Saturation' 200
uvcdynctrl -d video1 --set='Saturation' 200
uvcdynctrl -d video2 --set='Saturation' 200
echo 'white'
uvcdynctrl -d video0 --set='White Balance Temperature, Auto' 0
uvcdynctrl -d video1 --set='White Balance Temperature, Auto' 0
uvcdynctrl -d video2 --set='White Balance Temperature, Auto' 0
echo 'exposure auto'
uvcdynctrl -d video0 --set='Exposure, Auto' 1
uvcdynctrl -d video1 --set='Exposure, Auto' 1
uvcdynctrl -d video2 --set='Exposure, Auto' 1
echo 'exposure'
#uvcdynctrl -d video0 --set='Exposure (Absolute)' 5  # very dark
uvcdynctrl -d video0 --set='Exposure (Absolute)' 10  # full sunlight
uvcdynctrl -d video1 --set='Exposure (Absolute)' 10  # full sunlight
uvcdynctrl -d video2 --set='Exposure (Absolute)' 10  # full sunlight
#uvcdynctrl --set='Exposure (Absolute)' 20  # afternoon
