#!/bin/sh
#./mjpg_streamer -i "./input_uvc.so --device /dev/video1 -f 10 -r 160x120" -o "./output_http.so --port 5801 -w www"
#./mjpg_streamer -i "./input_uvc.so --device /dev/video1 -f 10 -r 176x144" -o "./output_http.so --port 5801 -w www"



cd /home/ubuntu/mjpg-streamer/mjpg-streamer

rm ~/streamerlog.txt2
mv ~/streamerlog.txt1 ~/streamerlog.txt2
mv ~/streamerlog.txt ~/streamerlog.txt1

until ./mjpg_streamer -i "./input_uvc.so --device /dev/video1 -f 10 -r 320x240" -o "./output_http.so --port 5801 -w www"  >> /home/ubuntu/streamerlog.txt ; do
#    if [ $? -eq 1 ]; then
#        echo "Vision program exited with code $? (safe shutdown via signal). Closing wrapper."
#        exit 1
#    fi
    echo "stream program crashed with code $?. Respawning..."
    sleep 1
done

