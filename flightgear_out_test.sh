#!/bin/bash
#gibt die Flugdaten auf UDP-Port 5550 aus. Ausgabefrequenz: 60 Hz,
#Protokoll findet sich in: /usr/share/games/flightgear/Protocol/f1serial.xml
fgfs --aircraft=c172p --generic=socket,out,60,,5550,udp,f1serial --disable-ai-traffic --disable-real-weather-fetch --timeofday=noon

