# eng_tutor_voice_assistant
English tutor voice assistant with a friendly user interface for user interaction

Implements:
- FFmpeg tool to capture transcript from streams multimedia content.
- LangChain to prompt LLM via Groq and Memory https://python.langchain.com/docs/integrations/chat/groq/
- ElevenLabs Python library https://elevenlabs.io/docs/developer-guides/how-to-use-tts-with-streaming
- Streamlit in Python to user interface

Steps:

1) run a docker image

   $ docker container run -d -p 8501:8501 python:3.10 sleep infinity

2) install ffmpeg

apt-get update

apt-get install -y ffmpeg

ffmpeg -version

ffmpeg version 5.1.6-0+deb12u1 Copyright (c) 2000-2024 the FFmpeg developers built with gcc 12 (Debian 12.2.0-14) configuration: --prefix=/usr --extra-version=0+deb12u1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libglslang --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librist --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libsvtav1 --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --disable-sndio --enable-libjxl --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-libplacebo --enable-librav1e --enable-shared libavutil 57. 28.100 / 57. 28.100 libavcodec 59. 37.100 / 59. 37.100 libavformat 59. 27.100 / 59. 27.100 libavdevice 59. 7.100 / 59. 7.100 libavfilter 8. 44.100 / 8. 44.100 libswscale 6. 7.100 / 6. 7.100 libswresample 4. 7.100 / 4. 7.100 libpostproc 56. 6.100 / 56. 6.100 

3) pip install -r requirements.txt

4) create TSL certificates

openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365

5) create a config.toml file indicating the certificates to streamlit

mkdir .streamlit
cat .streamlit/config.toml

[server]  
sslCertFile = './cert.pem'  
sslKeyFile = './key.pem'

6) streamlit run main2.py

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.

Enter PEM pass phrase:

  You can now view your Streamlit app in your browser.

  Local URL: https://localhost:8501
  Network URL: https://172.17.0.2:8501
  External URL: https://177.194.47.227:8501