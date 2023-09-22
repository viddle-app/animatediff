FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx pkg-config libglib2.0-0 \
  libavformat-dev libavcodec-dev libavdevice-dev \
  libavutil-dev libavfilter-dev libswscale-dev \
  libswresample-dev build-essential libpcap-dev \
  libpq-dev postgresql-client \
  git portaudio19-dev 

RUN apt install ffmpeg -y

RUN rm -rf /usr/local/cuda/lib64/stubs

RUN useradd -m huggingface

USER huggingface

WORKDIR /home/huggingface

RUN mkdir -p /home/huggingface/data

COPY data/stable-diffusion-xl-base-1.0 /home/huggingface/data/stable-diffusion-xl-base-1.0/
COPY data/stable-diffusion-xl-refiner-1.0 /home/huggingface/data/stable-diffusion-xl-refiner-1.0/
COPY data/whisper-large-v2 /home/huggingface/data/whisper-large-v2

RUN python -m pip install pip --upgrade
COPY requirements.txt /home/huggingface
RUN pip install -r requirements.txt 

ENV USE_TORCH=1

RUN mkdir -p /home/huggingface/.cache/huggingface \
  && mkdir -p /home/huggingface/input \
  && mkdir -p /home/huggingface/output

COPY src/ /home/huggingface/src/

COPY watermark.png /home/huggingface/watermark.png

COPY run.sh /home/huggingface/run.sh

ENV PATH="/home/huggingface/.local/bin:${PATH}"

ENTRYPOINT [ "/home/huggingface/run.sh" ]
