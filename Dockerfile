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

RUN python -m pip install pip --upgrade
COPY requirements.txt /home/huggingface
RUN pip install -r requirements.txt 

ENV USE_TORCH=1

RUN mkdir -p /home/huggingface/.cache/huggingface \
  && mkdir -p /home/huggingface/input \
  && mkdir -p /home/huggingface/output

COPY motion-models/mm_sd_v15_v2.ckpt /home/huggingface/motion-models/mm_sd_v15_v2.ckpt
COPY base_models/v1-5 /home/huggingface/base_models/v1-5
COPY data/latent_cache /home/huggingface/data/latent_cache
COPY data/output.csv /home/huggingface/data/output.csv
COPY outputs/training-2023-10-25T12-47-50/checkpoints/checkpoint-epoch-10.ckpt /home/huggingface/outputs/training-2023-10-25T12-47-50/checkpoints/checkpoint-epoch-10.ckpt

COPY src/ /home/huggingface/src/

COPY run.sh /home/huggingface

COPY train_overlapping.py /home/huggingface/train_overlapping.py
COPY config/* /home/huggingface/config/

ENV PATH="/home/huggingface/.local/bin:${PATH}"

ENTRYPOINT [ "/home/huggingface/run.sh" ]
