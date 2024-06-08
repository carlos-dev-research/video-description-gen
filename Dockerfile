FROM python:3.10

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    ca-certificates \
    software-properties-common \
    build-essential \
    wget \
    python3-pip \
    ffmpeg

# Install nvidia toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-5

# Install Python dependencies
COPY requirements.txt /dependencies/requirements.txt
RUN pip3 install -r /dependencies/requirements.txt

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN mkdir -p /var/log/ollama
RUN nohup ollama start &> /var/log/ollama/ollama.log & sleep 10 && ollama pull llava
# Copy the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# Ensure the entrypoint script is executable
RUN chmod +x /usr/local/bin/entrypoint.sh


COPY ./App /App 
WORKDIR /App
RUN python3 whisper-first-run.py
EXPOSE 7860

# Set the entrypoint script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]