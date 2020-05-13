FROM pytorch/pytorch:latest

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt 
RUN pip install --upgrade --no-cache-dir -r requirements.txt && \     
	rm requirements.txt

RUN git clone https://github.com/NVIDIA/apex && cd apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /home
