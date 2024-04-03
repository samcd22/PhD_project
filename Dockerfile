FROM ubuntu

# Install git
RUN apt-get update && \
    apt-get install -y git

# Clone repository
RUN git clone https://github.com/samcd22/PhD_project.git

# Set working directory
WORKDIR /PhD_project

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Copy requirements.txt file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt
