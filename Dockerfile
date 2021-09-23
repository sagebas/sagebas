# Create a docker image that contains Python.
# The image can be used to run the SPIDER GGF forecast application.

# Use a specific version of Python
FROM python:3.8.5

# Set working directory inside the Docker build
WORKDIR /run_environment/

#Clone the github repository to the working directory in the Docker environment
RUN git clone https://github.com/robert-m-shore/GGF_realtime_forecast .

# Take all the files located in the current directory and copy them into the Docker image
COPY . .

# Install the Python packages we'll use
RUN pip install --no-cache-dir -r ./requirements.txt

# Alter permissions
RUN chmod +x ./GGF_RTF.py
RUN chmod +x ./hello_world.py

# Run the SPIDER GGF_RTF Python script
CMD [ "python", "./GGF_RTF.py" ]
