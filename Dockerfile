# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# For this simple example, no additional packages are needed

RUN pip install -r requirements.txt

# Run app.py when the container launches
CMD ["python", "./app.py"]