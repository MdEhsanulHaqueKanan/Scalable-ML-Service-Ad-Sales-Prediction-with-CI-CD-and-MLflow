# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
# This is done first to leverage Docker layer caching.
# The dependencies layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# 5. Copy the rest of the application's code into the container
# This includes the 'src', 'data', and 'mlruns' directories.
COPY . .

# 6. Expose the port the app runs on to the outside world
EXPOSE 5001

# 7. Define the command to run the application using Gunicorn
# This tells Gunicorn to look inside the 'src' package for the 'app' module
# and run the Flask object named 'app'.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "src.app:app"]