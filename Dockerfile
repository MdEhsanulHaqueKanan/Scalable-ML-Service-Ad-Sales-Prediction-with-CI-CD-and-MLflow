# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container first to leverage Docker layer caching.
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# 5. Copy the trained model into the container's application directory
# This makes the image self-contained and ready for deployment.
COPY ./model /app/model

# 6. Copy the rest of the application's code into the container
# This includes the 'src', 'templates', and 'static' directories.
COPY . .

# 7. Expose the port the app runs on to the outside world
EXPOSE 5001

# 8. Define the command to run the application using a production-ready Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "src.app:app"]