## Picking file image from docker hub
FROM python:3-slim 
EXPOSE 5002
# Inside a container create a app folder
WORKDIR /app  
# Whatever is inside our working directory. copy it to the app folder inside a container 
COPY . /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# During debugging, this entry point will be overridden. Gunicorn web server
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]
