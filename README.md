# Computer Vision and Image Processing - Lab Sessions
To build the docker images, clone the repository, cd to the directory with the Dockerfile and run:
```
docker build . -t cvlab
```
To run the docker image:
```
docker run -v `pwd`:/home/cvlab -p 8888:8888 -it cvlab:latest
```
Inside the docker image, to start jupyter run:
```
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```
Click on the link highlighted in the following picture:
![jupyter](Images/jupyter.png)
