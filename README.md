# ring-year-wizard

## operation

clone this repo  
cd to that folder  
build the image from the Dockerfile  
this builds the image and calls it 'ring-year-wizard', using the file called Dockerfile (default)  

`docker build -t ring-year-wizard .`

this pull down the miniconda image and then conda install a few modules
it takes a while the first time the image is built  
subsequent builds, with changes to just the ENTRYPOINT, etc. or initial RUN commands, are very fast  

this host command will run a container using the image built above, in interactive mode.  
it also 'volume-bind's the local folder to the container's /app folder, and starts a bash session in the container  
`docker run -it -v$ $(pwd):/app ring-year-wizard bash`

the volume-bind in windows is this:  
`docker run -it -v $%cd%:/app ring-year-wizard bash`

there are a variety of ways to start the container with an initial command.  
initial testing had spotty results, so this is still under review.  
this may have worked:  

`docker run -v$(pwd):/app ring-year-wizard bash -c "cd code && python repeat_predict.py -m look"`

this does what it looks like:
- runs the container using the image, but not in interactive mode
- volume-binds to the folder it was run from
- starts a bash session, cds to the /app/code folder (the volume-bind drops it in the containers /app)
- then runs the python script that continually will process images from the /look folder
- best to use the flag.txt mechanism as a file traffic cop

