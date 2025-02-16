# ring-year-wizard

## building it

clone this repo  
cd to that folder  
build the image from the Dockerfile  
this builds the image and calls it 'ring-year-wizard', using the file called Dockerfile (default)  

`docker build -t ring-year-wizard .`

this pulls down the miniconda image and then conda install a few modules
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

# running it

when the container is started as above, you will be left in an interactrive shell within that container:  

`(base) root@5449e5e0db96:/app#`  

from here, cd to the `code` folder, where the prediction command can be executed

`(base) root@5449e5e0db96:/app/code#`  

the `repeat_predict.py` script is the single command that will pre-process the image, do the prediction, and output the results.  

so the typical usage is this:  

`(base) root@5449e5e0db96:/app/code# python repeat_predict.py -m look -p serial`   

there are are 2 important switches for the script.  

`-m look` - this is `look` mode, and will look for images in the `images/look` folder.  

`-p parallel` or  `-p serial`  
this controls the prediction path among the two models (custom and openai).  

### `parallel`  
takes the result of the pre-processed image and gives them to the 2 models in parallel.  
the results from both models are in the `output`.

### `serial`  
first uses openai model to determine proper orientation, and those results used to predict on the properly oriented image.  

### output
the output will be in 3 places.  
`file: output/ring-pred.json`  
`stdout:` - printed to the console  
`log:` - `logs/app-YYYYMMDDHHMMSS.log` - each new 'run' will create a new timestamped log file  

### image pre-processing description
after a lot of testing and discovery, this is the approach that was settled on.  
- look for a big ellipse which roughly isolates the stone on the ring
- make a mask using that ellipse, then turn the area outside the ellipse black
- align that ellipse with the major axis in the vertical direction
- move it to the center and stretch to full H
- (if a reasonable ellipse could not be detected, it just passes the image through as is)
- make a duplicate that is rotated 180 degrees
- this process was done on all the images before training
- so that is the process used for pre-processing

to see these pre-preocessed images, they are stored in `images/tmp/e_img.jpg` and `images/tmp/flip_img.jpg` during operation (and overwritten each new prediction).



