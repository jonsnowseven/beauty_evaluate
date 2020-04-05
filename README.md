## Beauty Evalutation
evaluate the attractiveness of a face. Goal: sort of discover the facial features contributing to beauty

### Structure
- Beau: Folder with all images and annotations of index
- images: Testing images
- beauty.ipynb: notebook where we got the illustrations in my article, you can get the visualization of the model in there
- model.py: model/architecture we used
- haarcascade_frontalface_default.xml: detection of face
- history.pickle: history of our training of model
- use.py: main script to test out the model

### Usage
You need to first get the weight "beau.h5" you can ask me through email for that (Please also consider following my [medium](https://medium.com/@michaelchan_2146) and github!) and then put the weight in the root folder of the repository.
```
python3 use.py path_to_image
```

### Results
![alt text](https://github.com/miki998/beauty_evaluate/outscarlett.jpg "Logo Title Text 1")
