# InMind-Project
 Final ML project for InMind Academy


Documentation steps that i will organize later:

steps for Part 2: -1 (Train an object detection model based on YOLOv5):

git clone https://github.com/ultralytics/yolov5
cd yolov5

then I used "pip install --no-deps -r requirements.txt" for yolov5 dependencies installation to not modify my already existing version because i want to use cuda.

Step 2: Prepare My Dataset

YOLOv5 requires annotations in a specific format. Each image should have a corresponding .txt file with annotations, where each line represents an object in the image in the format:

php-template
Copy
<class_id> <x_center> <y_center> <width> <height>

so i need to make sure that my images and annotation files are organized in the following structure:
/dataset
    /images
        /train
            image1.jpg
            image2.jpg
            ...
        /val
            val_image1.jpg
            val_image2.jpg
            ...
    /labels
        /train
            image1.txt
            image2.txt
            ...
        /val
            val_image1.txt
            val_image2.txt
            ...

then:
Step 3: Create a Custom Data YAML File

Create a .yaml file to define my dataset. This file specifies the paths to my training and validation data and the class names. For example, dataset.yaml:

yaml
Copy
train: /path/to/dataset/images/train
val: /path/to/dataset/images/val

nc: <number_of_classes>  # Number of classes in my dataset
names: ['class1', 'class2', 'class3']  # List of class names

and finally:
Step 4: Train the Model
Now, you're ready to train the model. To start the training process, use the following command in my terminal:

bash
Copy
python train.py --img 640 --batch 16 --epochs 50 --data /path/to/dataset.yaml --weights yolov5s.pt --cache
Explanation of parameters:

--img 640: Image size (you can adjust this depending on my dataset).

--batch 16: Batch size (you can adjust based on available GPU memory).

--epochs 50: Number of epochs (adjust based on how long you want to train).

--data /path/to/dataset.yaml: Path to my dataset YAML file.

--weights yolov5s.pt: The pre-trained weights file (you can use yolov5s.pt or yolov5m.pt, etc. depending on the size of the model you prefer).

--cache: Cache images for faster training.