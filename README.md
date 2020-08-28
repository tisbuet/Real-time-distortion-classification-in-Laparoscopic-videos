# Real-time distortion classification in Laparoscopic videos
It's not a traditional video/image classification problem, where recognition of pattern is primary goal. Rather, these frames of every videos have been destroyed using noisy parameters like white gaussian noise, smokes, uneven illuminations, blurriness due to defocus and motions, and combinations of these noisy features. And our motive of this project is to classify the noise or distortion features. So, the goal of recognizing pattern has become second part of this project, where our primary goal has become the extraction of noisy features.  This distortion classification problem has many aspects in Biomedical and Signal processing area specially to enhance the video quality of laparoscopy, endoscopy.

### Dataset source
Here is a link below to download 10 seperated zip files for 10 classes of noise.
> https://drive.google.com/drive/folders/1NxU3BRj_qk8MepHXBkOdQ4lFNn6tYujd

> You will found a brief description of the dataset [here](https://drive.google.com/file/d/1zzdwdvvI834xi6E6gKMxG3H5p6ls9Ibs/view?usp=sharing) 

*The ICIP LVQ Challenge dataset is publicly released under the Creative Commons licence CC-BY-NC-SA 4.0. This implies that:*
*- the dataset cannot be used for commercial purposes,*
*- the dataset can be transformed (additional annotations, etc.),*
*- the dataset can be redistributed as long as it is redistributed under the same license with the obligation to cite the contributing work which led to the* *generation of the LVQ and cholec80 datasets (mentioned above).*

*By downloading and using this dataset, you agree on these terms and conditions.*
### Folder Description

> ffmpeg.exe is necessary in current directory. So, make sure you've downloaded, or you can download it from [here](https://ffmpeg.org/ "here").

The folder should contain the followings files and folders.

Files:
1. requirements.txt
2. README.md
3. BUET_Endgame_report.pdf
4. train.py
5. test.py
6. utils.py
7. demo_test_code.ipynb
8. ffmpeg.exe
9. Confusion_Matrix.png (on 20% unseen data)

Folders:
1. train data
2. test data
3. demo data
4. model
					
For the python part ,you're requested to see the 'requirements.txt' file to know if you're up to date. To ensure you're up to date, run:

`$ pip install -r requirements.txt`
	

	
### Dataset location

1. Put all the all unzip folders in **BUET_ENDGAME/train data** folder to train the network.

##### train data 
- awgn
- defocus_blur
- defocus_uneven
- motion_blur
- noise_smoke
- noise_smoke_uneven
- noise_uneven
- smoke
- smoke_uneven
- uneven_illum

2. Put all the all unzip folders in **BUET_ENDGAME/test data** folder to predict the result.

##### test data 
- test_data1
- test_data2
- test_data3
- test_data4
- test_data5
 



### Train The Network

After the successful completation of previous two sections you may train the network once again. You can 'SKIP' this portion as it's already trained.
Team BUET_ENDGAME encourage you to retrain the model.

To extract frames and retrain the network do the followings:
1. Open command window in **BUET_ENDGAME** folder.
2. Type `$ python train.py -extract` and hit **Enter**.

It will extract frames from train data and store in **BUET_ENDGAME/Extracted_train_data_images** folder. And it will be trained eventually.

If extracted frames are present in **BUET_ENDGAME/Extracted_train_data_images** folder already you can retrain our model again and again without extraction of frames.

So, to train the network only, in case you have extracted frames already, do the following:
1. Open command window in **BUET_ENDGAME** folder.
2. Type `$ python train.py` and hit **Enter**.


After training, these files will be generated shown below:
1. **BUET_ENDGAME/model/trained_model.h5** model file.
2. accuracy curve.png 
3. loss curve.png 





### Test The Network 
 
To test the network do the folowings:
1. Open command window in **BUET_ENDGAME** folder.
2. Type `$ python test.py` and hit **Enter**.





### The Result 

After testing, these files will be generated in current directory shown below:

- result.csv (predicted result on test data)
	

		
### Demo code

An interactive notebook file named `demo_test_code.ipynb` has been attached. To test the demo code, you have to store video files in **demo data** folder. There will be only video files. There should not be any subfolders in **demo data** folder.

##### demo data
- video1_2.avi
- video3_2.avi
- video4_2.avi
 


### For Any Query
You're welcome to contact us:
1. tisbuet@gmail.com
2. shouborno@ieee.org
