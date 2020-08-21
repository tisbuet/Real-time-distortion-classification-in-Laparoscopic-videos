# Real-time-distortion-classification-in-Laparoscopic-videos
# Folder Description

The folder should contain the followings files and folders.
Files:
1. requirements.txt
2. readme.md
3. BUET_Endgame_report.pdf
4. train.py
5. test.py
6. utils.py
7. demo_test_code.ipynb
8. ffmpeg.exe
9. Confusion_Matrix.png   	 #on 20% unseen data

Folders:
1. train data
2. test data
3. demo data
4. model
					
For the python part ,you're requested to see the 'requirements.txt' file to know if you're up to date. To ensure you're up to date, run:

`$ pip install -r requirements.txt`
	

	
# Get The Data 

1. Put all the all unzip folders in 'BUET_ENDGAME/train data' folder to train the network.

├── train data 
	├── awgn
	├── defocus_blur
	├── defocus_uneven
	├── motion_blur
	├── noise_smoke
	├── noise_smoke_uneven
	├── noise_uneven
	├── smoke
	├── smoke_uneven
	├── uneven_illum

2. Put all the all unzip folders in 'BUET_ENDGAME/test data' folder to predict the result.

├── test data 
	├── test_data1
	├── test_data2
	├── test_data3
	├── test_data4
	├── test_data5
 



# Train The Network

After the successful completation of previous two sections you may train the network once again. You can 'SKIP' this portion as it's already trained.
Team BUET_ENDGAME encourage you to retrain the model.

To extract frames and retrain the network do the followings:
			1. Open command window in 'BUET_ENDGAME' folder.
			2. Type 'python train.py -extract' and hit 'Enter'.

It will extract frames from train data and store in 'BUET_ENDGAME/Extracted_train_data_images' folder. And it will be trained eventually.

If extracted frames are present in 'BUET_ENDGAME/Extracted_train_data_images' folder already you can retrain our model again and again without extraction of frames.

So, to train the network only, in case you have extracted frames already, do the following:
			1. Open command window in 'BUET_ENDGAME' folder.
			2. Type 'python train.py' and hit 'Enter'.


After training, these files will be generated shown below:
			1. 'BUET_ENDGAME/model/trained_model.h5 model file.
			2. accuracy curve.png 
	       	3. loss curve.png 





# Test The Network 
 
To test the network do the folowings:
			1. Open command window in 'BUET_ENDGAME' folder.
			2. Type 'python test.py' and hit 'Enter'.





# The Result 

After testing, these files will be generated in current directory shown below:

	       	1. result.csv     ## predicted result on test data
	

		
# Demo code

An interactive notebook file named 'demo_test_code.ipynb' has been attached. To test the demo code, you have to store video files in 'demo data' folder. There will be only video files. There should not be any subfolders in 'demo data' folder.

├── demo data├── video1_2.avi
 	     ├── video3_2.avi
	     ├── video4_2.avi
 


# For Any Query
You're welcome to contact us:
	1. tisbuet@gmail.com
	2. shouborno@ieee.org

