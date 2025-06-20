import tensorflow as tf
import cv2 as cv
import numpy as np
import sys
import os


# Params:
Loss = tf.keras.losses.Huber()
Metrics = ["mse"]


def Main():
    Args = sys.argv

    for File_Name in Args[1:]:
        Process_Image(File_Name)


def Process_Image(File_Name: str):
    print(f"File_Name = \"{File_Name}\"")
    
    Base_Directory = os.path.dirname(File_Name)
    print(f"Base_Directory = \"{Base_Directory}\"")
    
    Sub_Directory = os.path.join(Base_Directory, "Upscale")
    print(f"Sub_Directory = \"{Sub_Directory}\"")
    
    File_Name_No_Path = os.path.basename(File_Name)
    print(f"File_Name_No_Path = \"{File_Name_No_Path}\"")
    
    File_Name_No_Path_No_Extension = os.path.splitext(File_Name_No_Path)[0]
    print(f"File_Name_No_Path_No_Extension = \"{File_Name_No_Path_No_Extension}\"")
    
    File_Name_No_Extension_Sub_Directory = os.path.join(Sub_Directory, File_Name_No_Path_No_Extension)
    print(f"File_Name_No_Extension_Sub_Directory = \"{File_Name_No_Extension_Sub_Directory}\"")

    Output_File_Name = File_Name_No_Extension_Sub_Directory + ".png"
    print(f"Output_File_Name = \"{Output_File_Name}\"")

    Image = cv.imread(File_Name)
    
    Image = AI_Upscale(Image)
    cv.imwrite(Output_File_Name, Image)
    print(f"saved \"{Output_File_Name}\" after first AI upscale")
    
    Image = Raw_Upscale(Image)
    cv.imwrite(Output_File_Name, Image)
    print(f"saved \"{Output_File_Name}\" after first raw upscale")
    
    Image = Blur(Image)
    cv.imwrite(Output_File_Name, Image)
    print(f"saved \"{Output_File_Name}\" after first blur")
    
    Image = AI_Upscale(Image, Scaling_Factor=4, Reference_Range=4)
    cv.imwrite(Output_File_Name, Image)
    print(f"saved \"{Output_File_Name}\" after second AI upscale")
    
    Image = Raw_Upscale(Image)
    cv.imwrite(Output_File_Name, Image)
    print(f"saved \"{Output_File_Name}\" after second raw upscale")
    
    Image = Blur(Image)
    
    cv.imwrite(Output_File_Name, Image)
    print(f"saved \"{Output_File_Name}\" after final blur")


def Raw_Upscale(Image: np.ndarray, Scaling_Factor=2) -> np.ndarray:
    return cv.resize(Image, [Image.shape[0]*Scaling_Factor, Image.shape[0]*Scaling_Factor])


def Blur(Image: np.ndarray, Radius=1) -> np.ndarray:
    return cv.blur(Image, [2*Radius + 1, 2*Radius + 1])


def AI_Upscale(Picture: np.ndarray, Scaling_Factor=2, Reference_Range=2, Layer_1_Nodes=64, Epochs=500) -> np.ndarray:
    Original_Picture = Picture

    Original_Shape = Original_Picture.shape
    print(f"Original_Shape = {Original_Shape}")

    Small_Picture = cv.resize(Original_Picture, [Original_Shape[0]//Scaling_Factor, Original_Shape[0]//Scaling_Factor])
    Small_Shape = Small_Picture.shape
    print(f"Small_Shape = {Small_Shape}")

    # Set up training data    
    In_Train = np.zeros([Small_Shape[0]*Small_Shape[1], 2*Reference_Range + 1, 2*Reference_Range + 1, Original_Shape[2] + 1], np.float64)

    Out_Train = np.zeros([Small_Shape[0]*Small_Shape[1], Scaling_Factor, Scaling_Factor, Original_Shape[2]], np.float64)

    ## Iterate over smaller image
    for X in range(Small_Shape[0]):
        for Y in range(Small_Shape[1]):

            ### Iterate over valid pixels in reference area
            for X_ in range(2*Reference_Range + 1):
                Pos_X = X + X_ - Reference_Range
                if Pos_X < 0 or Pos_X >= Small_Shape[0]:
                    continue
                for Y_ in range(2*Reference_Range + 1):
                    Pos_Y = Y + Y_ - Reference_Range
                    if Pos_Y < 0 or Pos_Y >= Small_Shape[1]:
                        continue

                    #### Set pixel data
                    for Channel in range(Original_Shape[2]):
                        In_Train[Y*Small_Shape[0] + X, X_, Y_, Channel] = Small_Picture[Pos_X, Pos_Y, Channel] / 255
                    In_Train[Y*Small_Shape[0] + X, X_, Y_, Original_Shape[2]] = 1

            ### Iterate over pixels in original (upscale relative to small) which correspond to position
            for X_ in range(Scaling_Factor):
                Pos_X = Scaling_Factor*X + X_
                for Y_ in range(Scaling_Factor):
                    Pos_Y = Scaling_Factor*Y + Y_

                    #### Set pixel data
                    for Channel in range(Original_Shape[2]):
                        Out_Train[Y*Small_Shape[0] + X, X_, Y_, Channel] = Original_Picture[Pos_X, Pos_Y, Channel] / 255

    # Create NN
    
    Model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(2*Reference_Range + 1, 2*Reference_Range + 1, Original_Shape[2] + 1)),
        tf.keras.layers.Dense(Layer_1_Nodes, activation="relu"),
        tf.keras.layers.Dense(Scaling_Factor * Scaling_Factor * Original_Shape[2], activation="sigmoid"),
        tf.keras.layers.Reshape((Scaling_Factor, Scaling_Factor, Original_Shape[2]))
    ])
    """
    Model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                               input_shape=(2*Reference_Range + 1, 2*Reference_Range + 1, Original_Shape[2] + 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(Scaling_Factor * Scaling_Factor * Original_Shape[2], activation='sigmoid'),
        tf.keras.layers.Reshape((Scaling_Factor, Scaling_Factor, Original_Shape[2]))
    ])
    """
    Model.compile(
        optimizer="adam",
        loss=Loss,
        metrics=Metrics
    )

    Data_Set = tf.data.Dataset.from_tensor_slices((In_Train, Out_Train))

    Data_Set = Data_Set.shuffle(buffer_size=In_Train.shape[0]).batch(32)

    Model.fit(Data_Set, epochs=Epochs)

    Large_Picture = cv.resize(Original_Picture, [Original_Shape[0]*Scaling_Factor, Original_Shape[0]*Scaling_Factor])
    Input = np.zeros([Original_Shape[0]*Original_Shape[1], 2*Reference_Range + 1, 2*Reference_Range + 1, Original_Shape[2] + 1])

    ## Iterate over smaller image
    for X in range(Original_Shape[0]):
        for Y in range(Original_Shape[1]):

            ### Iterate over valid pixels in reference area
            for X_ in range(2*Reference_Range + 1):
                Pos_X = X + X_ - Reference_Range
                if Pos_X < 0 or Pos_X >= Original_Shape[0]:
                    continue
                for Y_ in range(2*Reference_Range + 1):
                    Pos_Y = Y + Y_ - Reference_Range
                    if Pos_Y < 0 or Pos_Y >= Original_Shape[1]:
                        continue

                    #### Set pixel data
                    for Channel in range(Original_Shape[2]):
                        Input[Y*Original_Shape[0] + X, X_, Y_, Channel] = Original_Picture[Pos_X, Pos_Y, Channel] / 255
                    Input[Y*Original_Shape[0] + X, X_, Y_, Original_Shape[2]] = 1

    Prediction = Model.predict(Input)

    for X in range(Original_Shape[0]):
        for Y in range(Original_Shape[1]):

            ### Iterate over pixels in original (upscale relative to small) which correspond to position
            for X_ in range(Scaling_Factor):
                Pos_X = Scaling_Factor*X + X_
                for Y_ in range(Scaling_Factor):
                    Pos_Y = Scaling_Factor*Y + Y_

                    #### Set pixel data
                    for Channel in range(Original_Shape[2]):
                        Large_Picture[Pos_X, Pos_Y, Channel] = min(255, max(0, int(Prediction[Y*Original_Shape[0] + X, X_, Y_, Channel] * 255 + 0.5)))

    return Large_Picture

















if __name__ == "__main__":
    Main()
