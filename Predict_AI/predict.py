import numpy as np
import tensorflow as tf
import sys

# Load the pre-trained model
model = tf.keras.models.load_model("checkpoints/model.keras")

maybeResults = ["bad", "good"]

def main():
    nImages = 2
    
    wordResults = []
    
    for i in range(nImages):    
        img = "processed_images/test/" + str(i) + ".png"
        data = np.array([tf.keras.utils.load_img(img)])
           
        # Perform prediction using the loaded model
        wordResults.append(model.predict(data))
    
    a = ""    
    for i in range(nImages):
        a += f"{maybeResults[np.argmax(wordResults[i][0])]}\n"
    
    # Write the result to stdout
    sys.stdout.write(a)
    sys.stdout.flush()

if __name__ == "__main__":
    main()