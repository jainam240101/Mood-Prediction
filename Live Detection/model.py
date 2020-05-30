from keras.models import model_from_json
import numpy as np

class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry","Happy","Sad","Surprise","Neutral"]
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        expression= FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
        if(expression=="Neutral"):
            print("Counting Stars by OneRepublic")
        elif(expression=="Angry"):
            print("Lose Yourself by Eminem")
        elif(expression=="Happy"):
            print("The Nights by Avicci")
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]