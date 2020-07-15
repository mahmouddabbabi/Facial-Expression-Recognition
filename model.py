from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf


config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.15
session=tf.compat.v1.Session(config=config)


class FacialExpressionModel(object):
	list=["angry","disgust","fear","happy","neutral","sad","surprise"]

	def __init__(self,json_model,weights):
		with open(json_model,"r") as json_file:
			model=json_file.read()
			self.loaded_model=model_from_json(model)

		self.loaded_model.load_weights(weights)
		self.loaded_model._make_predict_function()
	def predict_emotion(self,img):
		self.preds = self.loaded_model.predict(img)
		return FacialExpressionModel.list[np.argmax(self.preds)]
