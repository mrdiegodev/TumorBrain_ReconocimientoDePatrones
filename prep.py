import cv2
import numpy as np
import imutils
from tensorflow.keras.models import load_model

def process_and_predict(file_path, model_path, img_size=(224, 224), add_pixels=0):
    # Cargar la imagen
    def process_single_image(image_path, img_size):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unable to load: {image_path}")
        img_resized = cv2.resize(img, img_size)
        return img_resized
    
    # Recortar la imagen
    def process_single_image_crop(img, add_pixels):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if len(cnts) == 0:
            raise ValueError("No contours found in the image.")
        
        c = max(cnts, key=cv2.contourArea)
        
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        cropped_img = img[extTop[1] - add_pixels:extBot[1] + add_pixels, extLeft[0] - add_pixels:extRight[0] + add_pixels].copy()
        
        return cropped_img
    
    # Preprocesar la imagen
    def preprocess_img(img, img_size):
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        # img = img.astype('float32')
        return img
    
    # Cargar la imagen y redimensionarla
    image = process_single_image(file_path, img_size)
    
    # Recortar la imagen
    cropped_img = process_single_image_crop(img=image, add_pixels=add_pixels)
    
    # Preprocesar la imagen
    prep_img = preprocess_img(img=cropped_img, img_size=img_size)

    # Cargar el modelo y hacer predicciones
    model = load_model(model_path)
    predictions = model.predict(np.expand_dims(prep_img, axis=0))  # Añadir una dimensión para el batch
    
    # Retornar la predicción redondeada
    prediction = np.round(predictions[0][0])
    
    return prediction

# Uso de la función
file_path = './VAL/YES/Y98.jpg'
model_path = 'BrainTumor.keras'
prediction = process_and_predict(file_path, model_path, img_size=(224, 224), add_pixels=0)
print(prediction)
