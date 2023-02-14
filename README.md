# Potato Disease Detector
A potato disease detector built from scratch using deep neural networks and deep learning. The project consists of two main parts: the model and the React frontend.<br/>
The deployed model can be accessed and tested via: https://potato-disease-detector-ynyn4uvs7a-uc.a.run.app

## Model
The model was trained on a dataset of potato plant leaves to identify different types of potato diseases. The model was then converted to an API and deployed to Google Cloud Platform (GCP).

## React Frontend
The React frontend was created to make the predictions made by the API easily accessible to users. Users have the option to upload an image of a potato plant leaf or take a picture using their device's camera. The app then sends the image to the API, which returns a prediction of the type of potato disease present.

The React frontend was also deployed to GCP and is available to the public. The application was dockerized and necessary CI/CD controls were put in place to ensure smooth and efficient deployment.

## Usage
To use the Potato Disease Detector, simply visit the deployed React frontend and either upload an image or take a picture. The app will then display the prediction made by the API.

## Conclusion
The Potato Disease Detector is a powerful tool for identifying potato diseases, helping farmers and researchers make informed decisions about their crops. The deployment of both the API and the React frontend to GCP makes it accessible to anyone with an internet connection, providing a valuable resource for the agriculture industry.
