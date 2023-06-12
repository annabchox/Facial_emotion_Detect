def main():
    pass

# -------------------
# IMPORTS - NEED TO CONDENSE
# -------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf 
import sys
import os
import seaborn as sns

# Set the Seaborn style to "darkgrid"
sns.set(style="darkgrid")

from PIL import Image
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Rescaling
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import  AUC, Precision, Recall
from functools import partial
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




def model_scores_to_csv(models, history_list, model_name):
    '''
    Append the last epoch scores of a model to a CSV file.

    Parameters:
        models : list
            List of model names.

        history_list : list
            List of history objects containing training history.

        model_name : str
            Name of the model to be appended.

    Returns:
        None

    Example:
        model_scores_to_csv(['model1', 'model2'], [h1, h2], 'model_1')
    '''
        
    #Specified Columns:
    columns = ['train_loss','train_acc','train_precision','train_recall','train_auc','val_loss','val_acc','val_precision','val_recall','val_auc']
    
    if 'model_eval.csv' not in os.listdir('./Results'):
        df = pd.DataFrame(columns=columns, index=['models'])
    
        df.to_csv('./Results/model_eval.csv')
    
    # Get the last epoch values from the history dictionary
    last_epoch_values = [list(values)[-1] for values in history_list.history.values()]

    # Create a dictionary with the model scores and the model name as the index
    model_scores = {col: [val] for col, val in zip(columns, last_epoch_values)}

    # Set the model name as the index in the dictionary
    model_scores['models'] = model_name

    # Create a temporary dataframe with the model scores
    df = pd.DataFrame(model_scores)
    
    df.set_index('models', inplace = True, drop = True)

    # Append the model scores to the CSV file
    df.to_csv('./Results/model_eval.csv', mode='a', header = False)

    return



def print_class_counts(dataset):
    """
    Prints the count and percentage of each class in a given dataset.

    Parameters:
        dataset (tf.data.Dataset): The dataset containing images and labels.

    Returns:
        None
    """
    # Get the list of class names
    class_names = dataset.class_names

    # Initialize a dictionary to store the counts
    class_counts = {class_name: 0 for class_name in class_names}

    # Iterate over the dataset and count the occurrences of each class label
    total_samples = 0
    for _, labels in dataset:
        total_samples += len(labels)
        for label in labels:
            class_counts[class_names[label.numpy().argmax()]] += 1

    # Print the index, class name, count, and percentage for each class
    for i, class_name in enumerate(class_names):
        count = class_counts[class_name]
        percentage = (count / total_samples) * 100
        print(f"Index {i}: {class_name} - Count: {count}, Percentage: {percentage:.2f}%")




def plot_class_distribution(dataset, title):
    """
    Plots the class distribution of a given dataset.

    Parameters:
        dataset (tf.data.Dataset): The dataset containing images and labels.
        title (str): The title of the plot.

    """
    # Extract class names from the dataset
    class_names = dataset.class_names

    # Initialize a dictionary to store the counts
    class_counts = {class_name: 0 for class_name in class_names}

    # Iterate over the dataset and count the occurrences of each class label
    for images, labels in dataset:
        for label in labels:
            class_counts[class_names[label.numpy().argmax()]] += 1

    # Extract class labels and counts
    class_labels = list(class_counts.keys())
    class_values = list(class_counts.values())

    # Define pastel colors
    # Source for color hex: https://www.color-hex.com/color-palette/4628
    colors = ['#66545e', '#a39193', '#aa6f73', '#eea990', '#f6e0b5']

    # Create a bar graph
    plt.bar(class_labels, class_values, color=colors)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45)

    plt.show();


    
def preprocess_and_filter_images(source_dir, destination_dir, num_images_to_keep):
    """
    Preprocesses and filters images in the source directory and saves them to the destination directory.

    Args:
        source_dir (str): Path to the source directory containing the original images.
        destination_dir (str): Path to the destination directory to save the preprocessed and filtered images.
        num_images_to_keep (dict): Dictionary specifying the number of images to keep for specific folders.

    """
    # Create the destination directory if it doesn't exist
    # Source: https://docs.python.org/3/library/os.html
    os.makedirs(destination_dir, exist_ok=True)

    # Iterate over the files in the source directory
    # Code moderated from: https://docs.python.org/3/library/os.html
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            # Check if the file has an image extension
            image_extensions = ['.jpg', '.jpeg', '.png']
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Extract the class label from the filename
                class_label = os.path.basename(root)

                # Create the class directory in the destination directory
                class_dir = os.path.join(destination_dir, class_label)
                os.makedirs(class_dir, exist_ok=True)

                # Check if the class directory needs random image dropping
                if class_label in num_images_to_keep:
                    # Check if the number of images in the class directory exceeds the desired count
                    num_images_in_class = len(os.listdir(class_dir))
                    if num_images_in_class >= num_images_to_keep[class_label]:
                        continue

                # Load the image
                #Source: https://pillow.readthedocs.io/en/stable/reference/Image.html
                image_path = os.path.join(root, filename)
                image = Image.open(image_path)

                # Resize the image to 48x48
                resized_image = image.resize((48, 48), Image.BILINEAR)

                # Convert the image to grayscale
                grayscale_image = resized_image.convert('L')

                # Save the preprocessed image to the class directory
                destination_path = os.path.join(class_dir, filename)
                grayscale_image.save(destination_path)    
    

    
def plot_hist(history, title):
    """
    Code Source Inspo: https://www.kaggle.com/code/ahmadjaved097/multiclass-image-classification-using-cnn
    Plots the training and validation loss/accuracy curves from a given history object.

    Parameters:
        history (tensorflow.keras.callbacks.History): History object obtained from model training.
        title (str): Title of the plot.

    Returns:
        None

    Example:
        plot_history(h1, 'CNN Model 1')
    """
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

    ax[0].set_title(f'{title} Accuracy')
    ax[0].plot(history.history['acc'], 'o-', label='Train', c='#6495ED')
    ax[0].plot(history.history['val_acc'], 'o-', label='Validation', c='#BA55D3')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='best')

    ax[1].set_title(f'{title} Loss')
    ax[1].plot(history.history['loss'], 'o-', label='Train', c='#6495ED')
    ax[1].plot(history.history['val_loss'], 'o-', label='Validation', c='#BA55D3')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend(loc='best')

    ax[2].set_title(f'{title} Learning Rate')
    ax[2].plot(history.history['lr'], 'o-', label='Learning Rate', c='#FF7F50')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Learning Rate')
    ax[2].legend(loc='best')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots
    plt.savefig(f'../../Images/{title} plot.jpg')
    plt.show();


    

def plot_confusion_matrix(dataset, model, title):
    """
    Plots the confusion matrix based on true labels and predicted labels.

    Args:
        dataset (tf.data.BatchDataset): The dataset containing images and labels.
        model_predictions (numpy.ndarray): The model predictions.
        title (str): The title of the plot.

    Returns:
        None
    """
    sns.set(style="dark")
    # Separate Image and Label Arrays
    dataset_as_array = list(dataset.as_numpy_iterator())
    label_batches = [dataset_as_array[i][1] for i in range(len(dataset_as_array))]
    image_batches = [dataset_as_array[i][0] for i in range(len(dataset_as_array))]

    # Unpack Image and Label Batches into Single Array
    unpacked_label_batches = np.vstack(label_batches)
    unpacked_image_batches = np.vstack(image_batches)

    # Get true labels
    true_labels = np.argmax(unpacked_label_batches, axis=1)

    # Get predicted labels
    pred_probs = model.predict(unpacked_image_batches)
    predicted_labels = np.argmax(pred_probs, axis=1)

    # Create confusion matrix display
    class_names = dataset.class_names
    cm = ConfusionMatrixDisplay.from_predictions(true_labels, predicted_labels, display_labels=class_names, cmap='PuBuGn')

    # Plot the confusion matrix
    cm.figure_.suptitle(title)
    plt.xticks(rotation=90)
    plt.savefig(f'../../Images/{title}_cm.jpg')
    plt.show();
    

    
    
def plot_correct_dist(dataset, model, title):
    """
    Plots the distribution of correct and incorrect predictions for each class based on true labels and predicted labels.

    Args:
        dataset (tf.data.BatchDataset): The dataset containing images and labels.
        model (tf.keras.Model): The trained model used for predictions.
        title (str): The title of the plot.

    Returns:
        None
    """
    sns.set(style="dark")
    # Create label dict
    label_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'}
    # Separate Image and Label Arrays
    dataset_as_array = list(dataset.as_numpy_iterator())
    label_batches = [dataset_as_array[i][1] for i in range(len(dataset_as_array))]
    image_batches = [dataset_as_array[i][0] for i in range(len(dataset_as_array))]

    # Unpack Image and Label Batches into Single Array
    unpacked_label_batches = np.vstack(label_batches)
    unpacked_image_batches = np.vstack(image_batches)

    # Get true labels
    true_labels = np.argmax(unpacked_label_batches, axis=1)

    # Get predicted labels
    pred_probs = model.predict(unpacked_image_batches)
    predicted_labels = np.argmax(pred_probs, axis=1)

    # Create confusion matrix display
    # Initialize counters for each class
    class_counts = {class_name: {"correct": 0, "incorrect": 0} for class_name in label_dict.values()}

    # Iterate through true labels and predicted labels
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        true_label_name = label_dict[true_label]
        predicted_label_name = label_dict[predicted_label]

        if true_label_name == predicted_label_name:
            class_counts[true_label_name]["correct"] += 1
        else:
            class_counts[true_label_name]["incorrect"] += 1
    
    # Prepare data for bar graph
    classes = list(class_counts.keys())
    correct_counts = [counts["correct"] for counts in class_counts.values()]
    incorrect_counts = [counts["incorrect"] for counts in class_counts.values()]

    # Set up the figure
    plt.figure(figsize=(10, 7))

    # Plot the bars
    sns.barplot(x=classes, y=correct_counts, color='#A7D5E4', label='Correct')
    sns.barplot(x=classes, y=incorrect_counts, color='#ECAE7B', label='Incorrect', bottom=correct_counts)

    # Add labels inside the bars
    for i in range(len(classes)):
        plt.text(i, correct_counts[i], f"{correct_counts[i]}", ha='center', va='bottom', color='blue')
        plt.text(i, correct_counts[i] + incorrect_counts[i], f"{incorrect_counts[i]}", ha='center', va='bottom', color='orange')

    # Add labels, title, and legend
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'Correct and Incorrect Predictions of {title}')
    plt.legend(loc="lower left")

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f'../../Images/{title}_prediction_dist.jpg')
    plt.show();
    
    
    
    
if __name__ == "__main__":
    main()