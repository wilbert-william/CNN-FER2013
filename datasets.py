import os
import cv2
import numpy as np
import pandas as pd

def load_images_from_folders(folder_paths, image_size=(48, 48)):
    """Load images and labels from specified folders."""
    data = []
    labels = []
    for label, folder_path in folder_paths.items():
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(folder_path, filename)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = cv2.resize(image, image_size)
                data.append(image.flatten())
                labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Create a DataFrame to mimic the fer2013.csv format
    df = pd.DataFrame(data)
    df['emotion'] = labels
    
    return df

def get_class_to_arg(dataset_name='fer2013'):
    """Get the mapping from emotion labels to numerical values."""
    if dataset_name == 'fer2013':
        return {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4,
                'surprise': 5, 'neutral': 6}
    else:
        raise ValueError('Invalid dataset name')

def convert_labels_to_numeric(df):
    """Convert string labels to numeric labels."""
    label_map = get_class_to_arg('fer2013')
    df['emotion'] = df['emotion'].map(label_map)
    return df

def main():
    folder_paths = {
        'angry': 'C:/Skripsi/Enhancing-FER2013-Imbalance-main/Enhancing-FER2013-Imbalance-main/FER-2013/train/angry',
        'disgust': 'C:/Skripsi/Enhancing-FER2013-Imbalance-main/Enhancing-FER2013-Imbalance-main/FER-2013/train/disgust',
        'fear': 'C:/Skripsi/Enhancing-FER2013-Imbalance-main/Enhancing-FER2013-Imbalance-main/FER-2013/train/fear',
        'happy': 'C:/Skripsi/Enhancing-FER2013-Imbalance-main/Enhancing-FER2013-Imbalance-main/FER-2013/train/happy',
        'sad': 'C:/Skripsi/Enhancing-FER2013-Imbalance-main/Enhancing-FER2013-Imbalance-main/FER-2013/train/sad',
        'surprise': 'C:/Skripsi/Enhancing-FER2013-Imbalance-main/Enhancing-FER2013-Imbalance-main/FER-2013/train/surprise',
        'neutral': 'C:/Skripsi/Enhancing-FER2013-Imbalance-main/Enhancing-FER2013-Imbalance-main/FER-2013/train/neutral'
    }

    # Load images and labels from folders
    df = load_images_from_folders(folder_paths)

    # Convert string labels to numeric labels
    df = convert_labels_to_numeric(df)

    # Combine pixel data into a single column as a string of space-separated pixel values
    df['pixels'] = df.iloc[:, :-1].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    df = df[['emotion', 'pixels']]

    # Save the DataFrame to a CSV file to mimic the fer2013.csv format
    csv_path = 'C:/Skripsi/Enhancing-FER2013-Imbalance-main/Enhancing-FER2013-Imbalance-main/fer2013.csv'
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()