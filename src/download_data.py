import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def download_data():
    label_names = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
                   'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)',
                   'Speed limit (100km/h)',
                   'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons',
                   'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
                   'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution',
                   'Dangerous curve to the left',
                   'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
                   'Road narrows on the right',
                   'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
                   'Beware of ice/snow',
                   'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
                   'Turn left ahead',
                   'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
                   'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons']

    with open(f'../pickles/data0.pickle', 'rb') as f:
        img_list = pickle.load(f)

    output = []

    for idx, (img, label) in enumerate(zip(img_list['x_train'], img_list['y_train'])):
        plt.imsave(f'images/{idx}.png', np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0)))
        output.append({'file_name': f'{idx}.png', 'label': label, 'label_name': label_names[label]})

    df = pd.DataFrame(output)
    df.to_csv('label_lookup.csv', index=False)


if __name__ == '__main__':
    download_data()
