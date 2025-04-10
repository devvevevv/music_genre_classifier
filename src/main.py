import os
from predict import predict

def main():
    print('---Welcome to the music genre classifier---')

    file_path = input('Enter the path to the audio file: ')
    model_path = 'model.pkl'

    if (not os.path.exists(file_path)) or (not(os.path.exists(model_path))):
        print('Error opening file.')
        return 1

    print('---Analyzing audio file---')
    genre = predict(model_path, file_path)

if __name__ == '__main__':
    main()