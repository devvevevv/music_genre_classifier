import os
import warnings

from chunks import analyze_audio_file
from config import resolve_model_path

warnings.filterwarnings("ignore")


def main():
    print("---Welcome to the music genre classifier---")

    file_path = input("Enter the path to the audio file: ")
    model_path = resolve_model_path()

    if (not os.path.exists(file_path)) or (not (os.path.exists(model_path))):
        print("Error opening file.")
        return 1

    print("---Analyzing audio file---")
    result = analyze_audio_file(file_path, model_path)

    if result is None:
        print("Error analyzing audio file.")
        return 1

    print(f"Predicted genre: {result}")


if __name__ == "__main__":
    main()
