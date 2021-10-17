import click
import os
from pathlib import Path


@click.group()
def main():
    pass


@click.command()
@click.option('-p', '--model_path', help="Path to trained model", required=True)
@click.option('-P', '--preprocess_path', required=True,
              help="Path to folder with saved preprocessor parameters")
@click.option('--debug', default=False, hidden=True)
def runserver(model_path, preprocess_path, debug):
    from src.runserver import start_server
    preprocess_path = os.path.join(preprocess_path, "preprocessor.pickle")

    start_server(model_path, preprocess_path, debug)


@click.command()
@click.option('-p', '--model_path', help="Path where folder should be saved after training", required=True)
@click.option('--preprocessor_root', default=None,
              help="Folder where preprocessor saved dataset and its parameters. "
                   "[default: ./Data/stored]")
@click.option('--n_neurons', default=200, help="Amount of neurons in each layer", show_default=True)
@click.option('--dropout', default=0.1, help="Dropout value for LSTM layers", show_default=True)
@click.option('--recurrent_dropout', default=0.1, help="Recurrent dropout for LSTM layers", show_default=True)
def train(model_path, preprocessor_root, n_neurons, dropout, recurrent_dropout):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # Disabling TF info messages

    if preprocessor_root is None:
        preprocessor_root = os.path.join("Data", "stored")

    from src.trainer import start_training
    start_training(model_path, preprocessor_root, n_neurons, dropout, recurrent_dropout)


@click.command()
@click.option('-f', '--data_folder', default=None,
              help="Root of folders with training data, each folder represents single genre. "
                   "[default: ./Data/genres_original]")
@click.option('-s', '--save_folder', default=None,
              help="Root folder where processed dataset and preprocessor "
                   "parameters are stored. [default: ./Data/stored]")
@click.option('-F', '--force', is_flag=True,
              help='Force preprocessing the data even if dataset was found in save_folder')
@click.option('-sr', '--sample_rate', default=22050, help='Sample rate of the loaded song', show_default=True)
@click.option('--frame_size', default=100, show_default=True,
              help='Number of time series elements that will be in a single frame')
@click.option('--frame_shift', default=20, show_default=True,
              help='Amount of shift between one input frame and another')
@click.option('--n_mfcc', default=50, help='Number of Mel coefficient to extract from raw song', show_default=True)
@click.option('--n_chroma', default=50, help='Number of Chroma frequencies to extract from raw song', show_default=True)
@click.option('--n_fft', default=2048, help='Number of samples in a single Fourier Transform frame', show_default=True)
@click.option('--hop_length', default=512, help='Shift of Fourier Transform frame', show_default=True)
@click.option('--roll_perc', default=0.85, help='Percentage for Spectral Rolloff', show_default=True)
@click.option('--batch_size', default=32, help="Number of samples in a single batch", show_default=True)
@click.option('--split', default=0.2, show_default=True,
              help='Percentage of entire dataset to go into test and validation sets')
def preprocess(data_folder, save_folder, force, sample_rate, frame_size, frame_shift,
               n_mfcc, n_chroma, n_fft, hop_length, roll_perc, batch_size, split):
    if data_folder is None:
        data_folder = os.path.join("Data", "genres_original")

    if save_folder is None:
        save_folder = os.path.join("Data", "stored")
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    from src.preprocess import start_preprocessing
    start_preprocessing(data_folder, save_folder, force, sample_rate, frame_size, frame_shift,
                        n_mfcc, n_chroma, n_fft, hop_length, roll_perc, batch_size, split)


main.add_command(preprocess)
main.add_command(train)
main.add_command(runserver)

if __name__ == "__main__":
    main()
