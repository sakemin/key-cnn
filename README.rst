.. image:: https://img.shields.io/badge/License-AGPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0

=======
Key-CNN combined with rule-based approach
=======

This is a hybrid framework for musical key estimation that combines CNN-based and rule-based approaches.

Features
========

Multiple model types:

- CNN: Uses deep learning to predict musical key
- Rule-based: Uses music theory rules and signal processing
- Ensemble: Combines both approaches with weighted averaging

Basic Usage
==========

To analyze the key of an audio file:

.. code-block:: console

    python predict.py audio_file.wav

By default, this uses the ensemble model which combines both CNN and rule-based predictions.

Options
=======

You can specify different model types and parameters:

.. code-block:: console

    python predict.py audio_file.wav --model cnn     # Use only CNN model
    python predict.py audio_file.wav --model cnn --cnn_model deepsquare  # Use DeepSquare model
    python predict.py audio_file.wav --model rule    # Use only rule-based model
    python predict.py audio_file.wav --model ensemble --alpha 0.5  # Weight ensemble more toward averaging
    python predict.py audio_file.wav --model ensemble --alpha 0.5 --json_out  # Save predictions to JSON file

Parameters:

- ``--model``: Choose between 'cnn', 'rule', or 'ensemble' (default: ensemble)
- ``--cnn_model``: Choose between 'deepspec', 'deepsquare', 'shallowspec', or 'winterreise' etc. (default: deepspec)
- ``--alpha``: Weight for ensemble combination (0 = multiply only, 1 = average only, default: 0.5)
- ``--top_k``: Number of top predictions to show (default: 5)
- ``--json_out``: Save predictions to JSON file
- ``--output_path``: Path to save the JSON file (default: output/key_prediction.json)

Output
======

The program outputs:

1. The predicted key with confidence score
2. Top K most confident key predictions with their probabilities

Installation
============

Clone this repo and install dependencies:

.. code-block:: console

    git clone https://github.com/sakemin/key-cnn.git
    cd key-cnn
    pip install -r requirements.txt

Required packages (Python 3.7 compatible):

- numpy==1.16.0
- scipy>=1.0.1
- tensorflow==1.15.2
- librosa>=0.6.2
- jams>=0.3.1
- matplotlib>=3.0.0
- h5py>=2.7.0
- numba==0.48
- h5py==2.10.0

Docker
======

Build the Docker image:

.. code-block:: console

    docker build -t key-cnn .

Run with the default configuration:

.. code-block:: console

    docker run key-cnn

Analyze a custom audio file (replace /path/to/your/audiofile.wav with the actual path):

.. code-block:: console

    docker run -v /path/to/your/audiofile.wav:/app/input.wav key-cnn input.wav --json_out

You can also pass additional parameters:

.. code-block:: console

    docker run -v /path/to/your/audiofile.wav:/app/input.wav key-cnn \
    -v /path/to/output/directory:/app/output \
    input.wav \
    --model ensemble \
    --alpha 0.5 \
    --json_out \
    --output_path output/key_prediction.json

License
=======

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE v3.
For details, please see the `LICENSE <LICENSE>`_ file.

Acknowledgments
==============

This project combines approaches from:
- Original Key-CNN project (https://github.com/hendriks73/key-cnn)
- Musical Key Finder by Jack McArthur (https://github.com/jackmcarthur/musical-key-finder)
