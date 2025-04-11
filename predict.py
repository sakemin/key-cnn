import argparse
import librosa
import numpy as np
from keyfinder import Tonal_Fragment
from keycnn.classifier import KeyClassifier
from keycnn.feature import read_features

def predict_key(audio_path, model_type='ensemble', cnn_model='deepspec', alpha=0.5):
    """
    Predict musical key from audio file using specified model type.
    
    Args:
        audio_path (str): Path to audio file
        model_type (str): One of 'cnn', 'rule', or 'ensemble'
        alpha (float): Weight between 0 and 1 for ensemble combination
                      0 = only multiply, 1 = only average
    
    Returns:
        dict: Predicted key probabilities
        tuple: Most likely key and confidence
    """
    # Load and preprocess audio
    y, sr = librosa.load(audio_path)
    y_harmonic, _ = librosa.effects.hpss(y)
    
    predictions = {}
    
    if model_type in ['cnn', 'ensemble']:
        # Load CNN model
        classifier = KeyClassifier(cnn_model)

        # Prepare input for CNN
        features = read_features(audio_path)
        
        # Get CNN predictions
        out_cnn = classifier.estimate_key_with_confidence(features)
        
        if model_type == 'cnn':
            predictions = out_cnn
    
    if model_type in ['rule', 'ensemble']:
        # Get rule-based predictions
        fragment = Tonal_Fragment(y_harmonic, sr)
        out_rulebased = fragment.get_key_softmax()
        
        if model_type == 'rule':
            predictions = out_rulebased
            
    if model_type == 'ensemble':
        # Combine predictions with weighted combination
        predictions = {}
        for key in out_cnn.keys():
            mult = np.sqrt(out_cnn[key] * out_rulebased[key])
            avg = (out_cnn[key] + out_rulebased[key]) / 2
            predictions[key] = (1 - alpha) * mult + alpha * avg
            
        # Normalize
        total = sum(predictions.values())
        predictions = {k: v/total for k,v in predictions.items()}
        
    # Get most likely key
    predicted_key = max(predictions.items(), key=lambda x: x[1])
    
    return predictions, predicted_key

def main():
    parser = argparse.ArgumentParser(description='Predict musical key from audio')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--model', choices=['cnn', 'rule', 'ensemble'], 
                       default='ensemble', help='Model type to use')
    parser.add_argument('--cnn_model',
                       default='deepspec', help='CNN model to use; check `./keycnn/models/` for available models')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Show top K predictions')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight between 0 and 1 for ensemble combination (0 = only multiply, 1 = only average)')
    
    args = parser.parse_args()
    
    predictions, (key, confidence) = predict_key(args.audio_path, args.model, args.cnn_model, args.alpha)
    
    print(f"\nPredicted key: {key} (confidence: {confidence:.3f})")
    
    print(f"\nTop {args.top_k} most confident keys:")
    top_keys = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:args.top_k]
    for key, conf in top_keys:
        print(f"{key}: {conf:.3f}")

if __name__ == '__main__':
    main()
