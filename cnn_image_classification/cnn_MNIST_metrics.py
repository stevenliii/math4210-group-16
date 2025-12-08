import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, test_images, test_labels):
    """
    Plot a confusion matrix for the model predictions
    """
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1) if test_labels.ndim > 1 else test_labels
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize = (10, 8))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', 
                xticklabels = range(10), yticklabels = range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_sample_predictions(model, test_images, test_labels, num_samples = 12):
    """
    Plot sample predictions with true vs predicted labels
    """
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis = 1)
    true_labels = np.argmax(test_labels, axis = 1) if test_labels.ndim > 1 else test_labels
    
    indices = np.random.choice(len(test_images), num_samples, replace = False)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    test_accuracy = 0
    
    for i, idx in enumerate(indices):
        axes[i].imshow(test_images[idx].squeeze(), cmap = 'gray')
        
        true_label = true_labels[idx]
        pred_label = predicted_labels[idx]
        confidence = np.max(predictions[idx])
        
        if true_label == pred_label:
            test_accuracy += 1
            color = 'green'
            title = f'True: {true_label}, Pred: {pred_label} \nConf: {confidence:.3f}'
        else:
            color = 'red'
            title = f'True: {true_label}, Pred: {pred_label} \nConf: {confidence:.3f}'
        
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f'Sample Predictions (Sample Accuracy: {test_accuracy / num_samples:.4f})', fontsize = 14)
    plt.tight_layout()
    plt.show()
    
    return test_accuracy / num_samples