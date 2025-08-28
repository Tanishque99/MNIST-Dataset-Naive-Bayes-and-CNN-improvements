import scipy.io
import numpy as np

def load_mat_file(file_path):
    # Load .mat file and return its content.
    file_data= scipy.io.loadmat(file_path)
    data = file_data[list(file_data.keys())[-1]]  # Get the last key's value
    return data
def extract_data(image):
    # Extract features from the image.
    n= image.shape[2] # Number of images
    features = np.zeros((n, 2))  # Initialize features array with shape (n, 2)
    for i in range(n):
        img = image[:, :, i]
        img=img/255.0  # Normalize the image
        img = img.flatten()
        features[i,0] = np.mean(img)
        features[i,1] = np.std(img)
    return features
    
def extract_parameters(features):
    # Extract parameters from the features.
    mu=np.mean(features, axis=0)  # Mean of features
    sigma = np.std(features, axis=0)  # Standard deviation of features
    return mu, sigma

def calculate_prob(features, mu_0, sigma_0, mu_1, sigma_1,prior_0, prior_1):
    # Calculate the likelihood of features given the parameters.
    gauss0 = (1 / (sigma_0 * np.sqrt(2 * np.pi))) * \
             np.exp(-0.5 * ((features - mu_0) / sigma_0) ** 2)
    prob_0 = np.prod(gauss0, axis=1) * prior_0
    
    # Likewise for class 1
    gauss1 = (1 / (sigma_1 * np.sqrt(2 * np.pi))) * \
             np.exp(-0.5 * ((features - mu_1) / sigma_1) ** 2)
    prob_1 = np.prod(gauss1, axis=1) * prior_1
    return (prob_1 > prob_0).astype(int)  # Return 1 if prob_1 > prob_0, else 0

#### MAIN CODE ####

# Load training and test data
train_0= load_mat_file('train_0_img-2.mat')
train_1 = load_mat_file('train_1_img-2.mat')
test_0 = load_mat_file('test_0_img-2.mat')
test_1 = load_mat_file('test_1_img-2.mat')

# Extract features and parameters
features_train_0 = extract_data(train_0)
features_train_1 = extract_data(train_1)
features_test_0 = extract_data(test_0)
features_test_1 = extract_data(test_1)

# Extract parameters for each class
mu_0, sigma_0 = extract_parameters(features_train_0)
mu_1, sigma_1 = extract_parameters(features_train_1)

# calculate priors
total_train_samples = features_train_0.shape[0] + features_train_1.shape[0]
prior_0 = features_train_0.shape[0] / total_train_samples
prior_1 = features_train_1.shape[0] / total_train_samples

# Calculate probability to predict for test data
pred_0=calculate_prob(features_test_0, mu_0, sigma_0, mu_1, sigma_1, prior_0, prior_1)
pred_1=calculate_prob(features_test_1, mu_0, sigma_0, mu_1, sigma_1, prior_0, prior_1)


accuracy_0 = np.mean(pred_0 == 0)  # Accuracy for test_0
accuracy_1 = np.mean(pred_1 == 1)  # Accuracy for test_1

print(f"Mean and Standard Deviation for class digit '0' :{mu_0, sigma_0}")
print(f"Mean and Standard Deviation for class digit '1' :{mu_1, sigma_1}")


print(f"Accuracy for test_0: {accuracy_0 * 100:.2f}%")
print(f"Accuracy for test_1: {accuracy_1 * 100:.2f}%")
print(f"Overall Accuracy: {(accuracy_0 + accuracy_1) / 2 * 100:.2f}%")