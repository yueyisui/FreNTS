import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.fftpack import dctn
import matplotlib.pyplot as plt


def conv_calculate(input_matrix, kernel):
    result = convolve2d(input_matrix, kernel, mode='same')
    return result

def DCT_calculate(input_matrix):
    dct_result = dctn(input_matrix, type=2, norm='ortho')
    dct_result = np.log(1+abs(dct_result))
    return dct_result

def generate_matrix_and_update(matrix_size, k):
    """
    Randomly generate a matrix a of size matrix_size x matrix_size.
    Find k random locations on matrix a and update their values randomly to get a new matrix b.

    Parameters:
    - matrix_size: size of the matrix, e.g. 4
    - k: number of positions to be updated, e.g. 3

    Returns:
    - matrix_a: generated matrix a
    - matrix_b: updated matrix b
    """
    # matrix_a = np.random.randint(1, matrix_size**2 + 1, size=(matrix_size, matrix_size))
    matrix_a = np.random.rand(matrix_size, matrix_size)

    # Random selection of k positions
    indices = np.random.choice(matrix_size**2, k, replace=False)

    # Backup matrix a for generating matrix b
    matrix_b = np.copy(matrix_a)

    # Randomly update values at selected locations
    for index in indices:
        row = index // matrix_size
        col = index % matrix_size
        # matrix_b[row, col] = np.random.randint(1, matrix_size**2 + 1)
        matrix_b[row, col] = np.random.rand()

    return matrix_a, matrix_b

def matrix_similarity(matrix1, matrix2):
    """
    Calculate the similarity (cosine similarity) of two 2D matrices
    Parameters.
    - matrix1: first 2D matrix
    - matrix2: second 2D matrix
    Returns.
    - similarity: similarity value in the range [-1, 1], 1 means completely similar, -1 means completely dissimilar.
    """
    # 将矩阵展平成向量
    vector1 = matrix1.flatten()
    vector2 = matrix2.flatten()

    # 计算余弦相似度
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (norm_vector1 * norm_vector2)

    return similarity

def calculate_similarity_statistics(matrix_size, w):
    """
    Calculate the mean and variance of w similarity results corresponding to different values of k and plot the graphs.
    Parameters.
    - matrix_size: size of the matrix, e.g. 4
    - w: number of repetitions, e.g. 100
    """
    k_values = np.arange(1, matrix_size**2 + 1)
    mean_values = []
    conv_mean_values = []
    dct_mean_values = []
    variance_values = []
    conv_variance_values = []
    dct_variance_values = []

    lower_std_dev_values = []
    upper_std_dev_values = []
    conv_lower_std_dev_values = []
    conv_upper_std_dev_values = []
    dct_lower_std_dev_values = []
    dct_upper_std_dev_values = []

    for k in k_values:
        conv_similarity_results = []
        dct_similarity_results = []
        similarity_results = []

        for _ in range(w):
            matrix_a, matrix_b = generate_matrix_and_update(matrix_size, k)
            kernel,kernel_b = generate_matrix_and_update(3, 0)

            matrix_a_conv = conv_calculate(matrix_a, kernel)
            matrix_b_conv = conv_calculate(matrix_b, kernel)
            matrix_a_dct  = DCT_calculate(matrix_a)
            matrix_b_dct  = DCT_calculate(matrix_b)

            similarity = matrix_similarity(matrix_a, matrix_b)
            similarity_conv = matrix_similarity(matrix_a_conv, matrix_b_conv)
            similarity_dct = matrix_similarity(matrix_a_dct, matrix_b_dct)
            similarity_results.append(similarity)
            conv_similarity_results.append(similarity_conv)
            dct_similarity_results.append(similarity_dct)

        # Calculate mean and variance
        mean_value = np.mean(similarity_results)
        mean_value_conv = np.mean(conv_similarity_results)
        mean_value_dct = np.mean(dct_similarity_results)
        variance_value = np.var(similarity_results)
        variance_value_conv = np.var(conv_similarity_results)
        variance_value_dct = np.var(dct_similarity_results)

        mean_values.append(mean_value)
        conv_mean_values.append(mean_value_conv)
        dct_mean_values.append(mean_value_dct)
        variance_values.append(variance_value)
        conv_variance_values.append(variance_value_conv)
        dct_variance_values.append(variance_value_dct)

        upper_std_dev_values.append(mean_value + np.sqrt(variance_value))
        lower_std_dev_values.append(mean_value - np.sqrt(variance_value))
        conv_upper_std_dev_values.append(mean_value_conv + np.sqrt(variance_value_conv))
        conv_lower_std_dev_values.append(mean_value_conv - np.sqrt(variance_value_conv))
        dct_upper_std_dev_values.append(mean_value_dct + np.sqrt(variance_value_dct))
        dct_lower_std_dev_values.append(mean_value_dct - np.sqrt(variance_value_dct))

    # draw
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 6))
    # plt.errorbar(k_values, mean_values, yerr=np.sqrt(variance_values), fmt='o-', label='Mean Similarity with Std Dev')
    # plt.errorbar(k_values, conv_mean_values, yerr=np.sqrt(conv_variance_values), fmt='o-', label='Mean of Convolution')
    # plt.errorbar(k_values, dct_mean_values, yerr=np.sqrt(dct_variance_values), fmt='o-', label='Mean of DCT')
    
    # plt.plot(k_values, mean_values, 'o-', label='Mean of Value')
    # plt.fill_between(k_values, upper_std_dev_values, lower_std_dev_values, alpha=0.2, label='Std of Value')
    plt.plot(k_values, conv_mean_values, 'o-', label='Mean of Convolution')
    plt.fill_between(k_values, conv_upper_std_dev_values, conv_lower_std_dev_values, alpha=0.2, label='Std of Convolution')
    plt.plot(k_values, dct_mean_values, 'o-', label='Mean of DCT')
    plt.fill_between(k_values, dct_upper_std_dev_values, dct_lower_std_dev_values, alpha=0.2, label='Std of DCT')


    plt.title('Similarity Statistics for Different k Values')
    plt.xlabel('k Values')
    plt.ylabel('Similarity')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('similarity_statistics.jpg', dpi=600,)
    plt.show()

if __name__ == '__main__':
    matrix_size = 4
    w = 1000

    calculate_similarity_statistics(matrix_size, w)
