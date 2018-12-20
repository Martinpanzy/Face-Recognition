from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def compute_eigenspace(X_data, mode):
    # Using high/low dimensional computation, return the average vector and the eigenspace of the given data.
    N, D = X_data.shape
    X_avg = X_data.mean(0)
    X_avgm = np.array([X_avg]*N)
    A = (X_data - X_avgm).T
    if mode == "high":
        S = A.dot(A.T) / N
        e_vals, e_vecs = np.linalg.eig(S)
    elif mode == "low":
        S = (A.T).dot(A) / N
        e_vals, e_vecs = np.linalg.eig(S)
        e_vecs = np.dot(A,e_vecs)
        e_vecs = e_vecs / np.linalg.norm(e_vecs, axis=0)
    return X_avg, A, e_vals, e_vecs

def plot_image(face_vector, w, h, filename):
    # Reshape the given image data, plot the image
    plt.figure()
    image = np.reshape(np.absolute(face_vector),(w,h)).T
    fig = plt.imshow(image, cmap = 'gist_gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
    plt.close()
    return

def plot_graph(type, eig_value, i, x, y, xtick, ytick, filename):
    # Plot the first i eigenvalues
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if type == "bar":
        plt.bar(list(range(0, i)), eig_value[:i])
    else:plt.plot(list(range(0, i)), eig_value[:i])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick))
    plt.xlabel(x, fontsize=10)
    plt.ylabel(y, fontsize=10)
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
    #plt.show(block=False)
    return