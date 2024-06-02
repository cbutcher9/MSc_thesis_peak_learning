import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.express as px


def load_data(sample, path_load):
    data_original = np.load(path_load + "_matrix.npy")
    mz_vector = np.load(path_load + "_mz_vector.npy")
    row2grid = np.load(path_load + "_row2grid.npy")

    data = np.copy(data_original)
    data[data <= 0] = 0
    residual = np.copy(data_original)
    residual[residual > 0] = 0
    print(sample+" data shape = ", data.shape)
    
    return data, residual, mz_vector, row2grid


def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def make_ion_image(mz_value, data, mz_vector, row2grid, save, path_save):
    mz_vector = np.ndarray.flatten(mz_vector)
    decimals = str(mz_value)[::-1].find('.')
    if decimals==-1:
        decimals=0
    index = np.where(np.round(mz_vector, decimals)==mz_value)[0]
    if len(index) == 0:
        index = find_nearest_idx(mz_vector, mz_value)
    else:
        index = index[0]
    result_2D = make_image(row2grid, data[:,index])
    plt.imshow(result_2D)
    plt.colorbar()
    rounded_val3 = np.round(mz_vector[index],3)
    rounded_val0 = np.round(mz_vector[index],0)
    formatted_val0 = "{:.0f}".format(rounded_val0)
    plt.title("m/z = "+ str(rounded_val3))
    if save:
        plt.savefig(f"{path_save}mz_{formatted_val0}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def make_ion_image_px(mz_value, data, mz_vector, row2grid, save, path_save):
    mz_vector = np.ndarray.flatten(mz_vector)
    decimals = str(mz_value)[::-1].find('.')
    if decimals==-1:
        decimals=0
    index = np.where(np.round(mz_vector, decimals)==mz_value)[0]
    if len(index) == 0:
        index = find_nearest_idx(mz_vector, mz_value)
    else:
        index = index[0]
        
    result_2D = make_image(row2grid, data[:,index])
    rounded_val3 = np.round(mz_vector[index],3)
    rounded_val0 = np.round(mz_vector[index],0)
    formatted_val0 = "{:.0f}".format(rounded_val0)
    fig = px.imshow(result_2D, title = "m/z = "+ str(rounded_val3), color_continuous_scale='viridis')
    # px.colorbar()
    if save:
        plt.savefig(f"{path_save}mz_{formatted_val0}.png", dpi=300, bbox_inches='tight')
    fig.show()
        
        
        
def compute_median(v):
    non_zero_elements = v[np.nonzero(v)]
    if len(non_zero_elements) > 0:
        return np.median(non_zero_elements)
    else:
        return 0

def compute_mean(v):
    non_zero_elements = v[np.nonzero(v)]
    if len(non_zero_elements) > 0:
        return np.mean(non_zero_elements)
    else:
        return 0
    

def grid2row(x,y, row2grid):
    xmax = np.max(row2grid[:,0])
    xmin = np.min(row2grid[:,0])
    ymax = np.max(row2grid[:,1])
    ymin = np.min(row2grid[:,1])
    # print(xmax+1, ymax+1, xmin, ymin, xmax-xmin, ymax-ymin)
    grid2row = np.zeros((xmax+1, ymax+1), dtype=int) + np.nan
    for r, c in enumerate(row2grid):
        grid2row[c[0], c[1]] = r
    return int(grid2row[x+xmin,y+ymin])
    

def make_image(row2grid, spatial_i):
    xmax = np.max(row2grid[:,0])
    xmin = np.min(row2grid[:,0])
    ymax = np.max(row2grid[:,1])
    ymin = np.min(row2grid[:,1])

    image_matrix = np.zeros([xmax-xmin+1,ymax-ymin+1])
    k = 0
    for e in row2grid:
        image_matrix[e[0]-xmin,e[1]-ymin] = spatial_i[k]
        k+=1
    return image_matrix
