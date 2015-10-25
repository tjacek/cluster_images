import utils,instances,read_frames,reduce_dim
import instances as inst

def cluster_images(path):
    images=read_frames.read_images(path)
    data=inst.get_data(images)
    reduced_data=reduce_dim.spectral_reduction(data)
    inst.set_reduced(images,reduced_data)
    inst.save_reduce("raw.csv",images)

if __name__ == "__main__":
    path="test_images/"
    cluster_images(path)
