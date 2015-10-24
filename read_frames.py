import utils

def read_images(path):
    action_files=utils.get_dirs(path)
    action_files=utils.append_path(path,action_files)
    images=[]
    for action_path in action_files:
        images+=read_action(action_path)
    return images

def read_action(action_path):
    all_files=utils.get_files(action_path)
    all_files=utils.append_path(action_path+"/",all_files)
    frames=utils.read_images(all_files)
    frames=[ frame.flatten() for frame in frames]
    return frames

if __name__ == "__main__":
    path="images/"
    print(len(read_images(path)))

