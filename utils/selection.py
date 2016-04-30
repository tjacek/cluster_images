import files
import shutil

def select_data(in_path,out_path,prefix="bin"):
    files.make_dir(out_path)
    all_files=files.get_files(in_path)
    prefix_files=[f for f in all_files 
	                if files.extract_prefix(f)==prefix]
    in_prefix_files=files.append_path(in_path,prefix_files)
    out_prefix_files=files.append_path(out_path,prefix_files)
    for in_file,out_file in zip(in_prefix_files,out_prefix_files):
        shutil.copyfile(in_file, out_file)
        print(in_file)
    
if __name__ == "__main__":
    path="../dataset/"
    in_path=path+"MSRDailyAct/"
    out_path=path+"binary"
    select_data(in_path,out_path)