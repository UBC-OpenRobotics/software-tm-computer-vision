import os
import json
import urllib.request


output_directory = "./ycb"
os.makedirs(output_directory, exist_ok=True)

# You can either set this to "all" or a list of the objects that you'd like to
# download.
#objects_to_download = "all"
#objects_to_download = ["002_master_chef_can", "003_cracker_box"]
objects_to_download = ["026_sponge"]

# You can edit this list to only download certain kinds of files.
# 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
# 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
# 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
# 'google_16k' contains google meshes with 16k vertices.
# 'google_64k' contains google meshes with 64k vertices.
# 'google_512k' contains google meshes with 512k vertices.
# See the website for more details.
#files_to_download = ["berkeley_rgbd", "berkeley_rgb_highres", "berkeley_processed", "google_16k", "google_64k", "google_512k"]
files_to_download = ["berkeley_processed", "berkeley_rgbd"]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = base_url + "objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def fetch_objects(url):
    response =response = urllib.request.urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]

def download_file(url, filename):
    try:
        with urllib.request.urlopen(url) as u:
            file_size = int(u.headers.get("Content-Length", 0))
            print(f"Downloading: {filename} ({file_size / 1_000_000:.2f} MB)")

            with open(filename, 'wb') as f:
                file_size_dl = 0
                block_sz = 65536
                while True:
                    buffer = u.read(block_sz)
                    if not buffer:
                        break

                    file_size_dl += len(buffer)
                    f.write(buffer)

                    progress = file_size_dl * 100.0 / file_size if file_size else 0
                    print(f"\r{file_size_dl / 1_000_000:.2f} MB [{progress:.2f}%]", end="", flush=True)
            print("\nDownload complete.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


def tgz_url(object, type):
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object,type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object,type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object,type=type)

def extract_tgz(filename, dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename,dir=dir)
    os.system(tar_command)
    os.remove(filename)

def check_url(url):
    try:
        request = urllib.request.Request(url, method="HEAD")
        response = urllib.request.urlopen(request)
        return True
    except Exception as e:
        return False


if __name__ == "__main__":
    objects = objects_to_download #fetch_objects(objects_url)

    for object_name in objects:
        if objects_to_download == "all" or object_name in objects_to_download:
            for file_type in files_to_download:
                url = tgz_url(object_name, file_type)
                if not check_url(url):
                    print(f"Skipping {url}: File not found")
                    continue
                filename = os.path.join(output_directory, f"{object_name}_{file_type}.tgz")
                download_file(url, filename)
                if extract:
                    extract_tgz(filename, output_directory)