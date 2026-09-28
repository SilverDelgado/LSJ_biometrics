import kagglehub

# Download latest version
path = kagglehub.dataset_download("debarghamitraroy/casia-webface", local_dir="data/dataset", unzip=True)

print("Path to dataset files:", path)