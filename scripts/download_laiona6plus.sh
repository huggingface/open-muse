conda create -n img2dataset python=3.9
conda activate img2dataset
pip install img2dataset
pip install fsspec[s3]

# login in to a cpu node
# srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --partition=production-cluster --pty bash

# Download metadata files
wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus/resolve/main/data/train-00000-of-00007-29aec9150af50f9f.parquet -O - | aws s3 cp - s3://muse-datasets/laion-aesthetic6plus-metadata/train-00000-of-00007-29aec9150af50f9f.parquet
wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus/resolve/main/data/train-00001-of-00007-060633a36bcf0956.parquet -O - | aws s3 cp - s3://muse-datasets/laion-aesthetic6plus-metadata/train-00001-of-00007-060633a36bcf0956.parquet
wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus/resolve/main/data/train-00002-of-00007-709151a2715d894d.parquet -O - | aws s3 cp - s3://muse-datasets/laion-aesthetic6plus-metadata/train-00002-of-00007-709151a2715d894d.parquet
wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus/resolve/main/data/train-00003-of-00007-2dc95366d4278bb8.parquet -O - | aws s3 cp - s3://muse-datasets/laion-aesthetic6plus-metadata/train-00003-of-00007-2dc95366d4278bb8.parquet
wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus/resolve/main/data/train-00004-of-00007-f06fcc8b41bf5fdf.parquet -O - | aws s3 cp - s3://muse-datasets/laion-aesthetic6plus-metadata/train-00004-of-00007-f06fcc8b41bf5fdf.parquet
wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus/resolve/main/data/train-00005-of-00007-f1ec12a5b5b3e6c0.parquet -O - | aws s3 cp - s3://muse-datasets/laion-aesthetic6plus-metadata/train-00005-of-00007-f1ec12a5b5b3e6c0.parquet
wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus/resolve/main/data/train-00006-of-00007-57e09e020b9c2df4.parquet -O - | aws s3 cp - s3://muse-datasets/laion-aesthetic6plus-metadata/train-00006-of-00007-57e09e020b9c2df4.parquet


# Download images
img2dataset \
  --url_list s3://muse-datasets/laion-aesthetic6plus-metadata \
  --input_format "parquet" \
  --url_col "URL" \
  --caption_col "TEXT" \
  --output_format webdataset \
  --output_folder  /scratch/muse-data/laion-aesthetic6plus-data \
  --processes_count 24 \
  --thread_count 128 \
  --image_size 384 \
  --resize_only_if_bigger=True \
  --resize_mode="keep_ratio" \
  --skip_reencode=True \
  --timeout 30 \
  --retries 2 \
  --save_additional_columns '["hash","similarity","punsafe","pwatermark","AESTHETIC_SCORE"]' \
  --enable_wandb \


# Copy images to s3
aws s3 cp /scratch/muse-data/laion-aesthetic6plus-data s3://muse-datasets/laion-aesthetic6plus-data --recursive

# Total images: 12096828
# Successes: 8974320