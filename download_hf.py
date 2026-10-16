import glob

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="keithito/lj_speech",
    repo_type="dataset",
    revision="refs/convert/parquet",
    allow_patterns=["main/train/*.parquet"],
    local_dir="data/hf_ljspeech",
)

files = sorted(glob.glob("data/hf_ljspeech/main/train/*.parquet"))
print(len(files), "parquet files")
pf = pq.ParquetFile(files[0])
print(pf.schema_arrow)
print("rows in first file:", pf.metadata.num_rows)
