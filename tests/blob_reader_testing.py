from pathlib import Path
import shutil

from syndiffix.blob import SyndiffixBlobReader

tests_dir = Path.cwd().joinpath('tests')
blob_test_path = tests_dir.joinpath('.sdx_blob_test_blob')

blob_test_path = tests_dir.joinpath('.sdx_blob_test_blob')
if blob_test_path.exists() and blob_test_path.is_dir():
    shutil.rmtree(blob_test_path)
sblob = SyndiffixBlobReader(blob_name='test_blob', path_to_dir=tests_dir)