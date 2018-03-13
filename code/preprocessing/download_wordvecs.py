# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Downloads the GloVe vectors and unzips them"""

import zipfile
import argparse
import os
from squad_preprocess import maybe_download

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", required=True) # where to put the downloaded glove files
    return parser.parse_args()


def main():
    args = setup_args()
    glove_base_url = "http://nlp.stanford.edu/data/"
    glove_filename = "glove.6B.zip"

    print "\nDownloading wordvecs to {}".format(args.download_dir)

    if not os.path.exists(args.download_dir):
        os.makedirs(args.download_dir)

    maybe_download(glove_base_url, glove_filename, args.download_dir, 862182613L)
    glove_zip_ref = zipfile.ZipFile(os.path.join(args.download_dir, glove_filename), 'r')

    glove_zip_ref.extractall(args.download_dir)
    glove_zip_ref.close()


if __name__ == '__main__':
    main()
