import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
from noisesampler import NoiseSampler
from constants import SAMPLERATE
import argparse
from urllib.request import urlretrieve
import zipfile

def sample_noise(wsj_root, output_root):

    def reporthook(blocknum, blocksize, totalsize):
        print(
            "\rdownloading: %5.1f%%" % (100.0 * blocknum * blocksize / totalsize),
            end="",
        )
    noise_path = os.path.join(output_root, 'demand')
    if not os.path.exists(noise_path):
        print("Download DEMAND noise dataset into %s" % output_root)
        urlretrieve(
            "https://data.deepai.org/DemandDataset.zip",
            os.path.join(output_root, "DemandDataset.zip"),
            reporthook=reporthook,
        )
        file = zipfile.ZipFile(os.path.join(output_root, "DemandDataset.zip"))
        file.extractall(path=output_root)
        os.remove(os.path.join(output_root, "DemandDataset.zip"))
    
    pypath = os.path.dirname(__file__)
    NOISE_SPLIT_CSV = os.path.join(pypath, 'metadata', 'file_splits.csv')
    FILELIST_STUB = os.path.join(pypath, 'metadata', 'mix_2_spk_filenames_{}.csv')
    SPLIT_NAMES = {'Train': 'tr', 'Valid': 'cv', 'Test': 'tt'}

    SEED = 28
    np.random.seed(SEED)

    METADATA_DIR = os.path.join(output_root, 'metadata')
    os.makedirs(METADATA_DIR, exist_ok=True)

    for split_long, split_short in SPLIT_NAMES.items():

        print('Running {} Set'.format(split_long))
        filelist_path = FILELIST_STUB.format(split_short)
        filelist_df = pd.read_csv(filelist_path)
        utt_ids = list(filelist_df['output_filename'])
        utt_ids1 = list(filelist_df['s1_path'])
        utt_ids2 = list(filelist_df['s2_path'])

        output_dir = os.path.join(output_root, split_short)
        os.makedirs(output_dir, exist_ok=True)

        nz_sampler = NoiseSampler(NOISE_SPLIT_CSV, output_root, output_root, split=split_long)

        utt_list, noise_param_list, mix_param_list = [], [], []
        for i_utt, utt in enumerate(utt_ids):
            s1_path = os.path.join(wsj_root, utt_ids1[i_utt])
            s2_path = os.path.join(wsj_root, utt_ids2[i_utt])
            n_speech_samples = max(len(wavfile.read(s1_path)[1]), len(wavfile.read(s2_path)[1]))
            noise_samples, noise_param_dict, mix_param_dict = nz_sampler.sample_utt_noise(n_speech_samples)

            wavfile.write(os.path.join(output_dir, utt), SAMPLERATE,  noise_samples.astype(np.float32))

            utt_list.append(utt)
            noise_param_list.append(noise_param_dict)
            mix_param_list.append(mix_param_dict)

            if (i_utt + 1) % 500 == 0:
                print('Completed {} of {} utterances'.format(i_utt + 1, len(utt_ids)))

        noise_param_df = pd.DataFrame(data=noise_param_list, index=utt_list,
                                      columns=['noise_file', 'start_sample_16k', 'end_sample_16k', 'noise_snr'])
        noise_param_path = os.path.join(METADATA_DIR, 'noise_meta_{}.csv'.format(split_short))
        noise_param_df.to_csv(noise_param_path, index=True, index_label='utterance_id')
        mix_param_df = pd.DataFrame(data=mix_param_list, index=utt_list,
                                    columns=['noise_samples_beginning_16k', 'noise_samples_end_16k', 'target_speaker1_snr_db'])
        mix_param_path = os.path.join(METADATA_DIR, 'mix_param_meta_{}.csv'.format(split_short))
        mix_param_df.to_csv(mix_param_path, index=True, index_label='utterance_id')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsj0-root', type=str,
                        help='Path to the folder containing wsj0/')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for writing binaural wsj0-2mix with noise')
    args = parser.parse_args()
    sample_noise(args.wsj0_root, args.output_dir)