import os
import numpy as np
import pandas as pd
import argparse
from utils import wavwrite, read_scaled_wav, fix_length3, convolve_hrtf3


def create_binaural_wsj0mix(wsj_root, output_root):

    S1_DIR = 's1'
    S2_DIR = 's2'
    S3_DIR = 's3'
    MIX_DIR = 'mix'
    FILELIST_STUB = os.path.join('metadata', 'mix_3_spk_filenames_{}.csv')
    BINAU = True  # Generate binaural audio

    scaling_npz_stub = os.path.join('metadata', 'scaling_{}3.npz')
    hrtf_meta_stub = os.path.join('metadata', 'hrtf_meta_{}3.csv')
    hrtf_wav_path = os.path.join('CIPIC_hrtf_database', 'wav_database')

    for sr_str in ['8k','16k']:
        wav_dir = 'wav' + sr_str
        if sr_str == '8k':
            sr = 8000
            downsample = True
        else:
            sr = 16000
            downsample = False

        for datalen_dir in ['min','max']:
            for splt in ['tt','cv','tr']:
                output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)

                s1_output_dir = os.path.join(output_path, S1_DIR)
                os.makedirs(s1_output_dir, exist_ok=True)
                s2_output_dir = os.path.join(output_path, S2_DIR)
                os.makedirs(s2_output_dir, exist_ok=True)
                s3_output_dir = os.path.join(output_path, S3_DIR)
                os.makedirs(s3_output_dir, exist_ok=True)
                mix_output_dir = os.path.join(output_path, MIX_DIR)
                os.makedirs(mix_output_dir, exist_ok=True)

                print('{} {} dataset, {} split'.format(wav_dir, datalen_dir, splt))

                # read filenames
                wsjmix_path = FILELIST_STUB.format(splt)
                wsjmix_df = pd.read_csv(wsjmix_path)
                # read scaling file
                scaling_path = scaling_npz_stub.format(splt)
                scaling_npz = np.load(scaling_path, allow_pickle=True)
                wsjmix_key = 'scaling_wsjmix_{}_{}'.format(sr_str, datalen_dir)
                scaling_mat = scaling_npz[wsjmix_key]

                hrtf_meta_path = hrtf_meta_stub.format(splt)
                hrtf_df = pd.read_csv(hrtf_meta_path)

                for i_utt, (output_name, s1_path, s2_path, s3_path) in enumerate(wsjmix_df.itertuples(index=False, name=None)):

                    s1 = read_scaled_wav(os.path.join(wsj_root, s1_path), scaling_mat[i_utt][0], downsample)
                    s2 = read_scaled_wav(os.path.join(wsj_root, s2_path), scaling_mat[i_utt][1], downsample)
                    s3 = read_scaled_wav(os.path.join(wsj_root, s3_path), scaling_mat[i_utt][2], downsample)

                    s1, s2, s3 = fix_length3(s1, s2, s3, datalen_dir)

                    # apply hrtf to binaural channels
                    if BINAU:
                        s1, s2, s3, output_name = convolve_hrtf3([s1, s2, s3], hrtf_wav_path, hrtf_df, output_name, sr)

                    mix = s1 + s2 +s3
                    wavwrite(os.path.join(mix_output_dir, output_name), mix, sr)
                    wavwrite(os.path.join(s1_output_dir, output_name), s1, sr)
                    wavwrite(os.path.join(s2_output_dir, output_name), s2, sr)
                    wavwrite(os.path.join(s3_output_dir, output_name), s3, sr)

                    if (i_utt + 1) % 500 == 0:
                        print('Completed {} of {} utterances'.format(i_utt + 1, len(wsjmix_df)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsj0-root', type=str,
                        help='Path to the folder containing wsj0/')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for writing binaural wsj0-3mix.')
    args = parser.parse_args()
    create_binaural_wsj0mix(args.wsj0_root, args.output_dir)

