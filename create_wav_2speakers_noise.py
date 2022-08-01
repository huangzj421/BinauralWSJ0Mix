import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
from constants import SAMPLERATE
import argparse
from utils import read_scaled_wav, quantize, fix_length, create_wham_mixes, append_or_truncate, convolve_hrtf


def create_binaural_wsj0mix(wsj_root, output_root, 
                            datafreqs=['8k','16k'], datamodes=['min','max'], wsjmix_16k_root=None, wsjmix_8k_root=None):

    
    SINGLE_DIR = 'mix_single'
    BOTH_DIR = 'mix_both'
    CLEAN_DIR = 'mix_clean'
    S1_DIR = 's1'
    S2_DIR = 's2'
    NOISE_DIR = 'noise'
    BINAU = True  # Generate binaural audio
    
    pypath = os.path.dirname(__file__)
    FILELIST_STUB = os.path.join(pypath, 'metadata', 'mix_2_spk_filenames_{}.csv')
    scaling_npz_stub = os.path.join(pypath, 'metadata', 'scaling_{}.npz')
    hrtf_meta_stub = os.path.join(pypath, 'metadata', 'hrtf_meta_{}.csv')
    hrtf_wav_path = os.path.join(pypath, 'CIPIC_hrtf_database', 'wav_database')
    
    noise_root = os.path.join(output_root, 'noisedata')
    os.makedirs(noise_root, exist_ok=True)
    if not os.path.exists(os.path.join(noise_root, 'metadata')):
        from run_sample_noise import sample_noise
        sample_noise(wsj_root, noise_root)

    if wsj_root is not None:
        from_scratch = True
    else:
        from_scratch = False

    for splt in ['tr', 'cv', 'tt']:

        wsjmix_path = FILELIST_STUB.format(splt)
        wsjmix_df = pd.read_csv(wsjmix_path)

        scaling_npz_path = scaling_npz_stub.format(splt)
        scaling_npz = np.load(scaling_npz_path, allow_pickle=True)

        noise_path = os.path.join(noise_root, splt)

        hrtf_meta_path = hrtf_meta_stub.format(splt)
        hrtf_df = pd.read_csv(hrtf_meta_path)

        for sr_dir in datafreqs:
            wav_dir = 'wav' + sr_dir
            if sr_dir == '8k':
                sr = 8000
                downsample = True
                wsjmix_path = wsjmix_8k_root
            else:
                sr = SAMPLERATE
                downsample = False
                wsjmix_path = wsjmix_16k_root

            for datalen_dir in datamodes:
                
                print('{} {} dataset, {} split'.format(sr_dir, datalen_dir, splt))
                
                output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)
                os.makedirs(os.path.join(output_path, CLEAN_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, SINGLE_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, BOTH_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, S1_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, S2_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, NOISE_DIR), exist_ok=True)

                wsjmix_key = 'scaling_wsjmix_{}_{}'.format(sr_dir, datalen_dir)
                wham_speech_key = 'scaling_wham_speech_{}_{}'.format(sr_dir, datalen_dir)
                wham_noise_key = 'scaling_wham_noise_{}_{}'.format(sr_dir, datalen_dir)

                utt_ids = scaling_npz['utterance_id']
                start_samp_16k = scaling_npz['speech_start_sample_16k']
                scaling_noise = scaling_npz[wham_noise_key]

                scaling_speech = scaling_npz[wham_speech_key]
                if from_scratch:
                    scaling_wsjmix = scaling_npz[wsjmix_key]

                for i_utt, output_name in enumerate(utt_ids):
                    if from_scratch:
                        utt_row = wsjmix_df[wsjmix_df['output_filename'] == output_name]
                        s1_path = os.path.join(wsj_root, utt_row['s1_path'].iloc[0])
                        s2_path = os.path.join(wsj_root, utt_row['s2_path'].iloc[0])

                        s1 = read_scaled_wav(s1_path, scaling_wsjmix[i_utt][0], downsample)
                        s1 = quantize(s1) * scaling_speech[i_utt]
                        s2 = read_scaled_wav(s2_path, scaling_wsjmix[i_utt][1], downsample)
                        s2 = quantize(s2) * scaling_speech[i_utt]
                        s1_samples, s2_samples = fix_length(s1, s2, datalen_dir)
                    else:
                        wsj_path = os.path.join(wsjmix_path, datalen_dir, splt)
                        s1_path = os.path.join(wsj_path, S1_DIR, output_name)
                        s1_samples = read_scaled_wav(s1_path, scaling_speech[i_utt])
                        s2_path = os.path.join(wsj_path, S2_DIR, output_name)
                        s2_samples = read_scaled_wav(s2_path, scaling_speech[i_utt])

                    noise_samples = read_scaled_wav(os.path.join(noise_path, output_name), scaling_noise[i_utt],
                                                    downsample_8K=downsample)
                    
                    # apply hrtf to binaural channels
                    if BINAU:
                        s1_samples, s2_samples, output_name = convolve_hrtf([s1_samples, s2_samples], hrtf_wav_path, hrtf_df, output_name, sr)
                        noise_samples = np.stack((noise_samples,noise_samples),1)

                    s1_samples, s2_samples, noise_samples = append_or_truncate(s1_samples, s2_samples, noise_samples,
                                                                               datalen_dir, start_samp_16k[i_utt], downsample)

                    mix_clean, mix_single, mix_both = create_wham_mixes(s1_samples, s2_samples, noise_samples)

                    # write audio
                    samps = [mix_clean, mix_single, mix_both, s1_samples, s2_samples, noise_samples]
                    dirs = [CLEAN_DIR, SINGLE_DIR, BOTH_DIR, S1_DIR, S2_DIR, NOISE_DIR]
                    for dir, samp in zip(dirs, samps):
                        wavfile.write(os.path.join(output_path, dir, output_name), sr, samp.astype(np.float32))

                    if (i_utt + 1) % 500 == 0:
                        print('Completed {} of {} utterances'.format(i_utt + 1, len(wsjmix_df)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsj0-root', type=str,
                        help='Path to the folder containing wsj0/')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for writing binaural wsj0-2mix with noise')
    args = parser.parse_args()
    create_binaural_wsj0mix(args.wsj0_root, args.output_dir)
