import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, fftconvolve
import os


def read_scaled_wav(path, scaling_factor, downsample_8K=False):
    sr_orig, samples = wavfile.read(path)

    if len(samples.shape) > 1:
        samples = samples[:, 0]

    if downsample_8K:
        samples = resample_poly(samples, 8000, sr_orig)
    samples = (samples * scaling_factor)
    return samples


def quantize(samples):
    return np.float64(samples) / (2 ** 15)


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    wavfile.write(file, sr, samples.astype(np.int16))


def append_or_truncate(s1_samples, s2_samples, noise_samples, min_or_max='max', start_samp_16k=0, downsample=False):
    if downsample:
        speech_start_sample = start_samp_16k // 2
    else:
        speech_start_sample = start_samp_16k

    speech_end_sample = speech_start_sample + len(s1_samples)

    if min_or_max == 'min':
        noise_samples = noise_samples[speech_start_sample:speech_end_sample]
    else:
        s1_append = np.zeros_like(noise_samples)
        s2_append = np.zeros_like(noise_samples)
        s1_append[speech_start_sample:speech_end_sample] = s1_samples
        s2_append[speech_start_sample:speech_end_sample] = s2_samples
        s1_samples = s1_append
        s2_samples = s2_append

    return s1_samples, s2_samples, noise_samples


def append_or_truncate3(s1_samples, s2_samples, s3_samples, noise_samples, min_or_max='max', start_samp_16k=0, downsample=False):
    if downsample:
        speech_start_sample = start_samp_16k // 2
    else:
        speech_start_sample = start_samp_16k

    speech_end_sample = speech_start_sample + len(s1_samples)

    if min_or_max == 'min':
        noise_samples = noise_samples[speech_start_sample:speech_end_sample]
    else:
        s1_append = np.zeros_like(noise_samples)
        s2_append = np.zeros_like(noise_samples)
        s3_append = np.zeros_like(noise_samples)
        s1_append[speech_start_sample:speech_end_sample] = s1_samples
        s2_append[speech_start_sample:speech_end_sample] = s2_samples
        s3_append[speech_start_sample:speech_end_sample] = s3_samples
        s1_samples = s1_append
        s2_samples = s2_append
        s3_samples = s3_append

    return s1_samples, s2_samples, s3_samples, noise_samples


def fix_length(s1, s2, min_or_max='max'):
    # Fix length
    if min_or_max == 'min':
        utt_len = np.min([len(s1), len(s2)])
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:  # max
        utt_len = np.max([len(s1), len(s2)])
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    return s1, s2


def fix_length3(s1, s2, s3, min_or_max='max'):
    # Fix length
    if min_or_max == 'min':
        utt_len = np.min([len(s1), len(s2), len(s3)])
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
        s3 = s3[:utt_len]
    else:  # max
        utt_len = np.max([len(s1), len(s2), len(s3)])
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
        s3 = np.append(s3, np.zeros(utt_len - len(s3)))
    return s1, s2, s3


def create_wham_mixes(s1_samples, s2_samples, noise_samples):
    mix_clean = s1_samples + s2_samples
    mix_single = noise_samples + s1_samples
    mix_both = noise_samples + s1_samples + s2_samples
    return mix_clean, mix_single, mix_both


def create_wham_mixes3(s1_samples, s2_samples, s3_samples, noise_samples):
    mix_clean = s1_samples + s2_samples + s3_samples
    mix_single = noise_samples + s1_samples
    mix_both = noise_samples + s1_samples + s2_samples +s3_samples
    return mix_clean, mix_single, mix_both


def convolve_hrtf(samples_ori_list, hrtf_wav_path, hrtf_df, output_name, sr_orig):
    hrtf_row = hrtf_df[hrtf_df['utterance_id'] == output_name]
    samples_list = []
    angles_list = []

    for sub in [1,2]:
        samples_ori = samples_ori_list[sub-1]
        samples = np.zeros((len(samples_ori),2)) # binaural
        subject = hrtf_row['subject'].iloc[0]
        azimuth = hrtf_row['azimuth{}'.format(sub)].iloc[0]
        elevation = hrtf_row['elevation{}'.format(sub)].iloc[0]
        elevation_index = hrtf_row['elevation_index{}'.format(sub)].iloc[0]
        angles_list.append(azimuth)
        angles_list.append(elevation)

        for i, loc in enumerate(['left','right']):
            hrtf_file = os.path.join(hrtf_wav_path, subject, '{}az{}.wav'.format(azimuth.astype('str').replace('-','neg'), loc))
            sr, hrtf = wavfile.read(hrtf_file)
            hrtf = resample_poly(hrtf, sr_orig, sr, axis=1)
            samples[:,i] = fftconvolve(samples_ori, hrtf[elevation_index], mode='same')
        
        # Make relative source energy same with original
        spatial_scaling = np.sqrt(np.sum(samples_ori ** 2) * 2 / np.sum(samples ** 2))
        samples_list.append(samples * spatial_scaling)

    output_name = subject+'_'+output_name.split('_')[0]+'_' \
                  +str(angles_list[0])+'_' \
                  +str(angles_list[1])+'_' \
                  +output_name.split('_')[2]+'_' \
                  +str(angles_list[2])+'_' \
                  +str(angles_list[3])+'.wav'
    samples_list.append(output_name)

    return samples_list


def convolve_hrtf3(samples_ori_list, hrtf_wav_path, hrtf_df, output_name, sr_orig):
    hrtf_row = hrtf_df[hrtf_df['utterance_id'] == output_name]
    samples_list = []
    angles_list = []

    for sub in [1,2,3]:
        samples_ori = samples_ori_list[sub-1]
        samples = np.zeros((len(samples_ori),2)) # binaural
        subject = hrtf_row['subject'].iloc[0]
        azimuth = hrtf_row['azimuth{}'.format(sub)].iloc[0]
        elevation = hrtf_row['elevation{}'.format(sub)].iloc[0]
        elevation_index = hrtf_row['elevation_index{}'.format(sub)].iloc[0]
        angles_list.append(azimuth)
        angles_list.append(elevation)
        
        for i, loc in enumerate(['left','right']):
            hrtf_file = os.path.join(hrtf_wav_path, subject, '{}az{}.wav'.format(azimuth.astype('str').replace('-','neg'), loc))
            sr, hrtf = wavfile.read(hrtf_file)
            hrtf = resample_poly(hrtf, sr_orig, sr, axis=1)
            samples[:,i] = fftconvolve(samples_ori, hrtf[elevation_index], mode='same')
        
        # Make relative source energy same with original
        spatial_scaling = np.sqrt(np.sum(samples_ori ** 2) * 2 / np.sum(samples ** 2))
        samples_list.append(samples * spatial_scaling)

    output_name = subject+'_'+output_name.split('_')[0]+'_' \
                  +str(angles_list[0])+'_' \
                  +str(angles_list[1])+'_' \
                  +output_name.split('_')[2]+'_' \
                  +str(angles_list[2])+'_' \
                  +str(angles_list[3])+'_' \
                  +output_name.split('_')[4]+'_' \
                  +str(angles_list[4])+'_' \
                  +str(angles_list[5])+'.wav'
    samples_list.append(output_name)

    return samples_list


def convolve_hrtf_reverb(samples_ori_list, hrtf_wav_path, hrtf_df, output_name, sr_orig):
    hrtf_row = hrtf_df[hrtf_df['utterance_id'] == output_name]
    samples_list = []
    angles_list = []

    for sub in [1,2]:
        samples_ori = samples_ori_list[sub-1]
        samples_ori = np.stack((samples_ori, samples_ori),1)
        reverb_time = hrtf_row['reverb_time'].iloc[0]
        azimuth = hrtf_row['azimuth{}'.format(sub)].iloc[0]
        angles_list.append(azimuth)
        
        hrtf_file = os.path.join(hrtf_wav_path, reverb_time, 'CATT_{}_{}.wav'.format(reverb_time, azimuth))
        sr, hrtf = wavfile.read(hrtf_file)
        hrtf = resample_poly(hrtf, sr_orig, sr)
        samples = fftconvolve(samples_ori, hrtf, mode='same')
        
        # Make relative source energy same with original
        spatial_scaling = np.sqrt(np.sum(samples_ori ** 2) / np.sum(samples ** 2))
        samples_list.append(samples * spatial_scaling)

    output_name = reverb_time+'_'+output_name.split('_')[0]+'_' \
                  +str(angles_list[0])+'_' \
                  +output_name.split('_')[2]+'_' \
                  +str(angles_list[1])+'.wav'
    samples_list.append(output_name)

    return samples_list
