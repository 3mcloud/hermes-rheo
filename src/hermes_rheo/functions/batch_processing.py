import numpy as np
import pandas as pd
import math
from scipy.fftpack import fft, fftshift

import re
from IPython.display import display
from io import BytesIO
import copy
import base64
from IPython.display import HTML

from cralds.dataio import read_file


def moduli_ofr_chirp(strain, stress, fs, om1, om2):
    """
    Function to compute G' and G'' from strain and stress in time domain
    The result from fft gets cut between f1 an f2

    Code adapted by Alessandro Perego from the matlab MITOWCh code from MIT (2022).

    Note that in this version, the padarray function in Matlab is replaced by numpy.pad, and the pow2 function is replaced
    by 2**n. Instead of indexing with n/2+1 in Matlab, I used integer division n//2 to slice the array in Python.
    The nextpow2 function is replaced by np.log2 and the additional +1 in the power of 2.

    Also note that in Matlab isrow(x) returns true if x is a row vector, and false otherwise.
    In python there is no such built-in function to check for a 1D array, but you can use x.shape to check the number of
    dimensions of the array and x.shape[0] to check the size of the first dimension of the array and according to this, you
    can use if condition to transpose your array.
    """
    # Calculate lower and upper frequency bounds
    f1 = om1 / (2 * np.pi)
    f2 = om2 / (2 * np.pi)
    # Find the length of the input signals
    m = len(strain)
    # Find the next power of 2 greater than the length of the input signals
    n = 2 ** (int(np.log2(m)) + 1)

    # Pad the input data with zeros
    strain = np.pad(strain, (0, n-m), 'constant')
    stress = np.pad(stress, (0, n-m), 'constant')

    # FFT
    Strain0 = fft(strain, n)
    Stress0 = fft(stress, n)

    # Shifting Signal
    f = (-n / 2 + np.arange(n)) * (fs / n)  # 0-centered frequency range
    Strain = fftshift(Strain0)  # Rearrange y values
    Stress = fftshift(Stress0)  # Rearrange y values

    # Computing Moduli
    G = Stress[int(n / 2):] / Strain[int(n / 2):]
    F = f[int(n / 2):]
    F1 = F[F <= f2]
    F2 = F1[F1 >= f1]
    W = 2 * np.pi * F2
    # Transpose if W is 1D array
    if np.size(W) == 1:
        W = W[np.newaxis]
    l1 = len(F1)
    l2 = len(F2)
    ge = np.real(G)
    gv = np.imag(G)
    G_star = G[l1 - l2:l1]
    Ge = ge[l1 - l2:l1]
    Gv = gv[l1 - l2:l1]
    # Transpose if Ge or Gv is 1D array
    if np.size(G_star) == 1:
        G_star = G_star[np.newaxis]
    if np.size(Ge) == 1:
        Ge = Ge[np.newaxis]
    if np.size(Gv) == 1:
        Gv = Gv[np.newaxis]


    # Correct for right absolute value
    Stress1 = Stress[int(n / 2):] / fs
    Strain1 = Strain[int(n / 2):] / fs
    Stress2 = Stress1[l1 - l2:l1]
    Strain2 = Strain1[l1 - l2:l1]
    # Transpose if Stress2 or Strain2 is 1D array
    if np.size(Stress2) == 1:
        Stress2 = Stress2[np.newaxis]
    if np.size(Strain2) == 1:
        Strain2 = Strain2[np.newaxis]

    return W, Strain2, Stress2, Ge, Gv, G_star


def prepare_fs_data(dataset, x, y1, y2, y3):

    dataset.switch_coordinates(independent_name=x, dependent_name=y1)

    frequency = dataset.x_values
    # Convert strain values to unitless if in percentage

    if dataset.y_unit == "MPa":
        stor = copy.deepcopy(dataset.y_values) * 1e6
        dataset.y_unit = "Pa"
    else:
        stor = copy.deepcopy(dataset.y_values)

    # Switch dataset to x and y2 coordinates (time and stress)
    dataset.switch_coordinates(independent_name=x, dependent_name=y2)

    # Convert stress values to Pa if in MPa
    if dataset.y_unit == "MPa":
        loss = copy.deepcopy(dataset.y_values) * 1e6
        dataset.y_unit = "Pa"
    else:
        loss = copy.deepcopy(dataset.y_values)

    # Switch dataset to x and y3 coordinates (time and temperature)
    dataset.switch_coordinates(independent_name=x, dependent_name=y3)
    temperature = copy.deepcopy(dataset.y_values)

    return frequency, stor, loss, temperature


def prepare_chirp_data(dataset, x, y1, y2, y3, y4):

    dataset.switch_coordinates(independent_name=x, dependent_name=y1)

    time = dataset.x_values
    # Convert strain values to unitless if in percentage
    if dataset.y_unit == "%":
        strain = copy.deepcopy(dataset.y_values) * 0.01
        dataset.y_unit = "unitless"
    else:
        strain = copy.deepcopy(dataset.y_values)

    # Switch dataset to x and y2 coordinates (time and stress)
    dataset.switch_coordinates(independent_name=x, dependent_name=y2)

    stress = None
    # Convert stress values to Pa if in MPa
    if dataset.y_unit == "MPa":
        stress = copy.deepcopy(dataset.y_values) * 1e6
        dataset.y_unit = "Pa"
    # Keep stress values as is if already in Pa
    elif dataset.y_unit == "Pa":
        stress = copy.deepcopy(dataset.y_values)

    if stress is None:
        raise ValueError("Stress not set")

    # Switch dataset to x and y3 coordinates (time and temperature)
    dataset.switch_coordinates(independent_name=x, dependent_name=y3)
    temperature = copy.deepcopy(dataset.y_values)

    # Switch dataset to x and y4 coordinates (time and run time)
    dataset.switch_coordinates(independent_name=x, dependent_name=y4)
    run_time = copy.deepcopy(dataset.y_values)

    return time, strain, stress, temperature, run_time


def filter_signal(t_w, time, signal, method='tw'):
    if method == 'tw':
        corrected_signal = average_until_tw(t_w, time, signal)
    elif method == 'chirp':
        corrected_signal = average_over_chirp(t_w, time, signal)
    elif method == 'all':
        corrected_signal = average_over_all(t_w, time, signal)
    elif method == "none" or method == "None":
        corrected_signal = signal
    else:
        print('Invalid method selected. Choose "tw", "chirp", "all", or "none"')
        return None

    return corrected_signal


def average_until_tw(t_w, time, signal):
    if t_w != 0:
        # Find the index of the last time value before t_w
        index_tw = np.where(time <= t_w)[0][-1]
        # Calculate the initial value of the signal
        initial_value = np.mean(signal[:index_tw + 1])
        # Subtract the initial value from the signal
        corrected_signal = signal - initial_value
    # Compute the difference

    else:
        corrected_signal = signal

    return corrected_signal


def average_over_chirp(t_w, time, signal):
    if t_w != 0:
    # Find the index of the last time value before t_w
        index_tw = np.where(time <= t_w)[0][-1]
    # Calculate the average of the signal over the chirp only
        avg_signal = np.mean(signal[index_tw + 1:])
    # Subtract the average from the signal
        corrected_signal = signal - avg_signal

    else:
        corrected_signal = signal

    return corrected_signal


def average_over_all(t_w, time, signal):
    # Find the index of the last time value before t_w
    if t_w !=0:
        index_tw = np.where(time <= t_w)[0][-1]
        # Subtract the time average and waiting time from the signal
        corrected_signal = signal - np.mean(signal[:index_tw + 1]) - np.mean(signal[index_tw + 1:])
    else:
        corrected_signal = signal

    return corrected_signal


def select_best_filter_method(t_w, time, signal):
    """
    the function performs the following steps:

    1- Iterate over the different filtering methods.
    2- Apply each filter method to the signal.
    3- Evaluate the corrected signal by calculating the sum of the absolute values.
    Select the best filtering method based on the lowest sum, indicating the most balanced signal around zero.
    """

    methods = ['tw', 'chirp', 'all', 'none']
    best_method = None
    min_sum_abs = float('inf')

    for method in methods:
        corrected_signal = filter_signal(t_w, time, signal, method)
        sum_abs = np.sum(np.abs(corrected_signal))

        if sum_abs < min_sum_abs:
            min_sum_abs = sum_abs
            best_method = method

    best_corrected_signal = filter_signal(t_w, time, signal, best_method)

    return best_corrected_signal, best_method


# def select_best_filter_method(t_w, time, signal): #old algorithm
#
#     """
#     The function performs the following steps:
#
#     1- Iterate over the different filtering methods.
#     2- Apply each filter method to the signal.
#     3- Calculate the positive and negative areas of the filtered signal.
#     4- Select the best filtering method based on the smallest absolute difference
#     between the positive and negative areas.
#     """
#
#     def calculate_area_difference(signal):
#         positive_area = np.sum(signal[signal >= 0])
#         negative_area = np.abs(np.sum(signal[signal < 0]))
#         return np.abs(positive_area - negative_area)
#
#     methods = ['tw', 'chirp', 'all', 'none']
#     best_method = None
#     min_area_diff = float('inf')
#     best_corrected_signal = None
#
#     for method in methods:
#         corrected_signal = filter_signal(t_w, time, signal, method)
#         area_diff = calculate_area_difference(corrected_signal)
#
#         if area_diff < min_area_diff:
#             min_area_diff = area_diff
#             best_method = method
#             best_corrected_signal = corrected_signal
#
#     best_corrected_signal = filter_signal(t_w, time, signal, best_method)
#     signal_diff = best_corrected_signal - signal
#
#     return best_corrected_signal, best_method


def calculate_wave_parameters(wave_data: dict):
    # Extract the wave information
    duration_values = [wave_data[f'wave {i}']['duration (s)'] for i in range(1, len(wave_data))]
    coef_values = [wave_data[f'wave {i}']['coef'] for i in range(1, len(wave_data))]
    if len(duration_values) == 1:
        tw = 0
        T = duration_values[0]
        r = 0
        alpha = coef_values[1][2]
        w0_alpha = coef_values[1][3]

    elif len(duration_values) == 2:  # needs to split if tw in front or back
        if coef_values[0] == [0.0]:  # tw is in front
            tw = duration_values[0]
            T = duration_values[1]
            r = 0
            alpha = coef_values[1][2]
            w0_alpha = coef_values[1][3]
        else:
            tw = duration_values[1]
            T = duration_values[0]
            r = 0
            alpha = coef_values[1][2]
            w0_alpha = coef_values[1][3]

    elif len(duration_values) == 3:
        tw = 0
        T = sum(duration_values)
        pirT = coef_values[0][1]
        alpha = coef_values[0][4]
        w0_alpha = coef_values[0][5]
        r = round(math.pi / (pirT * T), 4)
    else:
        if coef_values[0] == [0.0]:  # tw is in front
            tw = duration_values[0]
            T = sum(duration_values[1:])
            pirT = coef_values[1][1]
            alpha = coef_values[1][4]
            w0_alpha = coef_values[1][5]
            r = round(math.pi / (pirT * T), 4)
        else:
            tw = duration_values[3]
            T = sum(duration_values) - tw
            pirT = coef_values[0][1]
            alpha = coef_values[0][4]
            w0_alpha = coef_values[0][5]
            r = round(math.pi / (pirT * T), 4)

    w0 = round(alpha * w0_alpha, 3)
    # high frequency w1
    if w0 * math.exp(alpha * T) > 1:
        w1 = round(w0 * math.exp(alpha * T), 3)
    else:
        w1 = round(w0 * math.exp(alpha * T), 3)
    return tw, T, r, w0, w1


def create_excel_download_link_buffer(excel_buffer, filename):
    excel_bytes = excel_buffer.getvalue()
    encoded_data = base64.b64encode(excel_bytes).decode()

    download_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' \
                    f'{encoded_data}" download="{filename}.xlsx">Download {filename}.xlsx</a>'

    return HTML(download_link)


def chirp_analysis(filepath, fft=False, original_signal=False, dma=False):
    experiments = read_file(filepath, create_composite_datasets=True, merge_redundant=False)
    filename = ".".join(experiments.details['source_file_name'].split(".")[:-1])

    excel_buffer = BytesIO()
    writer = pd.ExcelWriter(excel_buffer, engine='openpyxl')

    results = {}
    metadata = {}

    for i, experiment in enumerate(experiments):
        method = experiment.conditions['method']

        if all(x not in method for x in ['Arbitrary Wave', 'Frequency', 'Temperature', 'Multiwave']):
            continue

        if 'Arbitrary Wave' in method:

            method_key = method.split('\t')[-1]
            wave_data = experiment.details[method_key]
            strain_applied = wave_data['wave 2']['coef'][0]
            tw, T, r, w0, w1 = calculate_wave_parameters(wave_data)

            fs = wave_data['rate (pts/s)']

            print(f"Processing experiment {i} with method {method}")

            dataset = experiment[0].datasets[0]

            if dma:
                time, strain, stress, temperature, run_time = prepare_chirp_data(dataset, 'step time', 'strain', 'stress',
                                                                       'temperature', 'time')
            else:
                time, strain, stress, temperature, run_time = prepare_chirp_data(dataset, 'step time', 'strain', 'stress (step)',
                                                                       'temperature', 'time')

            strain_filtered, strain_filter = select_best_filter_method(tw, time, strain)
            stress_filtered, stress_filter = select_best_filter_method(tw, time, stress)

            f, strainFFT, stressFFT, stor, loss, G_star = moduli_ofr_chirp(strain_filtered, stress_filtered, fs, w0, w1)
            f_hz = f / (2 * np.pi)
            tan_d = loss / stor
            complex_viscosity = np.sqrt(stor ** 2 + loss ** 2) / f
            sheet_name = re.findall(r'Arbitrary Wave.*', method)[0]

            results[sheet_name] = {
                'f (rad/s)': f,
                'f (Hz)': f_hz,
                'G` (Pa)': stor,
                'G``(Pa)': loss,
                'tan(delta)': tan_d,
                '|eta|* (Pa s)': complex_viscosity,
                'G_star (abs)': np.abs(G_star)
            }

            if fft:
                results[sheet_name].update({
                    'FFT strain (real)': np.real(strainFFT),
                    'FFT strain (imag)': np.imag(strainFFT),
                    'FFT strain (abs)': np.abs(strainFFT),
                    'FFT stress (real)': np.real(stressFFT),
                    'FFT stress (imag)': np.imag(stressFFT),
                    'FFT stress (abs)': np.abs(stressFFT),
                    'G_star (real)': np.real(G_star),
                    'G_star(imag)': np.imag(G_star),
                })

            if original_signal:
                results[sheet_name].update({
                    'time (s)': time,
                    'stress (Pa)': stress_filtered,
                    'strain': strain_filtered
                })

            metadata[sheet_name] = {
                'Filename': filename,
                'Applied strain': strain_applied,
                'Sampling frequency (pts/s)': fs,
                'Waiting time (tw)': tw,
                'Starting frequency (rad/s)': w0,
                'Final frequency (rad/s)': w1,
                'Tapering parameter r': r,
                'Filter used on strain signal': strain_filter,
                'Filter used on stress signal': stress_filter,
                'Average Temperature (℃)': round(np.average(temperature), 2),
                'Standard dev. Temperature': np.std(temperature),
                'Wave start time (s)': round(run_time[0], 2),
            }

            # Find the maximum length of arrays
            max_length = max(len(value) for value in results[sheet_name].values())

            # Pad arrays with NaN values
            padded_results = {
                key: np.pad(value, (0, max_length - len(value)), mode='constant', constant_values=np.nan)
                for key, value in results[sheet_name].items()
            }

            # Convert the padded dictionary to a DataFrame
            df = pd.DataFrame(padded_results)

            # Save the DataFrame to an Excel file
            df.to_excel(writer, sheet_name=sheet_name, startrow=12, startcol=0, index=False)

            # Save the metadata for the current sheet to an Excel file
            metadata_df = pd.DataFrame(metadata[sheet_name], index=[0])
            metadata_df = metadata_df.T
            metadata_df.reset_index(inplace=True)
            metadata_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, header=False, index=False)

        elif 'Frequency' in method:

            print(f"Processing experiment {i} with method {method}")

            dataset = experiment[0].datasets[0]

            if dma:
                f, stor, loss, temperature = prepare_fs_data(dataset, 'angular frequency', 'storage modulus',
                                                             'loss modulus', 'temperature')
            else:
                f, stor, loss, temperature = prepare_fs_data(dataset, 'angular frequency', 'storage modulus',
                                                             'loss modulus', 'Temperature')
            f_hz = f / (2 * np.pi)
            tan_d = loss / stor
            complex_viscosity = np.sqrt(stor ** 2 + loss ** 2) / f

            sheet_name = re.findall(r'Frequency.*', method)[0]

            results[sheet_name] = {
                'f (rad/s)': f,
                'f (Hz)': f_hz,
                'G` (Pa)': stor,
                'G``(Pa)': loss,
                'tan(delta)': tan_d,
                '|eta|* (Pa s)': complex_viscosity,
            }

            metadata[sheet_name] = {
                'Filename': filename,
                'Starting frequency (rad/s)': f[0],
                'Final frequency (rad/s)': f[-1],
                'Average Temperature (℃)': round(np.average(temperature), 2),
                'Standard dev. Temperature': np.std(temperature)
            }

            # Find the maximum length of arrays
            max_length = max(len(value) for value in results[sheet_name].values())

            # Pad arrays with NaN values
            padded_results = {
                key: np.pad(value, (0, max_length - len(value)), mode='constant', constant_values=np.nan)
                for key, value in results[sheet_name].items()
            }

            # Convert the padded dictionary to a DataFrame
            df = pd.DataFrame(padded_results)

            # Save the DataFrame to an Excel file
            df.to_excel(writer, sheet_name=sheet_name, startrow=12, startcol=0, index=False)

            # Save the metadata for the current sheet to an Excel file
            metadata_df = pd.DataFrame(metadata[sheet_name], index=[0])
            metadata_df = metadata_df.T
            metadata_df.reset_index(inplace=True)
            metadata_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, header=False, index=False)

        elif 'Temperature' in method:

            print(f"Processing experiment {i} with method {method}")

            dataset = experiment[0].datasets[0]

            f, stor, loss, temperature = prepare_fs_data(dataset, 'angular frequency', 'storage modulus',
                                                             'loss modulus', 'Temperature')
            f_hz = f / (2 * np.pi)
            tan_d = loss / stor
            complex_viscosity = np.sqrt(stor ** 2 + loss ** 2) / f
            sheet_name = re.findall(r'Temperature sweep.*', method)[0]

            results[sheet_name] = {
                'f (rad/s)': f,
                'f (Hz)': f_hz,
                'G` (Pa)': stor,
                'G``(Pa)': loss,
                'tan(delta)': tan_d,
                '|eta|* (Pa s)': complex_viscosity,
            }

            metadata[sheet_name] = {
                'Filename': filename,
                'Starting frequency (rad/s)': f[0],
                'Final frequency (rad/s)': f[-1],
                'Average Temperature (℃)': round(np.average(temperature), 2),
                'Standard dev. Temperature': np.std(temperature)
            }

            # Find the maximum length of arrays
            max_length = max(len(value) for value in results[sheet_name].values())

            # Pad arrays with NaN values
            padded_results = {
                key: np.pad(value, (0, max_length - len(value)), mode='constant', constant_values=np.nan)
                for key, value in results[sheet_name].items()
            }

            # Convert the padded dictionary to a DataFrame
            df = pd.DataFrame(padded_results)

            # Save the DataFrame to an Excel file
            df.to_excel(writer, sheet_name=sheet_name, startrow=12, startcol=0, index=False)

            # Save the metadata for the current sheet to an Excel file
            metadata_df = pd.DataFrame(metadata[sheet_name], index=[0])
            metadata_df = metadata_df.T
            metadata_df.reset_index(inplace=True)
            metadata_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, header=False, index=False)

        elif 'Multiwave' in method:

            print(f"Processing experiment {i} with method {method}")

            dataset = experiment[0].datasets[0]

            f, stor, loss, temperature = prepare_fs_data(dataset, 'angular frequency', 'storage modulus',
                                                             'loss modulus', 'Temperature')
            f_hz = f / (2 * np.pi)
            tan_d = loss / stor
            complex_viscosity = np.sqrt(stor ** 2 + loss ** 2) / f

            sheet_name = re.findall(r'Multiwave.*', method)[0]

            results[sheet_name] = {
                'f (rad/s)': f,
                'f (Hz)': f_hz,
                'G` (Pa)': stor,
                'G``(Pa)': loss,
                'tan(delta)': tan_d,
                '|eta|* (Pa s)': complex_viscosity,
            }

            metadata[sheet_name] = {
                'Filename': filename,
                'Starting frequency (rad/s)': f[0],
                'Final frequency (rad/s)': f[-1],
                'Average Temperature (℃)': round(np.average(temperature), 2),
                'Standard dev. Temperature': np.std(temperature)
            }

            # Find the maximum length of arrays
            max_length = max(len(value) for value in results[sheet_name].values())

            # Pad arrays with NaN values
            padded_results = {
                key: np.pad(value, (0, max_length - len(value)), mode='constant', constant_values=np.nan)
                for key, value in results[sheet_name].items()
            }

            # Convert the padded dictionary to a DataFrame
            df = pd.DataFrame(padded_results)

            # Save the DataFrame to an Excel file
            df.to_excel(writer, sheet_name=sheet_name, startrow=12, startcol=0, index=False)

            # Save the metadata for the current sheet to an Excel file
            metadata_df = pd.DataFrame(metadata[sheet_name], index=[0])
            metadata_df = metadata_df.T
            metadata_df.reset_index(inplace=True)
            metadata_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, header=False, index=False)
    writer.close()
    excel_buffer.seek(0)

    excel_link = create_excel_download_link_buffer(excel_buffer, filename)

    display(excel_link)

    return results, metadata
