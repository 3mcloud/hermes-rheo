import matplotlib.pyplot as plt
import io
import copy
import os
from cralds.dataio import read_file


def plot_moduli(f, stor, loss, filename, y1=None, y2=None):
    # Plotting Parameters
    plt.figure(figsize=(8, 6))
    plt.rc('font', family='DejaVu Sans')
    plt.rc('text', usetex=False)

    plt.rcParams['xtick.major.pad'] = '10'
    plt.rcParams['ytick.major.pad'] = '10'

    plt.rcParams['axes.linewidth'] = 1.5  # set the value globally

    plt.tick_params(bottom=True, top=True, left=True, right=True,
                    direction='in', which='minor', size=4)
    plt.tick_params(bottom=True, top=True, left=True, right=True,
                    direction='in', which='major', size=8)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.xlabel(r'$\omega$ (rad/s)', fontsize=20)
    plt.ylabel(r"$G'$, $G''$ (Pa)", fontsize=20)

    plt.yscale("log")
    plt.xscale("log")

    plt.plot(f, stor, marker='s', label=r"$G'$", color='red', markersize=10, linestyle='-')
    plt.plot(f, loss, marker='^', label=r"$G''$", color='blue', markersize=10, markerfacecolor='none', linestyle='--')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    leg = plt.legend(loc='best', frameon=False, prop={'size': 19},
                     ncol=1, fontsize=18)
    leg.get_frame().set_edgecolor('k')

    plt.title(filename, fontsize=16)

    if y1 is not None and y2 is not None:
        plt.ylim(y1, y2)

    plt.tight_layout()

    # Save the figure as a svg file (you can change format to .png, .jpeg)
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    buf.seek(0)
    binary_data = buf.getvalue()
    plt.show()

    return binary_data


def plot_temperature_sweep(temp, stor, loss, filename, method_condition):
    # Plotting Parameters
    plt.figure(figsize=(8, 6))
    plt.rc('font', family='DejaVu Sans')
    plt.rc('text', usetex=False)

    plt.rcParams['xtick.major.pad'] = '10'
    plt.rcParams['ytick.major.pad'] = '10'

    plt.rcParams['axes.linewidth'] = 1.5  # set the value globally

    plt.tick_params(bottom=True, top=True, left=True, right=True,
                    direction='in', which='minor', size=4)
    plt.tick_params(bottom=True, top=True, left=True, right=True,
                    direction='in', which='major', size=8)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.xlabel(r'$T$ (${^o}$C)', fontsize=20)
    plt.ylabel(r"$G'$, $G''$ (Pa)", fontsize=20)

    plt.yscale("log")

    plt.plot(temp, stor, marker='s', label=r"$G'$", color='red', markersize=10, linestyle='-')
    plt.plot(temp, loss, marker='^', label=r"$G''$", color='blue', markersize=10, markerfacecolor='none', linestyle='--')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    leg = plt.legend(loc='best', frameon=False, prop={'size': 19},
                     ncol=1, fontsize=18)
    leg.get_frame().set_edgecolor('k')

    plt.title((f"{filename} - {method_condition}"), fontsize=16)

    plt.tight_layout()

    # Save the figure as a svg file (you can change format to .png, .jpeg)
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    buf.seek(0)
    binary_data = buf.getvalue()
    plt.show()

    return binary_data


def plot_frequency_sweep(f, stor, loss, filename, method_condition):
    # Plotting Parameters
    plt.figure(figsize=(8, 6))
    plt.rc('font', family='DejaVu Sans')
    plt.rc('text', usetex=False)

    plt.rcParams['xtick.major.pad'] = '10'
    plt.rcParams['ytick.major.pad'] = '10'

    plt.rcParams['axes.linewidth'] = 1.5  # set the value globally

    plt.tick_params(bottom=True, top=True, left=True, right=True,
                    direction='in', which='minor', size=4)
    plt.tick_params(bottom=True, top=True, left=True, right=True,
                    direction='in', which='major', size=8)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.xlabel(r'$\omega$ (rad/s)', fontsize=20)
    plt.ylabel(r"$G'$, $G''$ (Pa)", fontsize=20)

    plt.yscale("log")
    plt.xscale("log")

    plt.plot(f, stor, marker='s', label=r"$G'$", color='red', markersize=10, linestyle='-')
    plt.plot(f, loss, marker='^', label=r"$G''$", color='blue', markersize=10, markerfacecolor='none', linestyle='--')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    leg = plt.legend(loc='best', frameon=False, prop={'size': 19},
                     ncol=1, fontsize=18)
    leg.get_frame().set_edgecolor('k')

    plt.title(f"{filename} - {method_condition}", fontsize=16)

    plt.tight_layout()

    # Save the figure as a svg file (you can change format to .png, .jpeg)
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    buf.seek(0)
    binary_data = buf.getvalue()
    plt.show()

    return binary_data


def convert_to_Pa(dataset):
    """
    Checks if the y_unit of the dataset is MPa and converts it to Pa if necessary.
    """
    stress = None
    if dataset.y_unit == "MPa":
        stress = copy.deepcopy(dataset.y_values) * 1e6
        dataset.y_unit = "Pa"
    elif dataset.y_unit == "Pa":
        stress = copy.deepcopy(dataset.y_values)
    return stress


def process_files(filepath):
    """
    Reads all .txt files in the directory specified by filepath, and processes the datasets in each file according to
    the method specified in the Measurement objects.
    """
    all_dataframes = []

    # loop through all files in the directory
    for filename in os.listdir(filepath):
        if filename.endswith(".txt"):
            # construct the full path to the file
            file_path = os.path.join(filepath, filename)

            # read the file and create the experiments object
            experiments = read_file(file_path, create_composite_datasets=True)

            # process each dataset in the experiments object
            for experiment in experiments:
                for dataset in experiment[0].datasets:
                    # determine the method and switch coordinates accordingly
                    m = experiment[0]
                    method_condition = m.conditions["method"]
                    if method_condition.startswith("Temperature Ramp"):
                        dataset.switch_coordinates(independent_name='temperature', dependent_name='G\'')
                        temperature = dataset.x_values
                        gprime = convert_to_Pa(dataset)

                        dataset.switch_coordinates(independent_name='temperature', dependent_name='G\"')
                        gdprime = convert_to_Pa(dataset)

                        dataset.switch_coordinates(independent_name='temperature', dependent_name='frequency')
                        frequency = dataset.y_values

                        filename = ".".join(experiments.details['source_file_name'].split(".")[:-1])

                        t_sweep = plot_temperature_sweep(temperature, gprime, gdprime, filename, method_condition)

                    elif method_condition.startswith("Frequency"):
                        dataset.switch_coordinates(independent_name='frequency', dependent_name='G\'')
                        frequency = dataset.x_values
                        gprime = convert_to_Pa(dataset)

                        dataset.switch_coordinates(independent_name='frequency', dependent_name='G\"')
                        gdprime = convert_to_Pa(dataset)

                        dataset.switch_coordinates(independent_name='frequency', dependent_name='temperature')
                        temperature = dataset.y_values

                        # get the filename from the experiments details
                        filename = ".".join(experiments.details['source_file_name'].split(".")[:-1])

                        # plot the temperature/frequency, gprime, and gdprime values on a linear/log scale with a title
                        f_sweep = plot_frequency_sweep(frequency, gprime, gdprime, filename, method_condition)
#                         fig, ax = plt.subplots()
#                         ax.plot(frequency, gprime, label="G'")
#                         ax.plot(frequency, gdprime, label="G\"")
#                         ax.set_xlabel('Frequency (rad/s)')
#                         ax.set_xscale('log')
#                         ax.set_yscale('log')
#                         ax.set_ylabel('Modulus (Pa)')
#                         ax.set_title(f"{filename} - {method_condition}")
#                         ax.legend()
#                         plt.show()

# data = {'Temperature': temperature, 'Frequency': frequency, 'G\'': gprime, 'G\"': gdprime}
# df = pd.DataFrame(data)

# append the DataFrame to the list of all dataframes
# sheet_name = f"{filename} - {method_condition}"
# all_dataframes.append((sheet_name, df))

