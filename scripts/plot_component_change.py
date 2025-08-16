import re
import matplotlib.pyplot as plt
from datetime import datetime
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler('color', ['#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', ])


# Paths to merging thresholds
co_05_log = './component_change_logs/mt_05/camera_obscura.log'
mh_05_log = './component_change_logs/mt_05/modern_hall.log'
lr_05_log = './component_change_logs/mt_05/living_room.log'
dr_05_log = './component_change_logs/mt_05/dining_room.log'

co_09_log = './component_change_logs/mt_09/camera_obscura.log'
mh_09_log = './component_change_logs/mt_09/modern_hall.log'
lr_09_log = './component_change_logs/mt_09/living_room.log'
dr_09_log = './component_change_logs/mt_09/dining_room.log'

co_065_log = './component_change_logs/mt_065/camera_obscura.log'
mh_065_log = './component_change_logs/mt_065/modern_hall.log'
lr_065_log = './component_change_logs/mt_065/living_room.log'
dr_065_log = './component_change_logs/mt_065/dining_room.log'

# Paths to splitting thresholds
co_100_log = './component_change_logs/st_100/camera_obscura.log'
mh_100_log = './component_change_logs/st_100/modern_hall.log'
lr_100_log = './component_change_logs/st_100/living_room.log'
dr_100_log = './component_change_logs/st_100/dining_room.log'

co_1000_log = './component_change_logs/st_1000/camera_obscura.log'
mh_1000_log = './component_change_logs/st_1000/modern_hall.log'
lr_1000_log = './component_change_logs/st_1000/living_room.log'
dr_1000_log = './component_change_logs/st_1000/dining_room.log'

co_10000_log = './component_change_logs/st_10000/camera_obscura.log'
mh_10000_log = './component_change_logs/st_10000/modern_hall.log'
lr_10000_log = './component_change_logs/st_10000/living_room.log'
dr_10000_log = './component_change_logs/st_10000/dining_room.log'

GMM_PATTERN = re.compile(
    r'weight\s*=\s*([\d.eE+-]+)\s*'
    r'mean\s*=\s*\[([^\]]+)\]\s*'
    r'covariance\s*=\s*\[\[([\d.eE+\-\s\[\]]+)\]\]'
)
LOG_PATTERN = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+)\s+(\S+) \[(.*?)\] (.*)$')

def parse_gmm(message):
    components = []
    for match in GMM_PATTERN.finditer(message):
        weight = float(match.group(1))
        mean = [float(x) for x in match.group(2).strip().split()]
        covariance = [[float(x) for x in row.split()] for row in match.group(3).split('][')]
        components.append({
            'weight': weight,
            'mean': mean,
            'covariance': covariance
        })
    return components

def _parse_entry(entry_lines):
    first_line = entry_lines[0]
    match = LOG_PATTERN.match(first_line)
    if not match:
        return None

    timestamp, log_level, thread, obj, message = match.groups()
    message += ''.join(entry_lines[1:])  # Append multiline content if any
    return {
        'timestamp': timestamp,
        'log_level': log_level,
        'thread': thread,
        'object': obj,
        'message': message.strip()
    }

def log_entry_generator(file_path):
    with open(file_path, 'r') as file:
        current_entry = []
        for line in file:
            match = LOG_PATTERN.match(line)
            if match:
                # If we already have an entry collected, yield it first
                if current_entry:
                    yield _parse_entry(current_entry)
                    current_entry = []

                # Start a new entry
                current_entry.append(line)
            elif current_entry:
                # Multiline message continuation
                current_entry.append(line)

        # Yield the last entry if any
        if current_entry:
            yield _parse_entry(current_entry)

def get_comp_count_and_timestamps(log_file_path):
    timestamps = []
    component_counts = []
    for entry in log_entry_generator(log_file_path):
        message = entry["message"]
        if 'GMM[' in message:
            gmm = parse_gmm(message)
            component_counts.append(len(gmm))
            timestamp_str = entry["timestamp"]
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            timestamps.append(timestamp)
    return (component_counts, timestamps)

def get_normalized_progression(log_file_path):
    x_positions = []
    component_counts = []

    current_iteration = -1
    current_iteration_gmm_counts = []

    for entry in log_entry_generator(log_file_path):
        message = entry["message"]

        if 'ITERATION' in message:
            # Finalize previous iteration
            total = len(current_iteration_gmm_counts)
            for idx, count in enumerate(current_iteration_gmm_counts):
                normalized_x = current_iteration + (idx / total)
                x_positions.append(normalized_x)
                component_counts.append(count)
            current_iteration_gmm_counts = []
            current_iteration += 1

        elif 'GMM[' in message:
            gmm = parse_gmm(message)
            current_iteration_gmm_counts.append(len(gmm))

    # Finalize the last iteration
    if current_iteration_gmm_counts:
        total = len(current_iteration_gmm_counts)
        for idx, count in enumerate(current_iteration_gmm_counts):
            normalized_x = current_iteration + (idx / total)
            x_positions.append(normalized_x)
            component_counts.append(count)

    return x_positions, component_counts

def get_progression(log_file_path):
    x_positions = []
    component_counts = []
    iter = 0
    for entry in log_entry_generator(log_file_path):
        message = entry["message"]

        if 'GMM[' in message:
            gmm = parse_gmm(message)
            component_counts.append(len(gmm))
            x_positions.append(iter)
            iter+=1

    return x_positions, component_counts

def get_end_of_iteration_component_counts(log_file_path):
    component_counts = []
    last_gmm_count = None

    for entry in log_entry_generator(log_file_path):
        message = entry["message"]

        if 'GMM[' in message:
            gmm = parse_gmm(message)
            last_gmm_count = len(gmm)

        elif 'ITERATION' in message:
            # Push the last seen GMM count before the iteration line
            if last_gmm_count is not None:
                component_counts.append(last_gmm_count)
                last_gmm_count = None

    # Optionally include final GMM state after the last iteration
    if last_gmm_count is not None:
        component_counts.append(last_gmm_count)

    return list(range(len(component_counts))), component_counts


x_co_05, y_co_05 = get_end_of_iteration_component_counts(co_05_log)
x_dr_05, y_dr_05 = get_end_of_iteration_component_counts(dr_05_log)
x_lr_05, y_lr_05 = get_end_of_iteration_component_counts(lr_05_log)
x_mh_05, y_mh_05 = get_end_of_iteration_component_counts(mh_05_log)

x_co_065, y_co_065 = get_end_of_iteration_component_counts(co_065_log)
x_dr_065, y_dr_065 = get_end_of_iteration_component_counts(dr_065_log)
x_lr_065, y_lr_065 = get_end_of_iteration_component_counts(lr_065_log)
x_mh_065, y_mh_065 = get_end_of_iteration_component_counts(mh_065_log)

x_co_09, y_co_09 = get_end_of_iteration_component_counts(co_09_log)
x_dr_09, y_dr_09 = get_end_of_iteration_component_counts(dr_09_log)
x_lr_09, y_lr_09 = get_end_of_iteration_component_counts(lr_09_log)
x_mh_09, y_mh_09 = get_end_of_iteration_component_counts(mh_09_log)

x_co_100, y_co_100 = get_end_of_iteration_component_counts(co_100_log)
x_dr_100, y_dr_100 = get_end_of_iteration_component_counts(dr_100_log)
x_lr_100, y_lr_100 = get_end_of_iteration_component_counts(lr_100_log)
x_mh_100, y_mh_100 = get_end_of_iteration_component_counts(mh_100_log)

x_co_1000, y_co_1000 = get_end_of_iteration_component_counts(co_1000_log)
x_dr_1000, y_dr_1000 = get_end_of_iteration_component_counts(dr_1000_log)
x_lr_1000, y_lr_1000 = get_end_of_iteration_component_counts(lr_1000_log)
x_mh_1000, y_mh_1000 = get_end_of_iteration_component_counts(mh_1000_log)

x_co_10000, y_co_10000 = get_end_of_iteration_component_counts(co_10000_log)
x_dr_10000, y_dr_10000 = get_end_of_iteration_component_counts(dr_10000_log)
x_lr_10000, y_lr_10000 = get_end_of_iteration_component_counts(lr_10000_log)
x_mh_10000, y_mh_10000 = get_end_of_iteration_component_counts(mh_10000_log)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
# axs[0][0].plot(x_co_05, y_co_05, linestyle='-', label='0.5')
# axs[0][0].plot(x_co_065, y_co_065, linestyle='-', label='0.65')
# axs[0][0].plot(x_co_09, y_co_09, linestyle='-', label='0.9')
# axs[0][0].grid(True)
# axs[0][0].set_title('Camera Obscura')
# # axs[0][0].legend()

# axs[0][1].plot(x_dr_05, y_dr_05, linestyle='-')
# axs[0][1].plot(x_dr_065, y_dr_065, linestyle='-')
# axs[0][1].plot(x_dr_09, y_dr_09, linestyle='-')
# axs[0][1].grid(True)
# axs[0][1].set_title('Dining Room')
# # axs[0][1].legend()

# axs[1][0].plot(x_lr_05, y_lr_05, linestyle='-')
# axs[1][0].plot(x_lr_065, y_lr_065, linestyle='-')
# axs[1][0].plot(x_lr_09, y_lr_09, linestyle='-')
# axs[1][0].grid(True)
# axs[1][0].set_title('Living Room')
# # axs[1][0].legend()

# axs[1][1].plot(x_mh_05, y_mh_05, linestyle='-')
# axs[1][1].plot(x_mh_065, y_mh_065, linestyle='-')
# axs[1][1].plot(x_mh_09, y_mh_09, linestyle='-')
# axs[1][1].grid(True)
# axs[1][1].set_title('Modern Hall')

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0][0].plot(x_co_100, y_co_100, linestyle='-', label='100')
axs[0][0].plot(x_co_1000, y_co_1000, linestyle='-', label='1000')
axs[0][0].plot(x_co_10000, y_co_10000, linestyle='-', label='10000')
axs[0][0].grid(True)
axs[0][0].set_title('Camera Obscura')
# axs[0][0].legend()

axs[0][1].plot(x_dr_100, y_dr_100, linestyle='-')
axs[0][1].plot(x_dr_1000, y_dr_1000, linestyle='-')
axs[0][1].plot(x_dr_10000, y_dr_10000, linestyle='-')
axs[0][1].grid(True)
axs[0][1].set_title('Dining Room')
# axs[0][1].legend()

axs[1][0].plot(x_lr_100, y_lr_100, linestyle='-')
axs[1][0].plot(x_lr_1000, y_lr_1000, linestyle='-')
axs[1][0].plot(x_lr_10000, y_lr_10000, linestyle='-')
axs[1][0].grid(True)
axs[1][0].set_title('Living Room')
# axs[1][0].legend()

axs[1][1].plot(x_mh_100, y_mh_100, linestyle='-')
axs[1][1].plot(x_mh_1000, y_mh_1000, linestyle='-')
axs[1][1].plot(x_mh_10000, y_mh_10000, linestyle='-')
axs[1][1].grid(True)
axs[1][1].set_title('Modern Hall')

# axs[1][1].legend()
fig.legend(loc='outside upper center', fontsize="x-large", ncol=3)
plt.tight_layout()
plt.show()
