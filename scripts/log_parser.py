import re

LOG_PATTERN = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+)\s+(\S+) \[(.*?)\] (.*)$')

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

GMM_PATTERN = re.compile(
    r'weight\s*=\s*([\d.eE+-]+)\s*'
    r'mean\s*=\s*\[([^\]]+)\]\s*'
    r'covariance\s*=\s*\[\[([\d.eE+\-\s\[\]]+)\]\]'
)

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

POINT_PATTERN = re.compile(r'point:\s*\[([^\]]+)\],\s*weight:\s*([\d.eE+-]+)')
VALID_SAMPLES_PATTERN = re.compile(r'Collected (\d+) valid samples')

def parse_point(message):
    match = POINT_PATTERN.search(message)
    if match:
        point = [float(x) for x in match.group(1).split(',')]
        weight = float(match.group(2))
        return {
            'point': point,
            'weight': weight
        }

INTERSECTION_PATTERN = re.compile(r'Intersection coordinates:\s*\[([^\]]+)\]')
RAY_PATTERN = re.compile(r'origin:\s*\[([^\]]+)\]\s*direction:\s*\[([^\]]+)\]')


def parse_intersection(message, next_message):
    intersection_match = INTERSECTION_PATTERN.search(message)
    ray_match = RAY_PATTERN.search(next_message)

    if intersection_match and ray_match:
        intersection = [float(x) for x in intersection_match.group(1).split(',')]
        origin = [float(x) for x in ray_match.group(1).split(',')]
        direction = [float(x) for x in ray_match.group(2).split(',')]

        return {
            'intersection': intersection,
            'origin': origin,
            'direction': direction
        }

    return None

OCTREE_PATTERN = re.compile(r'Octree\[')
AABB_PATTERN = re.compile(r'AABB3\[min=\[([^\]]+)\], max=\[([^\]]+)\]\]')
INDEX_ZERO_PATTERN = re.compile(r'Index:\s*0')

def parse_octree(message):
    lines = message.splitlines()
    result = {
        "main_aabb": None,
        "leaf_nodes": []
    }

    main_aabb_match = AABB_PATTERN.search(lines[0])
    if main_aabb_match:
        result["main_aabb"] = {
            "min": [float(x) for x in main_aabb_match.group(1).split(',')],
            "max": [float(x) for x in main_aabb_match.group(2).split(',')]
        }

    for line in lines[1:]:
        if INDEX_ZERO_PATTERN.search(line):
            aabb_match = AABB_PATTERN.search(line)
            if aabb_match:
                result["leaf_nodes"].append({
                    "min": [float(x) for x in aabb_match.group(1).split(',')],
                    "max": [float(x) for x in aabb_match.group(2).split(',')]
                })

    return result

def parse_log_file(log_file_path):
    GMMs = []
    valid_samples = []
    intersection_data = []
    previous_entry = None
    octrees = []

    current_valid_samples = None

    for entry in log_entry_generator(log_file_path):
        message = entry["message"]
        if 'Octree[' in message:
            octree = parse_octree(message)
            if octree:
                octrees.append(octree)
            continue


        if 'GMM[' in message:
            GMMs.append(parse_gmm(message))
            continue

        if previous_entry and 'Ray data: ' in message:
            intersection = parse_intersection(previous_entry["message"], message)
            if intersection:
                intersection_data.append(intersection)
                previous_entry = None
                continue

        if 'Intersection coordinates:' in message:
            previous_entry = entry
            continue

        valid_samples_match = VALID_SAMPLES_PATTERN.search(message)
        if valid_samples_match:
            n_samples = int(valid_samples_match.group(1))
            if n_samples != 0:
                current_valid_samples = []  # Start collecting samples
            continue

        if current_valid_samples is not None and 'point: [' in message:
            sample = parse_point(message)  # Parse the current line as a point
            if sample:
                current_valid_samples.append(sample)

                if len(current_valid_samples) == n_samples:
                    valid_samples.append(current_valid_samples)
                    current_valid_samples = None

    return (GMMs, valid_samples, intersection_data, octrees)
