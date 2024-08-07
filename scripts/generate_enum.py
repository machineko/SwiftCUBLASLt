import re
import os
from pathlib import Path

# Pre-defined values for external constants
PREDEFINED_VALUES = {
    'CUBLAS_POINTER_MODE_HOST': 0,
    'CUBLAS_POINTER_MODE_DEVICE': 1,
    # Add any other external constants here
}

def read_cublas_enum(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def convert_cpp_enum_to_swift(cpp_enum):
    enums = re.findall(r'typedef enum {([^}]*)}\s*(\w+);', cpp_enum, re.DOTALL)
    swift_enums = []
    all_defined_values = PREDEFINED_VALUES.copy()  # Start with pre-defined values

    for enum_content, enum_name in enums:
        lines = enum_content.split('\n')
        lines_iter = iter(lines)
        swift_enum_name = re.sub(r'_t$', '', enum_name)
        swift_enum = f"public enum {swift_enum_name}: Int {{\n"
        value_to_case = {}
        defined_values = {}

        for line in lines_iter:
            line = line.strip()
            if not line:
                continue
            comment = ""
            if '/*' in line and '*/' in line:
                line, comment = line.split('/*', 1)
                comment = '// ' + comment.replace('*/', '').strip()
            elif '/*' in line:
                comment = line[line.index('/*'):]
                line = line[:line.index('/*')].strip()
                comment_lines = []
                while '*/' not in comment:
                    next_line = next(lines_iter).strip()
                    comment += '\n' + next_line
                comment = comment.strip()
                comment_lines = comment.split('\n')
                comment = '\n'.join(['// ' + c.replace('/*', '').replace('*/', '').strip() for c in comment_lines])

            if "=" in line:
                parts = line.split('=')
                enum_case = parts[0].strip()
                enum_value = parts[1].split(',')[0].strip()
                swift_case = enum_case.replace(enum_name.upper() + '_', '').lower()
                swift_case = re.sub(r'([a-z])([A-Z])', r'\1_\2', swift_case).lower()

                if '|' in enum_value:
                    values = [v.strip().strip('()') for v in enum_value.split('|')]
                    enum_value = 0
                    for v in values:
                        enum_value |= parse_value(v, defined_values, all_defined_values)
                else:
                    enum_value = parse_value(enum_value.strip('()'), defined_values, all_defined_values)

                defined_values[enum_case] = enum_value
                all_defined_values[enum_case] = enum_value

                if str(enum_value) in value_to_case:
                    swift_enum += f"    public static var {swift_case}: {swift_enum_name} {{ return .{value_to_case[str(enum_value)]} }} {comment}\n"
                else:
                    swift_enum += f"    case {swift_case} = {enum_value} {comment}\n"
                    value_to_case[str(enum_value)] = swift_case
            elif line.endswith(','):
                enum_case = line.rstrip(',')
                swift_case = enum_case.replace(enum_name.upper() + '_', '').lower()
                swift_case = re.sub(r'([a-z])([A-Z])', r'\1_\2', swift_case).lower()
                swift_enum += f"    case {swift_case} {comment}\n"

        swift_enum += "}\n"
        swift_enums.append(swift_enum)

    return "\n".join(swift_enums)

def parse_value(value, defined_values, all_defined_values):
    if value in defined_values:
        return defined_values[value]
    elif value in all_defined_values:
        return all_defined_values[value]
    elif value.startswith('0x'):
        return int(value, 16)
    elif value.isdigit():
        return int(value)
    else:
        print(f"Warning: Unrecognized value {value}")
        return 0

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def get_cublas_header_path():
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is None:
        raise Exception('CUDA_HOME environment variable is not set')
    return os.path.join(cuda_home, 'include', 'cublasLt.h')

if __name__ == "__main__":
    cublas_header_path = get_cublas_header_path()
    cublas_enum_content = read_cublas_enum(cublas_header_path)
    swift_enum_content = convert_cpp_enum_to_swift(cublas_enum_content)
    output_file_path = Path("Sources/SwiftCUBLASLt/CUBLASLtTypes/CUBLASLtEnums.swift")
    write_to_file(output_file_path, swift_enum_content)
    # os.system(f"swift-format {output_file_path} -i {output_file_path}")
