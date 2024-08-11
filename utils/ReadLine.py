def read_lines(file_path, line_spec):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        if '-' in line_spec:  # Handle range of lines
            start_line, end_line = map(int, line_spec.split('-'))
            if start_line < 1 or end_line > len(lines):
                return f"Error: The range {line_spec} is out of bounds."
            return ''.join(lines[start_line-1:end_line]).strip()
        else:  # Handle single line
            line_number = int(line_spec)
            if line_number < 1 or line_number > len(lines):
                return f"Error: Line {line_number} does not exist in the file."
            return lines[line_number-1].strip()

    except FileNotFoundError:
        return f"Error: The file at {file_path} was not found."
    except Exception as e:
        return f"An error occurred: {e}"
