def replace_its_with_them_in_file(file_path):
    """
    Replace all occurrences of 'its' with 'them' in a single file and save changes.

    :param file_path: Path to the JSON file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace 'its' with 'them'
        updated_content = content.replace('its', 'them')
        updated_content = updated_content.replace('ites', 'them')
        updated_content = updated_content.replace('the it ', 'it ')
        updated_content = updated_content.replace('the it.', 'it.')
        updated_content = updated_content.replace('the it?', 'it?')
        updated_content = updated_content.replace('the them', 'them')
        
        # Save the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"File updated successfully: {file_path}")
    
    except Exception as e:
        print(f"Error processing file: {e}")

replace_its_with_them_in_file('alfred/data/splits/LLaMa3.1-8B_merged.json')
