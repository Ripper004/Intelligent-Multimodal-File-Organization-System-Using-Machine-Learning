import os

def show_full_tree(directory):
    print(f"✨ FULL PROJECT OUTPUT TREE ✨")
    print("="*30)
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}├── 📁 {os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}└── 📄 {f}")

# RUN THIS:
# Make sure this points to where your code actually moved the files
output_path = r"C:\Users\DELL\Desktop\FILE_ORGANIZER\audio\Organized_Files"
show_full_tree(output_path)