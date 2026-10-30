import os, re
from collections import Counter

def clean_file_tokens(paths):
    all_tokens = []
    # Expanded blacklist to remove emails and system noise
    blacklist = {
        'the', 'and', 'from', 'with', 'docx', 'pdf', 'pptx', 'image', 'video', 
        'final', 'copy', 'whatsapp', 'screenshot', 'untitled', 'protyush', 
        'saha', '22bce5078', 'module', 'years', 'later', 'category', 'collection',
        'document', 'file', 'attachment', 'original', 'revised', 'version',
        'gmail', 'com', 'mail'
    }
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0].lower()
        
        # 1. Catch Meet/Zoom recordings
        if re.match(r'[a-z]{3}-[a-z]{4}-[a-z]{3}', name):
            all_tokens.append("recordings")
            continue
            
        # 2. Remove email addresses from token consideration
        name = re.sub(r'\S+@\S+', ' ', name)
        name = re.sub(r'\b(19|20)\d{2}\b', ' ', name)
        name = re.sub(r'[^a-z]', ' ', name)
        
        tokens = [w for w in name.split() if len(w) > 3 and w not in blacklist]
        all_tokens.extend(tokens)
    return all_tokens

def generate_leaf_cluster_name(paths):
    tokens = clean_file_tokens(paths)
    if not tokens: return "Uncategorized_Files"
    
    if "recordings" in tokens: return "Meeting_Sessions"
    
    counts = Counter(tokens)
    common = counts.most_common(3)
    
    # Priority Keywords specific to your report findings
    priority_map = {
        "internship": "Internships",
        "compiler": "Compiler_Lab",
        "dropout": "Dropout_Analysis",
        "gender": "Gender_Studies",
        "chamber": "Chamber_Visuals",
        "assignment": "Assignments",
        "receipt": "Financial_Receipts"
    }
    
    for word, _ in common:
        if word in priority_map: return priority_map[word]

    if len(common) >= 2:
        return f"{common[0][0].capitalize()}_{common[1][0].capitalize()}"
    return common[0][0].capitalize()

def generate_superfolder_name(leaf_names):
    combined = " ".join(leaf_names).lower()
    
    # Smarter theme mapping
    if any(x in combined for x in ["meeting", "session", "clipchamp"]):
        return "Digital_Media_Production"
    if any(x in combined for x in ["gender", "dropout", "education"]):
        return "Social_Research_Project"
    if any(x in combined for x in ["compiler", "lab", "bcse"]):
        return "CS_Core_Curriculum"
    
    all_words = []
    for name in leaf_names:
        all_words.extend(re.sub(r'[^a-z]', ' ', name.lower()).split())
    
    counts = Counter(all_words)
    top_word, count = counts.most_common(1)[0] if counts else ("General", 0)
    
    theme_map = {
        "internship": "Professional_Career",
        "movie": "Entertainment_Vault",
        "assignment": "Academic_Submissions"
    }
    
    if top_word in theme_map: return theme_map[top_word]
    return f"{top_word.capitalize()}_Collection" if count >= 1 else "Miscellaneous_Archive"