import nltk
nltk.download('punkt')
from nrclex import NRCLex
import spacy
from textblob import TextBlob
import os
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from collections import Counter

# Load the English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path, 300)
    overall_text = ""
    for page in pages:
        text = pytesseract.image_to_string(page, lang='eng', config='--psm 6')
        overall_text += text
    return overall_text

# Analyze a single motivational letter
def analyze_letter(text):
    # Perform sentiment analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Perform emotion detection
    emotion = NRCLex(text)
    emotions = emotion.raw_emotion_scores
    
    # Process text with spaCy for NLP
    doc = nlp(text)
    
    # Extract keywords
    keywords = [
        token.lemma_.lower() for token in doc 
        if token.text.lower() in {'research', 'internship', 'leadership', 'achievements', 'experience', 
                                  'enthusiastic', 'project', 'skill', 'motivation', 'goal', 'team' }
    ]
    
    # Entity Recognition
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    
    return polarity, subjectivity, emotions, keywords, entities

# Criteria for rating the motivational letter
def rate_letter(polarity, subjectivity, keyword_count, emotion_scores):
    positive_emotions = emotion_scores.get('positive', 0)
    negative_emotions = emotion_scores.get('negative', 0)

    if polarity >= 0.3 and subjectivity < 0.5 and keyword_count >= 8 and positive_emotions > negative_emotions:
        return 5, "Excellent"
    elif polarity >= 0.2 and subjectivity < 0.6 and keyword_count >= 6 and positive_emotions > negative_emotions:
        return 4, "Very Good"
    elif polarity >= 0.1 and subjectivity < 0.7 and keyword_count >= 4 and positive_emotions > negative_emotions:
        return 3, "Good"
    elif polarity > 0 and subjectivity < 0.8 and keyword_count >= 2 and positive_emotions > negative_emotions:
        return 2, "Satisfactory"
    else:
        return 1, "Poor"

# Open a file dialog to select multiple PDF files
root = Tk()
root.withdraw()  # Hide the main window
file_paths = askopenfilenames(filetypes=[("PDF files", "*.pdf")], title="Select Motivational Letters")

# Lists to store the results
letter_names = []
all_polarity_scores = []
all_subjectivity_scores = []
all_emotion_scores = []
all_ratings = []
ai_keywords = []
human_keywords = []
ai_entities = []
human_entities = []

# Read and analyze each letter
for filepath in file_paths:
    filename = os.path.basename(filepath)
    letter_names.append(filename[0])  # Store the letter name
    text = extract_text_from_pdf(filepath)
    polarity, subjectivity, emotions, keywords, entities = analyze_letter(text)
    
    all_polarity_scores.append(polarity)
    all_subjectivity_scores.append(subjectivity)
    all_emotion_scores.append(emotions)
    
    # Determine if the letter is AI or Human generated
    if 'A' <= filename[0] <= 'J':  # AI generated
        ai_keywords.extend(keywords)
        ai_entities.extend(entities)
    elif 'K' <= filename[0]<= 'T':  # Human generated
        human_keywords.extend(keywords)
        human_entities.extend(entities)
    
    # Rate each letter
    rating, description = rate_letter(polarity, subjectivity, len(keywords), emotions)
    all_ratings.append((rating, description))

# Create output directory if not exists
output_folder = os.path.join(os.getcwd(), "Output")
os.makedirs(output_folder, exist_ok=True)

# Save the overall outcomes to text files
for idx, filepath in enumerate(file_paths):
    file_name = os.path.basename(filepath)
    overall_outcome_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_overall_outcome.txt")
    with open(overall_outcome_path, 'w', encoding='utf-8') as outcome_file:
        outcome_file.write("Letter: {}\n".format(file_name))
        outcome_file.write("Overall Keywords:\n{}\n".format(ai_keywords if 'A' <= file_name[0] <= 'J' else human_keywords))
        outcome_file.write("Entities:\n{}\n".format(ai_entities if 'A' <= file_name[0] <= 'J' else human_entities))
        outcome_file.write("Overall Sentiment:\n{}\n".format({'polarity': all_polarity_scores[idx], 'subjectivity': all_subjectivity_scores[idx]}))
        outcome_file.write("Rating (1 to 5):\n{}\n".format(all_ratings[idx][0]))
        outcome_file.write("Rating Description:\n{}\n".format(all_ratings[idx][1]))
        outcome_file.write("\nEmotion Scores:\n{}\n".format(all_emotion_scores[idx]))
    print("Overall outcome saved:", overall_outcome_path)

# Function to calculate average emotion scores
def calculate_average_emotions(emotion_list):
    all_emotions = Counter()
    for emotions in emotion_list:
        all_emotions.update(emotions)
    average_emotions = {emotion: count / len(emotion_list) for emotion, count in all_emotions.items()}
    return average_emotions

# Calculate average emotion scores
average_ai_emotions = calculate_average_emotions([emotions for i, emotions in enumerate(all_emotion_scores) if 'A' <= letter_names[i] <= 'J'])
average_human_emotions = calculate_average_emotions([emotions for i, emotions in enumerate(all_emotion_scores) if 'K' <= letter_names[i] <= 'T'])

# Create dataframes for plotting
data = pd.DataFrame({'Letter': letter_names, 'Polarity': all_polarity_scores, 'Subjectivity': all_subjectivity_scores})

# Identify AI and Human letters for color grouping
colors = ['blue' if 'A' <= name <= 'J' else 'green' for name in letter_names]

# 1. Graph of subjectivity of all motivational letters
plt.figure(figsize=(12, 6))
sns.barplot(x=data['Letter'], y=data['Subjectivity'], palette=colors)
plt.xlabel('Letter Name')
plt.ylabel('Subjectivity')
plt.title('Subjectivity Scores for Individual Letters')
plt.xticks(rotation=360)
plt.legend(handles=[plt.Rectangle((0,0),1,1, color="blue",ec="k"), plt.Rectangle((0,0),1,1, color="green",ec="k")], labels=['AI-generated', 'Human-written'])
plt.show()

# 2. Graph of polarity of all motivational letters
plt.figure(figsize=(12, 6))
sns.barplot(x=data['Letter'], y=data['Polarity'], palette=colors)
plt.xlabel('Letter Name')
plt.ylabel('Polarity')
plt.title('Polarity Scores for Individual Letters')
plt.xticks(rotation=360)
plt.legend(handles=[plt.Rectangle((0,0),1,1, color="blue",ec="k"), plt.Rectangle((0,0),1,1, color="green",ec="k")], labels=['AI-generated', 'Human-written'])
plt.show()

# 3. Combined average subjectivity and polarity scores for all letters
avg_subjectivity = sum(all_subjectivity_scores) / len(all_subjectivity_scores)
avg_polarity = sum(all_polarity_scores) / len(all_polarity_scores)
plt.figure(figsize=(10, 6))
plt.bar(['Average Subjectivity', 'Average Polarity'], [avg_subjectivity, avg_polarity], color=['orange', 'blue'])
plt.title('Combined Average Subjectivity and Polarity Scores of All Letters')
plt.show()

# 4. Comparison of AI-generated vs. human-generated letters for average polarity and subjectivity
avg_ai_subjectivity = sum([s for i, s in enumerate(all_subjectivity_scores) if 'A' <= letter_names[i] <= 'J']) / len([s for i, s in enumerate(all_subjectivity_scores) if 'A' <= letter_names[i] <= 'J'])
avg_ai_polarity = sum([p for i, p in enumerate(all_polarity_scores) if 'A' <= letter_names[i] <= 'J']) / len([p for i, p in enumerate(all_polarity_scores) if 'A' <= letter_names[i] <= 'J'])
avg_human_subjectivity = sum([s for i, s in enumerate(all_subjectivity_scores) if 'K' <= letter_names[i] <= 'T']) / len([s for i, s in enumerate(all_subjectivity_scores) if 'K' <= letter_names[i] <= 'T'])
avg_human_polarity = sum([p for i, p in enumerate(all_polarity_scores) if 'K' <= letter_names[i] <= 'T']) / len([p for i, p in enumerate(all_polarity_scores) if 'K' <= letter_names[i] <= 'T'])

plt.figure(figsize=(10, 6))
plt.bar(['AI Subjectivity', 'AI Polarity', 'Human Subjectivity', 'Human Polarity'],
        [avg_ai_subjectivity, avg_ai_polarity, avg_human_subjectivity, avg_human_polarity],
        color=['lightblue', 'cyan', 'lightgreen', 'green'])
plt.title('Average Polarity and Subjectivity: AI vs Human')
plt.show()

# 5. Emotion detection for all letters in one graph
emotion_data = pd.DataFrame(all_emotion_scores)
emotion_data.index = letter_names
emotion_data.plot(kind='bar', stacked=True, figsize=(14, 7), cmap='tab20')
plt.title('Emotion Scores for All Letters')
plt.xlabel('Letter')
plt.ylabel('Emotion Score')
plt.xticks(rotation=360)
plt.legend(loc='upper right')
plt.show()

# 6. Comparison of positive and negative scores
avg_ai_positive = average_ai_emotions.get('positive', 0)
avg_ai_negative = average_ai_emotions.get('negative', 0)
avg_human_positive = average_human_emotions.get('positive', 0)
avg_human_negative = average_human_emotions.get('negative', 0)

plt.figure(figsize=(10, 6))
plt.barh(['AI Positive', 'AI Negative', 'Human Positive', 'Human Negative'],
        [avg_ai_positive, avg_ai_negative, avg_human_positive, avg_human_negative],
        color=['blue', 'red', 'green', 'orange'])
plt.title('Comparison of Positive and Negative Scores: AI vs Human')
plt.xlabel('Average Score')
plt.show()

# Combined graph of the highest emotion score excluding positive/negative emotions

# Recalculate overall emotion scores for clarity
combined_emotions = Counter()
for emotion_scores in all_emotion_scores:
    combined_emotions.update(emotion_scores)

# Filter out positive and negative emotions
non_pn_emotions = {emotion: count for emotion, count in combined_emotions.items() if emotion not in ['positive', 'negative']}
highest_emotion_overall = max(non_pn_emotions, key=non_pn_emotions.get, default=None) if non_pn_emotions else None

ai_non_pn_emotions = {emotion: count for emotion, count in average_ai_emotions.items() if emotion not in ['positive', 'negative']}
highest_ai_emotion = max(ai_non_pn_emotions, key=ai_non_pn_emotions.get, default=None) if ai_non_pn_emotions else None

human_non_pn_emotions = {emotion: count for emotion, count in average_human_emotions.items() if emotion not in ['positive', 'negative']}
highest_human_emotion = max(human_non_pn_emotions, key=human_non_pn_emotions.get, default=None) if human_non_pn_emotions else None

if highest_emotion_overall and highest_ai_emotion and highest_human_emotion:
    plt.figure(figsize=(10, 6))
    plt.bar(['Overall Highest (' + highest_emotion_overall + ')', 
             'AI: ' + highest_ai_emotion, 
             'Human: ' + highest_human_emotion],
            [non_pn_emotions[highest_emotion_overall], 
             ai_non_pn_emotions[highest_ai_emotion], 
             human_non_pn_emotions[highest_human_emotion]],
            color=['purple', 'blue', 'green'])
    plt.title('Overall Highest Emotion Score and AI vs Human (Excluding Positive/Negative)')
    plt.ylabel('Frequency')
    plt.show()

# Graph of the overall rating for each letter
ratings_data = pd.DataFrame(all_ratings, columns=['Rating', 'Description'], index=letter_names)
colors_ratings = ['blue' if 'A' <= name <= 'J' else 'green' for name in letter_names]
ratings_data['Rating'].plot(kind='bar', figsize=(10, 6), color=colors_ratings, legend=False)
plt.title('Overall Rating for Each Letter')
plt.xlabel('Letter')
plt.ylabel('Rating (1 to 5)')
plt.xticks(rotation=360)
plt.ylim(0, 6)
plt.legend(handles=[plt.Rectangle((0,0),1,1, color="blue",ec="k"), plt.Rectangle((0,0),1,1, color="green",ec="k")], labels=['AI-generated', 'Human-written'])
plt.show()

# Output extracted keywords and entities
print("AI Generated Keywords:", ai_keywords)
print("Human Generated Keywords:", human_keywords)
print("AI Generated Entities:", ai_entities)
print("Human Generated Entities:", human_entities)
