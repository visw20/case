
import streamlit as st
import re
import nltk
import contractions
#import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller
import emoji
import regex 
import gensim.downloader as api
import fitz  # PyMuPDF
from nltk import pos_tag
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=UserWarning)

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

# Load the English language model
#nlp = spacy.load('en_core_web_sm')

# Set max_length to a value that accommodates your text length
#nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

# Load pre-trained Word2Vec model
w2v_model = api.load('word2vec-google-news-300')

# Initialize spell checker
spell = Speller()

# Define stopword2Vec
stop_words = set(stopwords.words('english'))
    
# Preprocessing function with lemmatization, spell checking, and NER tagging
def preprocess_text(text):
    
    # Correct spelling errors
    text = spell(text)
    
    # Remove HTML tags, URLs, and special characters
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove citations
    text = re.sub(r'\[[0-9]+\]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenization 
    tokens = word_tokenize(text)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Remove repeated characters
    tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]
    
    # Remove single characters and numeric tokens
    tokens = [word for word in tokens if len(word) > 1 and not word.isdigit()]
    
    # Handle emojis
    text = emoji.demojize(text)
    text = text.replace(":", "")
    
    # Handle emoticons
    emoticons = regex.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = regex.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
    text = text + ' '.join(emoticons)
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def calculate_similarity(sentence1, sentence2):
    # Tokenize the sentences
    tokens1 = word_tokenize(sentence1)
    tokens2 = word_tokenize(sentence2)
    
    # Filter out stopwords
    tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
    tokens2 = [word for word in tokens2 if word.lower() not in stop_words]
    
    # Get the Word2Vec vectors for each word
    vectors1 = [w2v_model[word] for word in tokens1 if word in w2v_model]
    vectors2 = [w2v_model[word] for word in tokens2 if word in w2v_model]

    # Calculate the average vectors for each sentence
    if vectors1 and vectors2:
        avg_vector1 = np.mean(vectors1, axis=0)
        avg_vector2 = np.mean(vectors2, axis=0)
        
        # Calculate the cosine similarity between the average vectors
        similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
        return similarity
    else:
        return 0.0  # Return 0 if no vectors are found or all words are OOV

# Keep track of previous questions and responses
previous_questions = []
previous_responses = []

def generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, top_k=10, similarity_threshold=0.5):
    global previous_questions, previous_responses

    bot_response = ''
    
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    
    # Check if the processed input is empty or contains only stopwords
    if not processed_input or all(word in stop_words for word in processed_input.split()):
        return "I am sorry, I don't understand."
    
    # Check if the current question is the same as a previous one
    if processed_input in previous_questions:
        index = previous_questions.index(processed_input)
        return previous_responses[index]
    
    # If not, continue with Word2Vec processing
    similarities = []
    for sent in sent_tokens:
        similarity = calculate_similarity(processed_input, sent)
        similarities.append(similarity)
    
    # Convert similarities to a NumPy array for easier processing
    similarities = np.array(similarities)
    
    # Sort the similarities in descending order
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Find the top k most similar sentences that are not in previous responses
    top_k_sentences = []
    for index in sorted_indices:
        if len(top_k_sentences) < top_k and similarities[index] >= similarity_threshold:
            top_k_sentences.append(sent_tokens[index])
    
    # Assign the top k sentences to bot_response
    if top_k_sentences:
        bot_response = '\n'.join(top_k_sentences)
        # Filter out URLs and unwanted tags from the response
        bot_response = re.sub(r'http\S+', '', bot_response)  # Remove URLs
        
        # Track previous questions and responses
        previous_questions.append(processed_input)
        previous_responses.append(bot_response)
    else:
        bot_response = "I am sorry, I don't get enough details."
    
    return bot_response  # Return the top k sentences separated by newline characters



# Generate response function
def generate_response(user_input, text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
    # Check if the user is asking for specific information
    if 'case number' in user_input or 'case no' in user_input:
        return extract_case_number(text)
    elif 'governing law' in user_input:
        return extract_governing_law(text)
    elif 'final verdict' in user_input:
        return extract_final_verdict(text)
    elif 'party' in user_input:
        return extract_parties(text)
    elif 'date' in user_input:
        return extract_date(text)
    elif 'title of the case' in user_input or 'case title' in user_input:
        return extract_case_title(text)
    elif 'summary of the case' in user_input or 'case summary' in user_input:
        return generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
    elif 'name of the court' in user_input or 'court name' in user_input:
        return extract_court_name(text)
    elif 'article' in user_input:
        return extract_articles_sections(text)
    else:
        # Handle other types of questions
        return "I'm sorry, I don't know."
    
def extract_case_number(text):
    # Regular expression pattern for matching case numbers, including ranges
    pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?(?<!\d\/)\b\d{2,}(?:-\d{2,})?(?:\s*(?:and|,)\s*\d{2,})*(?:\s*(?:of|OF)\s*|\s*\/\s*)\d{4}\b(?!\/\d{2})\b'
    
    # Find all matches of the pattern in the text
    case_numbers = re.findall(pattern, text)
    
    # Return a list of unique case numbers
    return list(set(case_numbers))

def extract_governing_law(text):
    # Define keywords for criminal law and civil law
    criminal_law_keywords = [
        'criminal', 'crime', 'offense', 'prosecution', 'defendant', 'felony', 'misdemeanor'
    ]
    civil_law_keywords = [
        'civil', 'plaintiff', 'defendant', 'contract', 'tort', 'property', 'family', 'probate'
    ]
    
    # Tokenize the text and tag the parts of speech
    tokens = word_tokenize(text.lower())
    tagged_tokens = pos_tag(tokens)
    
    # Extract nouns and adjectives from the tagged tokens
    nouns_adjectives = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]
    
    # Check for criminal law keywords
    for keyword in criminal_law_keywords:
        if keyword in nouns_adjectives:
            return "Criminal Law"
    
    # Check for civil law keywords
    for keyword in civil_law_keywords:
        if keyword in nouns_adjectives:
            return "Civil Law"
    
    return "Governing law not identified"

def extract_final_verdict(text):
    # Define regular expressions to match common patterns for final verdicts and dates
    verdict_patterns = [
        r'final\s*verdict[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
        r'judgment[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
        r'conclusion[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
        r'decision[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
        r'order[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
        r'final\s*verdict[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
        r'judgment[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
        r'conclusion[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
        r'decision[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
        r'order[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
        #r'DATED[:\s]*\d{2}\.\d{2}\.\d{4}',  # Pattern for DATED: dd.mm.yyyy
    ]
    
    # Search for the patterns in the text
    final_verdict = None
    for pattern in verdict_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            final_verdict = match.group(0).strip()
            break
    
    if not final_verdict:
        # If no final verdict date is found, try to find a date in the title of the case
        title_pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
        title_match = re.search(title_pattern, text, re.IGNORECASE)
        if title_match:
            final_verdict = title_match.group(2).strip()
    
    return final_verdict

def extract_parties(text):        
    # New patterns for petitioner and respondents
    new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
    new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

    petitioner_matches = new_petitioner_pattern.findall(text)
    respondents_matches = new_respondents_pattern.findall(text)

    if petitioner_matches and respondents_matches:
        petitioners = [match.strip() for match in petitioner_matches]
        respondents = []
        for match in respondents_matches:
            respondents_list = match.strip().split('\n')
            respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
        respondents = "\n".join([f"{respondent}" for respondent in respondents])
        return f"Petitioners:\n{petitioners}\n\nRespondents:\n{respondents}"

       
    new_pattern_7 = re.compile(
        r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
        r'\s*(?:\n*AND\n*|\s+AND\s+)'
        r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
        re.IGNORECASE | re.DOTALL
    )

    matches = new_pattern_7.findall(text)
    if matches:
        petitioners = []
        respondents = []
        for match in matches:
            petitioners.append(match[0].strip())
            respondents.append(match[1].strip())
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n".join(respondents)
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
        
    petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
    respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
    petitioners = petitioner_pattern.findall(text)
    respondents = respondent_pattern.findall(text)
    
    if petitioners and respondents:
        petitioners = [p.strip() for p in petitioners]
        respondents = [r.strip() for r in respondents]
        parties = []
        for petitioner, respondent in zip(petitioners, respondents):
            parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
        return "\n\n".join(parties)
    
    
    
    new_pattern_6 = re.compile(
        r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
        r'\s*(?:\n*AND\n*|\s+AND\s+)'
        r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
        re.IGNORECASE | re.DOTALL
    )

    matches = new_pattern_6.findall(text)
    if matches:
        petitioners = []
        respondents = []
        for match in matches:
            petitioners.append(match[0].strip())
            respondents.append(match[1].strip())
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n".join(respondents)
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
    
    new_pattern_2_revised = re.compile(
        r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
        re.IGNORECASE | re.DOTALL
    )

    matches = new_pattern_2_revised.findall(text)
    if matches:
        petitioners = []
        respondents = []
        for match in matches:
            petitioners.append(match[0].strip())
            respondents.append(match[1].strip())
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n".join(respondents)
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    specific_pattern = re.compile(
        r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
        re.IGNORECASE | re.DOTALL
    )

    matches = specific_pattern.findall(text)
    if matches:
        petitioners = []
        respondents = []
        for match in matches:
            petitioners.append(match[0].strip())
            respondents_text = match[1].strip()
            respondents_list = respondents_text.split('\n')
            respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n".join(respondents)
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    new_pattern = re.compile(
    r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
    re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = new_pattern.findall(text)
    if matches:
        petitioners = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n\n".join(respondents_list)
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
   
    new_pattern_2 = re.compile(
        r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
        re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = new_pattern_2.findall(text)
    if matches:
        petitioners = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n\n".join(respondents_list)
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    new_pattern_3 = re.compile(
        r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
        re.IGNORECASE | re.DOTALL
    )

    matches = new_pattern_3.findall(text)
    if matches:
        petitioners = []
        respondents = []
        for match in matches:
            petitioner = match[0].strip()
            respondents_text = match[1].strip()
            respondents_list = respondents_text.split('\n')
            respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n".join(respondents)
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
    
    new_pattern_4 = re.compile(
        r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
        re.IGNORECASE | re.DOTALL
    )

    matches = new_pattern_4.findall(text)
    if matches:
        petitioners = []
        respondents = []
        for match in matches:
            petitioners.append(match[0].strip())
            respondents.append(match[1].strip())
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n".join(respondents)
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
    additional_new_pattern = re.compile(
        r'\[.*?\]\s*(.*?)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = additional_new_pattern.findall(text)
    
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            
            # Extract only the relevant petitioner information and remove leading spaces
            petitioner_parts = petitioner.split('\n')
            relevant_petitioner = '\n'.join(part.strip() for part in petitioner_parts if '.' not in part[:3])
            
            petitioners_list.append(relevant_petitioner)
            respondents_list.append("\n".join(respondents))
        
        petitioners_text = "\n\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    

  
    new_pattern_5 = re.compile(
        r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
        re.IGNORECASE | re.DOTALL
    )

    matches = new_pattern_5.findall(text)
    if matches:
        petitioners = []
        respondents = []
        for match in matches:
            petitioners.append(match[0].strip())
            respondents.append(match[1].strip())
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n".join(respondents)
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
        
    pattern_11_1 = re.compile(
        r'\[.*?\]\s*(.*?)\s*\.{3,}\s*Applicant\s*.*?Versus\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_11_1.findall(text)
    if matches:
        parties = []
        for match in matches:
            # Remove leading/trailing spaces and align to left
            applicant_lines = [line.strip() for line in match[0].split('\n') if line.strip()]
            applicant = '\n'.join(applicant_lines)
            
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            parties.append(f"Applicant:\n{applicant}\n\nRespondents:\n{respondents_text}")
        return "\n\n".join(parties)
    
    
    pattern_11 = re.compile(
        r'([^\n\r]+)\s*\.{3,}\s*Applicant\s*.*?VERSUS\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_11.findall(text)
    if matches:
        parties = []
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            parties.append(f"Petitioner:{petitioner}\n\nRespondents:{respondents_text}")
        return "\n\n".join(parties)
    
    
    
    pattern_12 = re.compile(
        r'([^\n\r]+)\s*\.{1,}\s*Appellant\s*.*?Versus\s*([^\n\r]+)\s*\.{1,}\s*Respondent',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_12.findall(text)
    if matches:
        parties = []
        for match in matches:
            petitioner = match[0].strip()
            respondent = match[1].strip()
            parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
        return "\n\n".join(parties)
    
    pattern_17 = re.compile(
        r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
        re.IGNORECASE
    )
    
    matches = pattern_17.findall(text)
    if matches:
        parties = []
        for match in matches:
            petitioner = match[0].strip()
            respondent = match[1].strip()
            parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
        return "\n\n".join(parties)
    
    pattern_19 = re.compile(
    r'([A-Z][A-Z\s&]+\.)\s*…\s*PETITIONERS\s*VERSUS\s*([A-Z][A-Z\s&]+\.)\s*…\s*RESPONDENTS',
    re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = pattern_19.findall(text)
    if matches:
        petitioners = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n\n".join(respondents_list)
        return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
    pattern_20 = re.compile(
    r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
    re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = pattern_20.findall(text)
    if matches:
        appellants_list = []
        respondents_list = []
        
        for match in matches:
            appellants = match[0].strip()
            respondents = match[1].strip()
            
            appellants_list.append(appellants)
            respondents_list.append(respondents)
        
        appellants_text = "\n".join(appellants_list)
        respondents_text = "\n".join(respondents_list)
        return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents}"
    
    pattern_21 = re.compile(
    r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
    re.IGNORECASE
    )
    
    # Find all matches in the text
    matches = pattern_21.findall(text)
    if matches:
        appellants_list = []
        respondents_list = []
        
        for match in matches:
            appellants = match[0].strip()
            respondents = match[1].strip()
            
            appellants_list.append(appellants)
            respondents_list.append(respondents)
        
        appellants_text = "\n".join(appellants_list)
        respondents_text = "\n".join(respondents_list)
        return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
    
    pattern_appellant = re.compile(
        r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_appellant.findall(text)
    
    if matches:
        parties = []
        for petitioner, respondent in matches:
            parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
        return "\n\n".join(parties)
    
    pattern_23 = re.compile(
    r'(?:\d+\s+of\s+\d+\s+)?(?P<petitioner>[\w\s]+?)\s*…\s*Petitioner\s+versus\s+(?P<respondent>[\w\s]+?\s+and\s+others)\s*…\s*Respondents',
    re.IGNORECASE
    )
    
    matches = pattern_23.findall(text)
    
    if matches:
        seen = set()
        results = []
        for match in matches:
            petitioner, respondent = match
            # Remove all numbers and 'of' from petitioner name
            petitioner = re.sub(r'\d+\s+of\s+\d+\s+', '', petitioner).strip()
            # Remove any leading/trailing whitespace and newlines
            petitioner = ' '.join(petitioner.split())
            if petitioner not in seen:
                seen.add(petitioner)
                results.append(f"Petitioner: {petitioner}\nRespondent: {respondent.strip()}")
        
        return "\n\n".join(results)

    pattern_10_1 = re.compile(
        r'((?:\d+\.[^\n\r]+\s*)+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_10_1.findall(text)
    
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip().split('\n')
            petitioner = [pet.strip() for pet in petitioner if pet.strip()]
            petitioner_text = "\n".join(petitioner)
            
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners_list.append(petitioner_text)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    pattern_10 = re.compile(
    r'([^\n\r]+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
    re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = pattern_10.findall(text)
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners_list.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
        
    pattern_13 = re.compile(
    r'([^\n\r]+)\s*\.{2,}\s*Petitione\s*.*?vs\s*([\s\S]*?)\s*\.{2,}\s*Respondents',
    re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = pattern_13.findall(text)
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners_list.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    pattern_14 = re.compile(
        r'(?:.*?WP\(C\)\sNo\.\s\d+\sof\s\d+\s+Date\sof\sDecision:.*?\n)(.*?)\s*:::\s*Petitioner\s*-\s*Vs\s*-\s*([\s\S]+?)\s*:::\s*Respondents',
        re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = pattern_14.findall(text)
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            # Split petitioner info into lines, strip each line, and rejoin
            petitioner_lines = match[0].strip().split('\n')
            petitioner = '\n'.join(line.strip() for line in petitioner_lines if line.strip())
            
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners_list.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    pattern_15_3 = re.compile(
        r'((?:\d+\.\s?[^\n\r]+\s*)+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:versus)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_15_3.findall(text)
    
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            petitioners = petitioner.split('\n')
            petitioners = [pet.strip() for pet in petitioners if pet.strip()]
            petitioners_text = "\n".join(petitioners)

            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners_list.append(petitioners_text)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
    pattern_15_2 = re.compile(
        r'Between:\s*(.*?)\s*\.{3,}\s*APPELLANT(?:\(S\))?\s*(?:AND)?\s*(.*?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_15_2.findall(text)
    
    if matches:
        appellants_list = []
        respondents_list = []
        
        for match in matches:
            appellants = match[0].strip().split('\n')
            respondents = match[1].strip().split('\n')
            
            appellants = [app.strip() for app in appellants if app.strip()]
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            
            appellants_text = "\n".join(appellants)
            respondents_text = "\n".join(respondents)
            
            appellants_list.append(appellants_text)
            respondents_list.append(respondents_text)
        
        appellants_text = "\n".join(appellants_list)
        respondents_text = "\n".join(respondents_list)
        
        return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents_text}"
    
    
    pattern_15_1 = re.compile(
        r'\b(?:Between:|Petitioner:|Petitioners:)\s*(.*?)\s*\.\.\.\s*Petitioner\s*(?:Versus|AND|Vs\.|v\.)\s*(.*?)\s*\.\.\.\s*(?:Respondents?|Respondent)',
        re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = pattern_15_1.findall(text)
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners_list.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        # Ensure no unnecessary leading or trailing whitespace in the final output
        petitioners_text = "\n".join([line.strip() for line in petitioners_text.split('\n')])
        respondents_text = "\n".join([line.strip() for line in respondents_text.split('\n')])
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    pattern_15_4 = re.compile(
        r'Between:\s*(.*?)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_15_4.findall(text)
    
    if matches:
        parties = []
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            parties.append(f"Petitioner:{petitioner}\n\nRespondents:{respondents_text}")
        return "\n\n".join(parties)
   
    
    pattern_15 = re.compile(
        r'([^\n\r]+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_15.findall(text)
    
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners_list.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    pattern_16 = re.compile(
    r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*([\s\S]+?)\s*\.\.\s*Responden',
    re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = pattern_16.findall(text)
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners_list.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    pattern_18 = re.compile(
    r'((?:\d+\.[A-Z][.\w\s]+\s*)+)\.\.Pet.*?vs\s+((?:\d+\..*?\.\s*)+)\.\.\s*Respo',
    re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches in the text
    matches = pattern_18.findall(text)
    if matches:
        petitioners_list = []
        respondents_list = []
        
        for match in matches:
            petitioner = match[0].strip()
            respondents = match[1].strip().split('\n')
            respondents = [resp.strip() for resp in respondents if resp.strip()]
            respondents_text = "\n".join(respondents)
            
            petitioners_list.append(petitioner)
            respondents_list.append(respondents_text)
        
        petitioners_text = "\n".join(petitioners_list)
        respondents_text = "\n\n".join(respondents_list)
        
        return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    pattern_22 = re.compile(
        r'BETWEEN:-\s*(?P<petitioner>.*?)\s*\(BY.*?\)\s*AND\s*(?P<respondents>(?:\d+\..*?)+)(?:\(BY|$)',
        re.DOTALL | re.IGNORECASE
    )
    
    match = pattern_22.search(text)
    if match:
        petitioner = match.group('petitioner').strip()
        respondents_text = match.group('respondents').strip()
        
        # Format the petitioner
        petitioner_formatted = "\n".join(line.strip() for line in petitioner.split('\n'))
        
        # Format the respondents
        respondents = re.findall(r'\d+\.(.*?)(?=\d+\.|\Z)', respondents_text, re.DOTALL)
        respondents_formatted = "\n".join(f"{i+1}. {' '.join(line.strip() for line in resp.split())}" 
                                          for i, resp in enumerate(respondents))
        
        return f"Petitioner:\n{petitioner_formatted}\n\nRespondents:\n{respondents_formatted}"
    
    specific_pattern_1 = re.compile(
        r'([^\n\r]+?)\s*\.\.\.\s*Pe\s*versus\s*([\s\S]+?)\s*\.\.\.\s*Re', 
        re.IGNORECASE | re.DOTALL
    )

    matches = specific_pattern_1.findall(text)
    if matches:
        petitioners = []
        respondents = []
        for match in matches:
            petitioners.append(match[0].strip())
            respondents_text = match[1].strip()
            respondents_list = respondents_text.split('\n')
            respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
        petitioners_text = "\n".join(petitioners)
        respondents_text = "\n".join(respondents)
        return f"Parties:\nPetitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#2    
    pattern_ellipsis = re.compile(
        r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', 
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_ellipsis.findall(text)
    
    if matches:
        parties = []
        for petitioner, respondent in matches:
            parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
        return "\n\n".join(parties)
    
    pattern_dots = re.compile(
        r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_dots.findall(text)
    
    if matches:
        parties = []
        for petitioner, respondent in matches:
            parties.append(f"Petitioner: {petitioner.strip()}\n\nRespondent: {respondent.strip()}")
        return "\n\n".join(parties)
    
    
    
    pattern_dashes = re.compile(
        r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_dashes.findall(text)
    
    if matches:
        parties = []
        for petitioner, respondent in matches:
            parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
        return "\n\n".join(parties)
    
    
    pattern_provided = re.compile(
        r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
        re.IGNORECASE | re.DOTALL
    )
    
    matches = pattern_provided.findall(text)
    
    if matches:
        parties = []
        for petitioner, respondent in matches:
            parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
        return "\n\n".join(parties)
    
    
    pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
    matches = pattern_specific_parties.findall(text)
    
    if matches:
        parties = []
        for petitioner, respondent in matches:
            parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
        return "\n\n".join(parties)
           
    pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    matches = pattern_specific_parties_1.findall(text) 
    if matches:
        parties = []
        for petitioner, respondent in matches:
            petitioner = petitioner.strip()
            respondent = respondent.strip()
            # Check if both petitioner and respondent are in uppercase
            if petitioner.isupper() and respondent.isupper():
                parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
        # Join and return the result
        return "\n\n".join(parties)
         
    return "Parties not found."

def extract_date(text):
    # Define regex pattern to match dates in various formats
    date_pattern = (
        r'(\b(?:[12][0-9]|3[01]|0?[1-9])(st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s\d{4}\b)|'
        r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s(?:[12][0-9]|3[01]),?\s\d{4}\b)|'
        r'(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b)|'
        r'(\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}\b)'
    )
    
    # Find all matches of the date pattern in the text
    matches = re.findall(date_pattern, text)
    
    # Flatten the list of tuples and remove empty strings
    dates = [date for match in matches for date in match if date]
    
    # Remove duplicates by converting to set and back to list
    unique_dates = list(set(dates))
    
    return unique_dates

def extract_case_title(text):
    # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
    pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/0-9@]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/0-9]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/0-9@]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
    
    # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
    pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-0-9@]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-0-9@]+))'
    
    # First try matching the original pattern
    match = re.search(pattern, text, re.IGNORECASE)
    
    # If no match, try the alternative pattern
    if not match:
        match = re.search(pattern_alt, text, re.IGNORECASE)
    
    if match:
        title = match.group(1).strip()
        if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
            date = match.group(2).strip()
            return f"{title} on {date}"
        else:
            # Check for unwanted "Author" or other text in the title
            if "Author" in title:
                title = title.split("Author")[0].strip()
            return title
    else:
        return "Title and date not found"

def extract_court_name(text):
    # Define a comprehensive pattern for court names, including spaces between letters
    comprehensive_pattern = (
        r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T\s*'
        r'O\s*F\s*I\s*N\s*D\s*I\s*A|'
        r'Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
        r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
    )
    
    # Search for the comprehensive pattern in the text
    match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(0).strip()
    else:
        # Define a fallback pattern for court names, including spaces between letters
        fallback_pattern = r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T|Supreme|High|District) Court'
        
        # Search for the fallback pattern in the text
        match = re.search(fallback_pattern, text, re.IGNORECASE)
        
        if match:
            return match.group(0).strip()
        else:
            return "Court name not found"



def extract_articles_sections(text):
    # Comprehensive pattern for articles
    article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
    # Comprehensive pattern for sections (including plural 'Sections')
    section_pattern = re.compile(r'\b(?:Section|Sec\.?|Sections|Secs\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
    # Pattern for clauses
    clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
    # Pattern for sub-sections
    subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

    unique_references = set()  # Using a set to remove duplicates

    # Function to process matches
    def process_matches(pattern, prefix):
        for match in pattern.finditer(text):
            reference = match.group().strip()
            if prefix not in reference.lower():
                reference = f"{prefix} {reference}"
            unique_references.add(reference)

    # Processing all patterns
    process_matches(article_pattern, "Article :")
    process_matches(section_pattern, "Section :")
    process_matches(clause_pattern, "Clause :")
    process_matches(subsection_pattern, "Sub-section :")

    if unique_references:
        return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
    else:
        return "No articles, sections, clauses, or sub-sections found." 
    
def sanitize_text(text):
    # Remove unwanted symbols using regular expressions
    sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)
    return sanitized_text



def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    raw_text = ''
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        raw_text += page.get_text()
    pdf_document.close()
    return raw_text

def main():
    st.title("Legal Case Summarization")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Read and process the PDF
        raw_text = read_pdf(uploaded_file)
        
        # Preprocess the text
        sent_tokens = nltk.sent_tokenize(raw_text)
        preprocessed_sent_tokens = [preprocess_text(sent) for sent in sent_tokens]
        
        # Create TF-IDF vectors
        word_vectorizer = TfidfVectorizer(
            tokenizer=word_tokenize,
            stop_words=stopwords.words('english'),
            ngram_range=(1, 3),
            max_features=15000,
            token_pattern=r'\b\w+\b',
            sublinear_tf=True,
            smooth_idf=True,
            norm='l2'
        )
        word_vectors = word_vectorizer.fit_transform(preprocessed_sent_tokens)
        
  
        # Display information
        st.subheader("Case Information")
        st.write("Case Number:", extract_case_number(raw_text))
        st.write("Governing Law:", extract_governing_law(raw_text))
        st.write("Final Verdict:", extract_final_verdict(raw_text))
        st.write("Parties:", extract_parties(raw_text))
        st.write("Date:", extract_date(raw_text))
        st.write("Title of the Case:", extract_case_title(raw_text))
        st.write("Name of the Court:", extract_court_name(raw_text))
        st.write("Articles:", extract_articles_sections(raw_text))

        # Case summary
        st.subheader("Case Summary")
        summary = generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
        st.write(summary)

if __name__ == "__main__":
    main()
    


