# import re
# import nltk
# import contractions
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from autocorrect import Speller
# import emoji
# import regex 
# import gensim.downloader as api
# import fitz  # PyMuPDF
# from nltk import pos_tag
# import warnings
# import numpy as np

# warnings.simplefilter(action='ignore', category=UserWarning)

# #nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Load the English language model
# nlp = spacy.load('en_core_web_sm')

# # Set max_length to a value that accommodates your text length
# nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

# # Load pre-trained Word2Vec model
# w2v_model = api.load('word2vec-google-news-300')

# # Initialize spell checker
# spell = Speller()

# # Define stopword2Vec
# stop_words = set(stopwords.words('english'))


# # Preprocessing function with lemmatization, spell checking, and NER tagging
# def preprocess_text(text):
    
#     # Correct spelling errors
#     text = spell(text)
    
#     # Remove HTML tags, URLs, and special characters
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Expand contractions
#     text = contractions.fix(text)
    
#     # Remove citations
#     text = re.sub(r'\[[0-9]+\]', '', text)
    
#     # Remove extra whitespaces
#     text = re.sub(r'\s+', ' ', text)
    
#     # Tokenization and NER tagging using spaCy
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if token.ent_type_:
#             tokens.append(token.ent_type_)
#         else:
#             tokens.append(token.text)
    
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Remove repeated characters
#     tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]
    
#     # Remove single characters and numeric tokens
#     tokens = [word for word in tokens if len(word) > 1 and not word.isdigit()]
    
#     # Handle emojis
#     text = emoji.demojize(text)
#     text = text.replace(":", "")
    
#     # Handle emoticons
#     emoticons = regex.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = regex.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
#     text = text + ' '.join(emoticons)
    
#     # Join tokens back into a string
#     preprocessed_text = ' '.join(tokens)
    
#     return preprocessed_text

# def calculate_similarity(sentence1, sentence2):
#     # Tokenize the sentences
#     tokens1 = word_tokenize(sentence1)
#     tokens2 = word_tokenize(sentence2)
    
#     # Filter out stopwords
#     tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
#     tokens2 = [word for word in tokens2 if word.lower() not in stop_words]
    
#     # Get the Word2Vec vectors for each word
#     vectors1 = [w2v_model[word] for word in tokens1 if word in w2v_model]
#     vectors2 = [w2v_model[word] for word in tokens2 if word in w2v_model]

#     # Calculate the average vectors for each sentence
#     if vectors1 and vectors2:
#         avg_vector1 = np.mean(vectors1, axis=0)
#         avg_vector2 = np.mean(vectors2, axis=0)
        
#         # Calculate the cosine similarity between the average vectors
#         similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
#         return similarity
#     else:
#         return 0.0  # Return 0 if no vectors are found or all words are OOV

# # Keep track of previous questions and responses
# previous_questions = []
# previous_responses = []

# def generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, top_k=15, similarity_threshold=0.5):
#     global previous_questions, previous_responses

#     bot_response = ''
    
#     # Preprocess user input
#     processed_input = preprocess_text(user_input)
    
#     # Check if the processed input is empty or contains only stopwords
#     if not processed_input or all(word in stop_words for word in processed_input.split()):
#         return "I am sorry, I don't understand."
    
#     # Check if the current question is the same as a previous one
#     if processed_input in previous_questions:
#         index = previous_questions.index(processed_input)
#         return previous_responses[index]
    
#     # If not, continue with Word2Vec processing
#     similarities = []
#     for sent in sent_tokens:
#         similarity = calculate_similarity(processed_input, sent)
#         similarities.append(similarity)
    
#     # Convert similarities to a NumPy array for easier processing
#     similarities = np.array(similarities)
    
#     # Sort the similarities in descending order
#     sorted_indices = np.argsort(similarities)[::-1]
    
#     # Find the top k most similar sentences that are not in previous responses
#     top_k_sentences = []
#     for index in sorted_indices:
#         if len(top_k_sentences) < top_k and similarities[index] >= similarity_threshold:
#             top_k_sentences.append(sent_tokens[index])
    
#     # Assign the top k sentences to bot_response
#     if top_k_sentences:
#         bot_response = '\n'.join(top_k_sentences)
#         # Filter out URLs and unwanted tags from the response
#         bot_response = re.sub(r'http\S+', '', bot_response)  # Remove URLs
        
#         # Track previous questions and responses
#         previous_questions.append(processed_input)
#         previous_responses.append(bot_response)
#     else:
#         bot_response = "I am sorry, I don't get enough details."
    
#     return bot_response  # Return the top k sentences separated by newline characters

# # Read text from the local PDF file using PyMuPDF
# file_path = 'C:/Users/Viswajith/Downloads/Arulmigu_Moolathuvazhiamman_vs_The_State_Of_Tamil_Nadu_on_12_September_2023.PDF'
# try:
#     pdf_document = fitz.open(file_path)
#     raw_text = ''
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         raw_text += page.get_text()
#     pdf_document.close()
#     print("PDF file read successfully.")
# except Exception as e:
#     print("Error reading PDF file:", e)
#     exit()

# # Tokenize the raw text to obtain sentence tokens
# sent_tokens = nltk.sent_tokenize(raw_text)

# # Preprocess the text
# preprocessed_sent_tokens = [preprocess_text(sent) for sent in sent_tokens]

# # Define TF-IDF vectorizer with optimized parameters and TF-IDF weighting schemes
# word_vectorizer = TfidfVectorizer(
#     tokenizer=word_tokenize,     # Use NLTK's word tokenizer
#     stop_words='english',        # Use English stopwords
#     ngram_range=(1, 3),          # Use unigrams and bigrams
#     max_features=15000,           # Limit the vocabulary size to the top 5000 features
#     token_pattern=r'\b\w+\b',    # Use words as tokens
#     sublinear_tf=True,           # Apply sublinear term frequency scaling
#     smooth_idf=True,             # Smooth IDF weights by adding one to document frequencies
#     norm='l2'                    # Normalize TF-IDF vectors to unit length
# )

# # Apply TF-IDF vectorization on the preprocessed text
# word_vectors = word_vectorizer.fit_transform(preprocessed_sent_tokens)


# # Generate response function
# def generate_response(user_input, text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     # Check if the user is asking for specific information
#     if 'case number' in user_input or 'case no' in user_input:
#         return extract_case_number(text)
#     elif 'governing law' in user_input:
#         return extract_governing_law(text)
#     elif 'final verdict' in user_input:
#         return extract_final_verdict(text)
#     elif 'party' in user_input:
#         return extract_parties(text)
#     elif 'date' in user_input:
#         return extract_date(text)
#     elif 'title of the case' in user_input or 'case title' in user_input:
#         return extract_case_title(text)
#     elif 'summary of the case' in user_input or 'case summary' in user_input:
#         return generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
#     elif 'name of the court' in user_input or 'court name' in user_input:
#         return extract_court_name(text)
#     elif 'article' in user_input:
#         return extract_articles_sections(text)
#     else:
#         # Handle other types of questions
#         return "I'm sorry, I don't know."
    
# def extract_case_number(text):
#     # Regular expression pattern for matching case numbers
#     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?\d{1,}\s*(?:of|OF)\s*\d{4}\b'
    
#     # Find all matches of the pattern in the text
#     case_numbers = re.findall(pattern, text)
    
#     # Return a list of unique case numbers
#     return list(set(case_numbers))

# def extract_governing_law(text):
#     # Define keywords for criminal law and civil law
#     criminal_law_keywords = [
#         'criminal', 'crime', 'offense', 'prosecution', 'defendant', 'felony', 'misdemeanor'
#     ]
#     civil_law_keywords = [
#         'civil', 'plaintiff', 'defendant', 'contract', 'tort', 'property', 'family', 'probate'
#     ]
    
#     # Tokenize the text and tag the parts of speech
#     tokens = word_tokenize(text.lower())
#     tagged_tokens = pos_tag(tokens)
    
#     # Extract nouns and adjectives from the tagged tokens
#     nouns_adjectives = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]
    
#     # Check for criminal law keywords
#     for keyword in criminal_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Criminal Law"
    
#     # Check for civil law keywords
#     for keyword in civil_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Civil Law"
    
#     return "Governing law not identified"

# def extract_final_verdict(text):
#     # Define regular expressions to match common patterns for final verdicts and dates
#     verdict_patterns = [
#         r'final\s*verdict[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'final\s*verdict[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'DATED[:\s]*\d{2}\.\d{2}\.\d{4}',  # Pattern for DATED: dd.mm.yyyy
#     ]
    
#     # Search for the patterns in the text
#     final_verdict = None
#     for pattern in verdict_patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             final_verdict = match.group(0).strip()
#             break
    
#     return final_verdict

# # def extract_parties(text):
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
# #     # New patterns provided
# #     new_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_2 = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_3 = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_4 = re.compile(r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', re.IGNORECASE | re.DOTALL)
    
# #     # Additional new pattern to be added
# #     additional_new_pattern = re.compile(r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
# #     # New pattern to add (provided by user)
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     # Existing patterns
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.DOTALL | re.MULTILINE)
# #     pattern_ellipsis = re.compile(r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     pattern_dots = re.compile(r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondent\s*\(s\)', re.IGNORECASE | re.DOTALL)
# #     pattern_provided = re.compile(r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     pattern_appellant = re.compile(r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)

# #     # Try matching the specific pattern first
# #     specific_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = specific_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
     
# #     # Try matching the new pattern 6
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_6.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Try matching the new pattern next
# #     match = new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
 
# #     # Try matching the new pattern next
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_7.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Try matching the new pattern 2
# #     match = new_pattern_2.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }

# #     # Try matching the new pattern 3
# #     match = new_pattern_3.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     # Try matching the new pattern 4
# #     match = new_pattern_4.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

# #     # Try matching the additional new pattern
# #     match = additional_new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     # Try matching the new pattern 5 (provided by user)
# #     match = new_pattern_5.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

# #     # Try matching the new petitioner and respondents pattern
# #     petitioner_match = new_petitioner_pattern.search(text)
# #     respondents_match = new_respondents_pattern.search(text)

# #     if petitioner_match and respondents_match:
# #         petitioner = petitioner_match.group(1).strip()
# #         respondents_text = respondents_match.group(1).strip()
# #         respondents_list = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents_list if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"

# #     # Try matching the standard pattern
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the ellipses pattern
# #     matches = pattern_ellipsis.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the dots pattern
# #     matches = pattern_dots.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the provided specific format
# #     matches = pattern_provided.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the appellant pattern
# #     matches = pattern_appellant.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the specific parties pattern
# #     matches = pattern_specific_parties.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     new_pattern_8 = re.compile(
# #           r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     ) 
# #     match = new_pattern_8.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('AND')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     return "Parties not found."


# def extract_parties(text):
#     # New patterns for petitioner and respondents
#     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
#     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
#     new_pattern_2_revised = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Check for new_pattern_2_revised match
#     match = new_pattern_2_revised.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondent = match.group(2).strip()
#         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
#     # New patterns provided
#     new_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
#     new_pattern_2 = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
#     new_pattern_3 = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
#     new_pattern_4 = re.compile(r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', re.IGNORECASE | re.DOTALL)
    
#     # Additional new pattern to be added
#     additional_new_pattern = re.compile(r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
#     # New pattern to add (provided by user)
#     new_pattern_5 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
#         re.IGNORECASE | re.DOTALL
#     )
#     # Existing patterns
#     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
#     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.DOTALL | re.MULTILINE)
#     pattern_ellipsis = re.compile(r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', re.IGNORECASE | re.DOTALL)
#     pattern_dots = re.compile(r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondent\s*\(s\)', re.IGNORECASE | re.DOTALL)
#     pattern_provided = re.compile(r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
#     pattern_appellant = re.compile(r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
#     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)

#     # Try matching the specific pattern first
#     specific_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
#     match = specific_pattern.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents = "\n".join([f"{resp}" for resp in respondents])
#         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
     
#     # Try matching the new pattern 6
#     new_pattern_6 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )
#     match = new_pattern_6.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondent = match.group(2).strip()
#         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
#     # Try matching the new pattern next
#     match = new_pattern.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('and')
#         respondents = [resp.strip() for resp in respondents]
#         return {
#             'petitioner': petitioner,
#             'respondents': respondents
#         }
 
#     # Try matching the new pattern next
#     new_pattern_7 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )
#     match = new_pattern_7.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondent = match.group(2).strip()
#         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
#     # Try matching the new pattern 2
#     match = new_pattern_2.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('and')
#         respondents = [resp.strip() for resp in respondents]
#         return {
#             'petitioner': petitioner,
#             'respondents': respondents
#         }

#     # Try matching the new pattern 3
#     match = new_pattern_3.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents = "\n".join([f"{resp}" for resp in respondents])
#         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
#     # Try matching the new pattern 4
#     match = new_pattern_4.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondent = match.group(2).strip()
#         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

#     # Try matching the additional new pattern
#     match = additional_new_pattern.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
#         return {
#             'petitioner': petitioner,
#             'respondents': respondents
#         }
    
#     # Try matching the new pattern 5 (provided by user)
#     match = new_pattern_5.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondent = match.group(2).strip()
#         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

#     # Try matching the new petitioner and respondents pattern
#     petitioner_match = new_petitioner_pattern.search(text)
#     respondents_match = new_respondents_pattern.search(text)

#     if petitioner_match and respondents_match:
#         petitioner = petitioner_match.group(1).strip()
#         respondents_text = respondents_match.group(1).strip()
#         respondents_list = respondents_text.split('\n')
#         respondents = [resp.strip() for resp in respondents_list if resp.strip()]
#         respondents = "\n".join([f"{resp}" for resp in respondents])
#         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"

#     # Try matching the standard pattern
#     petitioners = petitioner_pattern.findall(text)
#     respondents = respondent_pattern.findall(text)
    
#     if petitioners and respondents:
#         petitioners = [p.strip() for p in petitioners]
#         respondents = [r.strip() for r in respondents]
#         parties = []
#         for petitioner, respondent in zip(petitioners, respondents):
#             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     # Try matching the ellipses pattern
#     matches = pattern_ellipsis.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     # Try matching the dots pattern
#     matches = pattern_dots.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     # Try matching the provided specific format
#     matches = pattern_provided.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     # Try matching the appellant pattern
#     matches = pattern_appellant.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     # Try matching the specific parties pattern
#     matches = pattern_specific_parties.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     new_pattern_8 = re.compile(
#           r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
#     ) 
#     match = new_pattern_8.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('AND')
#         respondents = [resp.strip() for resp in respondents]
#         return {
#             'petitioner': petitioner,
#             'respondents': respondents
#         }
    
#     return "Parties not found."



# # def extract_parties(text):
# #     # Pattern to match the specific format
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Check for new_pattern_2_revised match
# #     match = new_pattern_2_revised.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

# #     return "Parties not found."


# def extract_date(text):
#     # Define regex pattern to match dates in various formats
#     date_pattern = (
#         r'(\b(?:[12][0-9]|3[01]|0?[1-9])(st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s\d{4}\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}\b)'
#     )
    
#     # Find all matches of the date pattern in the text
#     matches = re.findall(date_pattern, text)
    
#     # Flatten the list of tuples and remove empty strings
#     dates = [date for match in matches for date in match if date]
    
#     # Remove duplicates by converting to set and back to list
#     unique_dates = list(set(dates))
    
#     return unique_dates



# # def extract_date(text):
# #     # Define regex pattern to match dates in various formats
# #     date_pattern = r'(?:\d{1,2}(?:st|nd|rd|th)?(?:\s)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{4})|(?:\d{1,2}(?:\/|-)\d{1,2}(?:\/|-)\d{2,4})'
    
# #     # Find all matches of the date pattern in the text
# #     matches = re.findall(date_pattern, text)
    
# #     return matches



# def extract_case_title(text):
#     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
#     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
    
#     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
#     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-]+))'
    
#     # First try matching the original pattern
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     # If no match, try the alternative pattern
#     if not match:
#         match = re.search(pattern_alt, text, re.IGNORECASE)
    
#     if match:
#         title = match.group(1).strip()
#         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
#             date = match.group(2).strip()
#             return f"{title} on {date}"
#         else:
#             # Check for unwanted "Author" or other text in the title
#             if "Author" in title:
#                 title = title.split("Author")[0].strip()
#             return title
#     else:
#         return "Title and date not found"


    
# # def extract_case_title(text):
# #     # Pattern to match case titles with varying formats, including dots, colons, and additional parts
# #     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*.*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
    
# #     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
# #     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’]+))'
    
# #     # Updated pattern to match titles like "State Of U.P. Through Principal ... vs M/S Saf Yeast Co.P.Ltd."
# #     pattern_new = r'([A-Z][a-zA-Z\s.,&()\'’\-]+(?:\s+Through\s+Principal\s+.+\s+)?vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-]+)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
    
# #     # First try matching the original pattern
# #     match = re.search(pattern, text, re.IGNORECASE)
    
# #     # If no match, try the alternative pattern
# #     if not match:
# #         match = re.search(pattern_alt, text, re.IGNORECASE)
    
# #     # If still no match, try the new pattern
# #     if not match:
# #         match = re.search(pattern_new, text, re.IGNORECASE)
    
# #     if match:
# #         title = match.group(1).strip()
# #         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
# #             date = match.group(2).strip()
# #             return f"{title} on {date}"
# #         else:
# #             # Check for unwanted "Author" or other text in the title
# #             if "Author" in title:
# #                 title = title.split("Author")[0].strip()
# #             return title
# #     else:
# #         return "Title and date not found"



# # def extract_case_title(text):
# #     # Pattern to match case titles with varying formats, including ellipsis and flexible date formats
# #     pattern = r'((?:(?:State Of|M/S)\s)?[A-Z][a-zA-Z\s.,&()\'\-/]+(?:\s+\.{3}\s+[A-Z][a-zA-Z\s.,&()\'\-/]+)?(?:\s+vs\.?\s+(?:(?:State Of|M/S)\s)?[A-Z][a-zA-Z\s.,&()\'\-/]+(?:\s+\.{3}\s+[A-Z][a-zA-Z\s.,&()\'\-/]+)?)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
    
# #     # Alternative pattern for cases without a date
# #     pattern_alt = r'((?:(?:State Of|M/S)\s)?[A-Z][a-zA-Z\s.,&()\'\-/]+(?:\s+\.{3}\s+[A-Z][a-zA-Z\s.,&()\'\-/]+)?(?:\s+vs\.?\s+(?:(?:State Of|M/S)\s)?[A-Z][a-zA-Z\s.,&()\'\-/]+(?:\s+\.{3}\s+[A-Z][a-zA-Z\s.,&()\'\-/]+)?)?)'
    
# #     # First try matching the pattern with date
# #     match = re.search(pattern, text, re.IGNORECASE)
    
# #     if match:
# #         title = match.group(1).strip()
# #         date = match.group(2).strip()
# #         return f"{title} on {date}"
    
# #     # If no match, try the alternative pattern
# #     match = re.search(pattern_alt, text, re.IGNORECASE)
    
# #     if match:
# #         return match.group(1).strip()
    
# #     return "Title and date not found"

# # def extract_case_title(text):
# #     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
# #     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*.*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
    
# #     # New pattern to match cases like "Aum Capital Market Pvt. Ltd vs Union Of India"
# #     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’]+))'
    
# #     # First try matching the original pattern
# #     match = re.search(pattern, text, re.IGNORECASE)
    
# #     # If no match, try the alternative pattern
# #     if not match:
# #         match = re.search(pattern_alt, text, re.IGNORECASE)
    
# #     if match:
# #         title = match.group(1).strip()
# #         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
# #             date = match.group(2).strip()
# #             return f"{title} on {date}"
# #         else:
# #             # Check for unwanted "Author" or other text in the title
# #             if "Author" in title:
# #                 title = title.split("Author")[0].strip()
# #             return title
# #     else:
# #         return "Title and date not found"




# def extract_court_name(text):
#     # Define a comprehensive pattern for court names, including spaces between letters
#     comprehensive_pattern = (
#         r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T\s*'
#         r'O\s*F\s*I\s*N\s*D\s*I\s*A|'
#         r'Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
#         r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
#     )
    
#     # Search for the comprehensive pattern in the text
#     match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
#     if match:
#         return match.group(0).strip()
#     else:
#         # Define a fallback pattern for court names, including spaces between letters
#         fallback_pattern = r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T|Supreme|High|District) Court'
        
#         # Search for the fallback pattern in the text
#         match = re.search(fallback_pattern, text, re.IGNORECASE)
        
#         if match:
#             return match.group(0).strip()
#         else:
#             return "Court name not found"


# # def extract_court_name(text):
# #     # Define a more comprehensive pattern for court names, including the new format
# #     comprehensive_pattern = (
# #         r'(?:Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
# #         r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
# #     )
    
# #     # Search for the comprehensive pattern in the text
# #     match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
# #     if match:
# #         return match.group(0).strip()
# #     else:
# #         # Define a fallback pattern for court names
# #         fallback_pattern = r'(?:Supreme|High|District) Court'
        
# #         # Search for the fallback pattern in the text
# #         match = re.search(fallback_pattern, text, re.IGNORECASE)
        
# #         if match:
# #             return match.group(0).strip()
# #         else:
# #             return "Court name not found"



# def extract_articles_sections(text):
#     # Comprehensive pattern for articles
#     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Comprehensive pattern for sections
#     section_pattern = re.compile(r'\b(?:Section|Sec\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for clauses
#     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for sub-sections
#     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

#     unique_references = set()  # Using a set to remove duplicates

#     # Function to process matches
#     def process_matches(pattern, prefix):
#         for match in pattern.finditer(text):
#             reference = match.group().strip()
#             if prefix not in reference.lower():
#                 reference = f"{prefix} {reference}"
#             unique_references.add(reference)

#     # Processing all patterns
#     process_matches(article_pattern, "Article :")
#     process_matches(section_pattern, "Section :")
#     process_matches(clause_pattern, "Clause :")
#     process_matches(subsection_pattern, "Sub-section :")

#     if unique_references:
#         return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
#     else:
#         return "No articles, sections, clauses, or sub-sections found."  
    
# def sanitize_text(text):
#     # Remove unwanted symbols using regular expressions
#     sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)
#     return sanitized_text

# # Updated `resolve_coreferences` function
# def resolve_coreferences(text):
#     doc = nlp(text)
#     resolved_text = []
#     for token in doc:
#         if token.dep_ == 'pronoun':
#             antecedent = token.head.text
#             resolved_text.append(antecedent)
#         else:
#             resolved_text.append(token.text)
    
#     return ' '.join(resolved_text)

# # Function to preprocess text with coreference resolution
# def preprocess_text_with_coref_resolution(text):
#     text = resolve_coreferences(text)
#     text = preprocess_text(text)
#     return text

# def print_all_information(text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     print("Case Number:", extract_case_number(text))
#     print("Governing Law:", extract_governing_law(text))
#     print("Final Verdict:", extract_final_verdict(text))
#     print("Parties:", extract_parties(text))
#     print("Date:", extract_date(text))
#     print("Title of the Case:", extract_case_title(text))
#     print("Name of the Court:", extract_court_name(text))
#     print("Articles:", extract_articles_sections(text))
#     print("Summary of the Case:", generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors))

# # Main interaction loop
# try:
#     print("Bot: I will provide information about the legal document.")
#     print_all_information(raw_text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model)
# except KeyboardInterrupt:
#     print("\nBot: Thanks for talking, Bye!")



























# import re
# import nltk
# import contractions
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from autocorrect import Speller
# import emoji
# import regex 
# import gensim.downloader as api
# import fitz  # PyMuPDF
# from nltk import pos_tag
# import warnings
# import numpy as np

# warnings.simplefilter(action='ignore', category=UserWarning)

# #nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Load the English language model
# nlp = spacy.load('en_core_web_sm')

# # Set max_length to a value that accommodates your text length
# nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

# # Load pre-trained Word2Vec model
# w2v_model = api.load('word2vec-google-news-300')

# # Initialize spell checker
# spell = Speller()

# # Define stopword2Vec
# stop_words = set(stopwords.words('english'))


# # Preprocessing function with lemmatization, spell checking, and NER tagging
# def preprocess_text(text):
    
#     # Correct spelling errors
#     text = spell(text)
    
#     # Remove HTML tags, URLs, and special characters
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Expand contractions
#     text = contractions.fix(text)
    
#     # Remove citations
#     text = re.sub(r'\[[0-9]+\]', '', text)
    
#     # Remove extra whitespaces
#     text = re.sub(r'\s+', ' ', text)
    
#     # Tokenization and NER tagging using spaCy
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if token.ent_type_:
#             tokens.append(token.ent_type_)
#         else:
#             tokens.append(token.text)
    
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Remove repeated characters
#     tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]
    
#     # Remove single characters and numeric tokens
#     tokens = [word for word in tokens if len(word) > 1 and not word.isdigit()]
    
#     # Handle emojis
#     text = emoji.demojize(text)
#     text = text.replace(":", "")
    
#     # Handle emoticons
#     emoticons = regex.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = regex.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
#     text = text + ' '.join(emoticons)
    
#     # Join tokens back into a string
#     preprocessed_text = ' '.join(tokens)
    
#     return preprocessed_text

# def calculate_similarity(sentence1, sentence2):
#     # Tokenize the sentences
#     tokens1 = word_tokenize(sentence1)
#     tokens2 = word_tokenize(sentence2)
    
#     # Filter out stopwords
#     tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
#     tokens2 = [word for word in tokens2 if word.lower() not in stop_words]
    
#     # Get the Word2Vec vectors for each word
#     vectors1 = [w2v_model[word] for word in tokens1 if word in w2v_model]
#     vectors2 = [w2v_model[word] for word in tokens2 if word in w2v_model]

#     # Calculate the average vectors for each sentence
#     if vectors1 and vectors2:
#         avg_vector1 = np.mean(vectors1, axis=0)
#         avg_vector2 = np.mean(vectors2, axis=0)
        
#         # Calculate the cosine similarity between the average vectors
#         similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
#         return similarity
#     else:
#         return 0.0  # Return 0 if no vectors are found or all words are OOV

# # Keep track of previous questions and responses
# previous_questions = []
# previous_responses = []

# def generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, top_k=15, similarity_threshold=0.5):
#     global previous_questions, previous_responses

#     bot_response = ''
    
#     # Preprocess user input
#     processed_input = preprocess_text(user_input)
    
#     # Check if the processed input is empty or contains only stopwords
#     if not processed_input or all(word in stop_words for word in processed_input.split()):
#         return "I am sorry, I don't understand."
    
#     # Check if the current question is the same as a previous one
#     if processed_input in previous_questions:
#         index = previous_questions.index(processed_input)
#         return previous_responses[index]
    
#     # If not, continue with Word2Vec processing
#     similarities = []
#     for sent in sent_tokens:
#         similarity = calculate_similarity(processed_input, sent)
#         similarities.append(similarity)
    
#     # Convert similarities to a NumPy array for easier processing
#     similarities = np.array(similarities)
    
#     # Sort the similarities in descending order
#     sorted_indices = np.argsort(similarities)[::-1]
    
#     # Find the top k most similar sentences that are not in previous responses
#     top_k_sentences = []
#     for index in sorted_indices:
#         if len(top_k_sentences) < top_k and similarities[index] >= similarity_threshold:
#             top_k_sentences.append(sent_tokens[index])
    
#     # Assign the top k sentences to bot_response
#     if top_k_sentences:
#         bot_response = '\n'.join(top_k_sentences)
#         # Filter out URLs and unwanted tags from the response
#         bot_response = re.sub(r'http\S+', '', bot_response)  # Remove URLs
        
#         # Track previous questions and responses
#         previous_questions.append(processed_input)
#         previous_responses.append(bot_response)
#     else:
#         bot_response = "I am sorry, I don't get enough details."
    
#     return bot_response  # Return the top k sentences separated by newline characters

# # Read text from the local PDF file using PyMuPDF
# file_path = 'C:/Users/Viswajith/Downloads/Amarjeet_Singh_Chawla_And_Anr_vs_Union_Of_India_And_Anr_on_9_May_2022 (1).PDF'
# try:
#     pdf_document = fitz.open(file_path)
#     raw_text = ''
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         raw_text += page.get_text()
#     pdf_document.close()
#     print("PDF file read successfully.")
# except Exception as e:
#     print("Error reading PDF file:", e)
#     exit()

# # Tokenize the raw text to obtain sentence tokens
# sent_tokens = nltk.sent_tokenize(raw_text)

# # Preprocess the text
# preprocessed_sent_tokens = [preprocess_text(sent) for sent in sent_tokens]

# # Define TF-IDF vectorizer with optimized parameters and TF-IDF weighting schemes
# word_vectorizer = TfidfVectorizer(
#     tokenizer=word_tokenize,     # Use NLTK's word tokenizer
#     stop_words='english',        # Use English stopwords
#     ngram_range=(1, 3),          # Use unigrams and bigrams
#     max_features=15000,           # Limit the vocabulary size to the top 5000 features
#     token_pattern=r'\b\w+\b',    # Use words as tokens
#     sublinear_tf=True,           # Apply sublinear term frequency scaling
#     smooth_idf=True,             # Smooth IDF weights by adding one to document frequencies
#     norm='l2'                    # Normalize TF-IDF vectors to unit length
# )

# # Apply TF-IDF vectorization on the preprocessed text
# word_vectors = word_vectorizer.fit_transform(preprocessed_sent_tokens)


# # Generate response function
# def generate_response(user_input, text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     # Check if the user is asking for specific information
#     if 'case number' in user_input or 'case no' in user_input:
#         return extract_case_number(text)
#     elif 'governing law' in user_input:
#         return extract_governing_law(text)
#     elif 'final verdict' in user_input:
#         return extract_final_verdict(text)
#     elif 'party' in user_input:
#         return extract_parties(text)
#     elif 'date' in user_input:
#         return extract_date(text)
#     elif 'title of the case' in user_input or 'case title' in user_input:
#         return extract_case_title(text)
#     elif 'summary of the case' in user_input or 'case summary' in user_input:
#         return generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
#     elif 'name of the court' in user_input or 'court name' in user_input:
#         return extract_court_name(text)
#     elif 'article' in user_input:
#         return extract_articles_sections(text)
#     else:
#         # Handle other types of questions
#         return "I'm sorry, I don't know."
    
# def extract_case_number(text):
#     # Regular expression pattern for matching case numbers
#     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?\d{1,}\s*(?:of|OF)\s*\d{4}\b'
    
#     # Find all matches of the pattern in the text
#     case_numbers = re.findall(pattern, text)
    
#     # Return a list of unique case numbers
#     return list(set(case_numbers))

# def extract_governing_law(text):
#     # Define keywords for criminal law and civil law
#     criminal_law_keywords = [
#         'criminal', 'crime', 'offense', 'prosecution', 'defendant', 'felony', 'misdemeanor'
#     ]
#     civil_law_keywords = [
#         'civil', 'plaintiff', 'defendant', 'contract', 'tort', 'property', 'family', 'probate'
#     ]
    
#     # Tokenize the text and tag the parts of speech
#     tokens = word_tokenize(text.lower())
#     tagged_tokens = pos_tag(tokens)
    
#     # Extract nouns and adjectives from the tagged tokens
#     nouns_adjectives = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]
    
#     # Check for criminal law keywords
#     for keyword in criminal_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Criminal Law"
    
#     # Check for civil law keywords
#     for keyword in civil_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Civil Law"
    
#     return "Governing law not identified"

# def extract_final_verdict(text):
#     # Define regular expressions to match common patterns for final verdicts and dates
#     verdict_patterns = [
#         r'final\s*verdict[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'final\s*verdict[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         #r'DATED[:\s]*\d{2}\.\d{2}\.\d{4}',  # Pattern for DATED: dd.mm.yyyy
#     ]
    
#     # Search for the patterns in the text
#     final_verdict = None
#     for pattern in verdict_patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             final_verdict = match.group(0).strip()
#             break
    
#     if not final_verdict:
#         # If no final verdict date is found, try to find a date in the title of the case
#         title_pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
#         title_match = re.search(title_pattern, text, re.IGNORECASE)
#         if title_match:
#             final_verdict = title_match.group(2).strip()
    
#     return final_verdict

# # def extract_parties(text):
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Check for new_pattern_2_revised match
# #     match = new_pattern_2_revised.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # New patterns provided
# #     new_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_2 = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_3 = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_4 = re.compile(r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', re.IGNORECASE | re.DOTALL)
     
# #     # Additional new pattern to be added
# #     additional_new_pattern = re.compile(r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
# #     # New pattern to add (provided by user)
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     # Existing patterns
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.DOTALL | re.MULTILINE)
# #     pattern_ellipsis = re.compile(r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     pattern_dots = re.compile(r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondent\s*\(s\)', re.IGNORECASE | re.DOTALL)
# #     pattern_provided = re.compile(r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     pattern_appellant = re.compile(r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)

# #     # Try matching the specific pattern first
# #     specific_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = specific_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
     
# #     # Try matching the new pattern 6
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_6.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Try matching the new pattern next
# #     match = new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
 
# #     # Try matching the new pattern next
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_7.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Try matching the new pattern 2
# #     match = new_pattern_2.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }

# #     # Try matching the new pattern 3
# #     match = new_pattern_3.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     # Try matching the new pattern 4
# #     match = new_pattern_4.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

# #     # Try matching the additional new pattern
# #     match = additional_new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     # Try matching the new pattern 5 (provided by user)
# #     match = new_pattern_5.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

# #     # Try matching the new petitioner and respondents pattern
# #     petitioner_match = new_petitioner_pattern.search(text)
# #     respondents_match = new_respondents_pattern.search(text)

# #     if petitioner_match and respondents_match:
# #         petitioner = petitioner_match.group(1).strip()
# #         respondents_text = respondents_match.group(1).strip()
# #         respondents_list = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents_list if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"

# #     # Try matching the standard pattern
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the ellipses pattern
# #     matches = pattern_ellipsis.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the dots pattern
# #     matches = pattern_dots.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the provided specific format
# #     matches = pattern_provided.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the appellant pattern
# #     matches = pattern_appellant.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the specific parties pattern
# #     matches = pattern_specific_parties.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     new_pattern_8 = re.compile(
# #           r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     ) 
# #     match = new_pattern_8.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('AND')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     return "Parties not found."


# #little bit ok

# # def extract_parties(text):
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_7.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_6.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Check for new_pattern_2_revised match
# #     match = new_pattern_2_revised.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # New patterns provided
# #     new_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_2 = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_3 = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_4 = re.compile(r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_4.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
     
# #     # Additional new pattern to be added
# #     additional_new_pattern = re.compile(r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = additional_new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }

    
# #     # New pattern to add (provided by user)
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_5.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Existing patterns
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.DOTALL | re.MULTILINE)
# #     pattern_ellipsis = re.compile(r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     pattern_dots = re.compile(r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondent\s*\(s\)', re.IGNORECASE | re.DOTALL)
# #     pattern_provided = re.compile(r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     pattern_appellant = re.compile(r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
# #     # Find all matches
# #     matches = pattern_specific_parties_1.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             petitioner = petitioner.strip()
# #             respondent = respondent.strip()
# #             # Check if both petitioner and respondent are in uppercase
# #             if petitioner.isupper() and respondent.isupper():
# #                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
# #         # Join and return the result
# #         return "\n\n".join(parties)

# #     # Try matching the specific pattern first
# #     specific_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = specific_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
         
# #     # Try matching the new pattern next
# #     match = new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     # Try matching the new pattern 2
# #     match = new_pattern_2.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }

# #     # Try matching the new pattern 3
# #     match = new_pattern_3.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"

# #     # Try matching the new petitioner and respondents pattern
# #     petitioner_match = new_petitioner_pattern.search(text)
# #     respondents_match = new_respondents_pattern.search(text)

# #     if petitioner_match and respondents_match:
# #         petitioner = petitioner_match.group(1).strip()
# #         respondents_text = respondents_match.group(1).strip()
# #         respondents_list = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents_list if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"

# #     # Try matching the standard pattern
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the ellipses pattern
# #     matches = pattern_ellipsis.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the dots pattern
# #     matches = pattern_dots.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the provided specific format
# #     matches = pattern_provided.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the appellant pattern
# #     matches = pattern_appellant.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     new_pattern_8 = re.compile(
# #           r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     ) 
# #     match = new_pattern_8.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('AND')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     return "Parties not found."

# #9/7/24
# # #continue tommorrow

# # def extract_parties(text):
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     petitioner_match = new_petitioner_pattern.search(text)
# #     respondents_match = new_respondents_pattern.search(text)

# #     if petitioner_match and respondents_match:
# #         petitioner = petitioner_match.group(1).strip()
# #         respondents_text = respondents_match.group(1).strip()
# #         respondents_list = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents_list if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_7.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_6.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Check for new_pattern_2_revised match
# #     match = new_pattern_2_revised.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     specific_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = specific_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     # New patterns provided
# #     new_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     new_pattern_2 = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_2.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     new_pattern_3 = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_3.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     new_pattern_4 = re.compile(r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_4.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
     
# #     # Additional new pattern to be added
# #     additional_new_pattern = re.compile(r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = additional_new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
  
# #     # New pattern to add (provided by user)
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_5.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Existing patterns
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.DOTALL | re.MULTILINE)
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_ellipsis = re.compile(r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_ellipsis.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_dots = re.compile(r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondent\s*\(s\)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_dots.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_provided = re.compile(r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_provided.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_appellant = re.compile(r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_appellant.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties_1.findall(text) 
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             petitioner = petitioner.strip()
# #             respondent = respondent.strip()
# #             # Check if both petitioner and respondent are in uppercase
# #             if petitioner.isupper() and respondent.isupper():
# #                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
# #         # Join and return the result
# #         return "\n\n".join(parties)
    
# #     new_pattern_8 = re.compile(
# #           r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     ) 
# #     match = new_pattern_8.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('AND')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     return "Parties not found."



# #as of now ok

# # def extract_parties(text):
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     petitioner_match = new_petitioner_pattern.search(text)
# #     respondents_match = new_respondents_pattern.search(text)

# #     if petitioner_match and respondents_match:
# #         petitioner = petitioner_match.group(1).strip()
# #         respondents_text = respondents_match.group(1).strip()
# #         respondents_list = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents_list if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_7.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_6.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Check for new_pattern_2_revised match
# #     match = new_pattern_2_revised.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     specific_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = specific_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     # New patterns provided
# #     new_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     new_pattern_2 = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_2.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     new_pattern_3 = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_3.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     new_pattern_4 = re.compile(r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_4.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
     
# #     # Additional new pattern to be added
# #     additional_new_pattern = re.compile(r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = additional_new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
  
# #     # New pattern to add (provided by user)
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_5.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Existing patterns
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.DOTALL | re.MULTILINE)
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_ellipsis = re.compile(r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_ellipsis.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_dots = re.compile(r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_dots.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_dashes = re.compile(r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_dashes.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_provided = re.compile(r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_provided.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_appellant = re.compile(r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_appellant.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # new_pattern_8 = re.compile(
# #     #       r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     # ) 
# #     # match = new_pattern_8.search(text)
# #     # if match:
# #     #     petitioner = match.group(1).strip()
# #     #     respondents_text = match.group(2).strip()
# #     #     respondents = respondents_text.split('AND')
# #     #     respondents = [resp.strip() for resp in respondents]
# #     #     return {
# #     #         'petitioner': petitioner,
# #     #         'respondents': respondents
# #     #     }
    
# #     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties_1.findall(text) 
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             petitioner = petitioner.strip()
# #             respondent = respondent.strip()
# #             # Check if both petitioner and respondent are in uppercase
# #             if petitioner.isupper() and respondent.isupper():
# #                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
# #         # Join and return the result
# #         return "\n\n".join(parties)
    
# #     # If no patterns match, return "Parties not found."
# #     return "Parties not found."


# ###want to change every patterns to findall
# def extract_parties(text):
    
#     # pattern_10 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',re.IGNORECASE | re.DOTALL)
    
#     # matches = pattern_10.search(text)
#     # if matches:
#     #     petitioner = matches.group(1).strip()
#     #     respondents = matches.group(2).strip().split('\n')
#     #     respondents = [resp.strip() for resp in respondents if resp.strip()]
#     #     respondents_text = "\n".join(respondents)
#     #     return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"

        
    
#     pattern_9 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*Petitio\s*.*?Vs\.\s*([\s\S]*?)\s*\.{3,}\s*Respondent',re.IGNORECASE | re.DOTALL)

#     matches = pattern_9.search(text)
#     if matches:
#         petitioner = matches.group(1).strip()
#         respondent = matches.group(2).strip()
#         return f"Petitioners:\n{petitioner}\nRespondents:\n{respondent}"
            
#     # New patterns for petitioner and respondents
#     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
#     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

#     petitioner_matches = new_petitioner_pattern.findall(text)
#     respondents_matches = new_respondents_pattern.findall(text)

#     if petitioner_matches and respondents_matches:
#         petitioners = [match.strip() for match in petitioner_matches]
#         respondents = []
#         for match in respondents_matches:
#             respondents_list = match.strip().split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
#         petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
#         respondents = "\n".join([f"{respondent}" for respondent in respondents])
#         return f"Petitioners:\n{petitioners}\nRespondents:\n{respondents}"
       
#     new_pattern_7 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_7.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
    
    
#     new_pattern_6 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_6.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
     
    
#     new_pattern_2_revised = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_2_revised.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
#     specific_pattern = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = specific_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
#     new_pattern = re.compile(
#         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('and')
#             respondents.extend([resp.strip() for resp in respondents_list])
        
#         return {
#             'petitioners': petitioners,
#             'respondents': respondents
#         }
    
#     new_pattern_2 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_2.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('and')
#             respondents.extend([resp.strip() for resp in respondents_list])
        
#         return {
#             'petitioners': petitioners,
#             'respondents': respondents
#         }
    
#     new_pattern_3 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_3.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
    
    
#     new_pattern_4 = re.compile(
#         r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_4.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
     
#     additional_new_pattern = re.compile(
#         r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = additional_new_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents_text = match[1].strip()
#             respondents_list = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
#             petitioners.append(petitioner)
#             respondents.extend(respondents_list)
        
#         return {
#             'petitioners': petitioners,
#             'respondents': respondents
#         }
  
#     new_pattern_5 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_5.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
#     # Existing patterns
#     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
#     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
#     petitioners = petitioner_pattern.findall(text)
#     respondents = respondent_pattern.findall(text)
    
#     if petitioners and respondents:
#         petitioners = [p.strip() for p in petitioners]
#         respondents = [r.strip() for r in respondents]
#         parties = []
#         for petitioner, respondent in zip(petitioners, respondents):
#             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_10 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',re.IGNORECASE | re.DOTALL)
    
#     matches = pattern_10.search(text)
#     if matches:
#         petitioner = matches.group(1).strip()
#         respondents = matches.group(2).strip().split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents_text = "\n".join(respondents)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     #return "No matches found."

# #2    
#     pattern_ellipsis = re.compile(
#         r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_ellipsis.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioners = [p.strip() for p in petitioner.split('\n') if p.strip()]
#             respondents = [r.strip() for r in respondent.split('\n') if r.strip()]
            
#             petitioner_str = "\n".join([f"Petitioner: {p}" for p in petitioners])
#             respondent_str = "\n".join([f"Respondent: {r}" for r in respondents])
            
#             parties.append(f"{petitioner_str}\n{respondent_str}")
        
#         return "\n\n".join(parties)
    
#     #return "No matches found."
    
#     pattern_dots = re.compile(
#         r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dots.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioners = [p.strip() for p in petitioner.split('\n') if p.strip()]
#             respondents = [r.strip() for r in respondent.split('\n') if r.strip()]
            
#             petitioner_str = "\n".join([f"Petitioner: {p}" for p in petitioners])
#             respondent_str = "\n".join([f"Respondent: {r}" for r in respondents])
            
#             parties.append(f"{petitioner_str}\n{respondent_str}")
        
#         return "\n\n".join(parties)
    
#     #return "No matches found"
    
#     pattern_dashes = re.compile(
#         r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dashes.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioners = [p.strip() for p in petitioner.split('\n') if p.strip()]
#             respondents = [r.strip() for r in respondent.split('\n') if r.strip()]
            
#             petitioner_str = "\n".join([f"Petitioner: {p}" for p in petitioners])
#             respondent_str = "\n".join([f"Respondent: {r}" for r in respondents])
            
#             parties.append(f"{petitioner_str}\n{respondent_str}")
        
#         return "\n\n".join(parties)
    
#     #return "No matches found."
    
#     pattern_provided = re.compile(
#         r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_provided.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioners = [p.strip() for p in petitioner.split('\n') if p.strip()]
#             respondents = [r.strip() for r in respondent.split('\n') if r.strip()]
            
#             petitioner_str = "\n".join([f"Petitioner: {p}" for p in petitioners])
#             respondent_str = "\n".join([f"Respondent: {r}" for r in respondents])
            
#             parties.append(f"{petitioner_str}\n{respondent_str}")
        
#         return "\n\n".join(parties)
    
#     #return "No matches found."
    
#     pattern_appellant = re.compile(
#         r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_appellant.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioners = [p.strip() for p in petitioner.split('\n') if p.strip()]
#             respondents = [r.strip() for r in respondent.split('\n') if r.strip()]
            
#             petitioner_str = "\n".join([f"Petitioner: {p}" for p in petitioners])
#             respondent_str = "\n".join([f"Respondent: {r}" for r in respondents])
            
#             parties.append(f"{petitioner_str}\n{respondent_str}")
        
#         return "\n\n".join(parties)
    
#     #return "No matches found."
    
#     pattern_specific_parties = re.compile(
#         r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_specific_parties.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioners = [p.strip() for p in petitioner.split('\n') if p.strip()]
#             respondents = [r.strip() for r in respondent.split('\n') if r.strip()]
            
#             petitioner_str = "\n".join([f"Petitioner: {p}" for p in petitioners])
#             respondent_str = "\n".join([f"Respondent: {r}" for r in respondents])
            
#             parties.append(f"{petitioner_str}\n{respondent_str}")
        
#         return "\n\n".join(parties)
    
#     #return "No matches found."
        
#     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
#     matches = pattern_specific_parties_1.findall(text) 
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioner = petitioner.strip()
#             respondent = respondent.strip()
#             # Check if both petitioner and respondent are in uppercase
#             if petitioner.isupper() and respondent.isupper():
#                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
#         # Join and return the result
#         return "\n\n".join(parties)
    
#     # pattern_10 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',re.IGNORECASE | re.DOTALL)
    
#     # matches = pattern_10.search(text)
#     # if matches:
#     #     petitioner = matches.group(1).strip()
#     #     respondents = matches.group(2).strip().split('\n')
#     #     respondents = [resp.strip() for resp in respondents if resp.strip()]
#     #     respondents_text = "\n".join(respondents)
#     #     return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    

            
#     # new_pattern_8 = re.compile(
#     #       r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
#     # ) 
#     # match = new_pattern_8.search(text)
#     # if match:
#     #     petitioner = match.group(1).strip()
#     #     respondents_text = match.group(2).strip()
#     #     respondents = respondents_text.split('AND')
#     #     respondents = [resp.strip() for resp in respondents]
#     #     return {
#     #         'petitioner': petitioner,
#     #         'respondents': respondents
#     #     }
    
#     # If no patterns match, return "Parties not found."
#     return "Parties not found."

# #check this tommorrow  ak gopalan
# # def extract_parties(text):
#     # petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*Vs\.', re.IGNORECASE)
#     # respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.MULTILINE)

#     # petitioner_match = petitioner_pattern.search(text)
#     # respondent_match = respondent_pattern.search(text)

#     # petitioner = petitioner_match.group(1).strip() if petitioner_match else "Not Found"
#     # respondent = respondent_match.group(1).strip() if respondent_match else "Not Found"

#     # return f"Petitioner: {petitioner}\nRespondent: {respondent}"

# # def extract_parties(text):
# #     # Pattern to match the specific format
# #     pattern = re.compile(r'([^\n\r]+)\s*\.{3,}\s*\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',re.IGNORECASE | re.DOTALL)
    
# #     matches = pattern.search(text)
# #     if matches:
# #         petitioner = matches.group(1).strip()
# #         respondents = matches.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         print(f"Petitioner:\n{petitioner}")
# #         print(f"\nRespondents:\n{respondents_text}")
# #     else:
# #         print("No match found.")
     
# def extract_date(text):
#     # Define regex pattern to match dates in various formats
#     date_pattern = (
#         r'(\b(?:[12][0-9]|3[01]|0?[1-9])(st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s\d{4}\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}\b)'
#     )
    
#     # Find all matches of the date pattern in the text
#     matches = re.findall(date_pattern, text)
    
#     # Flatten the list of tuples and remove empty strings
#     dates = [date for match in matches for date in match if date]
    
#     # Remove duplicates by converting to set and back to list
#     unique_dates = list(set(dates))
    
#     return unique_dates

# def extract_case_title(text):
#     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
#     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/0-9]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
  
#     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
#     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-0-9]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-0-9]+))'
    
#     # First try matching the original pattern
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     # If no match, try the alternative pattern
#     if not match:
#         match = re.search(pattern_alt, text, re.IGNORECASE)
    
#     if match:
#         title = match.group(1).strip()
#         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
#             date = match.group(2).strip()
#             return f"{title} on {date}"
#         else:
#             # Check for unwanted "Author" or other text in the title
#             if "Author" in title:
#                 title = title.split("Author")[0].strip()
#             return title
#     else:
#         return "Title and date not found"


# # def extract_case_title(text):
# #     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
# #     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
  
# #     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
# #     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-]+))'
    
# #     # First try matching the original pattern
# #     match = re.search(pattern, text, re.IGNORECASE)
    
# #     # If no match, try the alternative pattern
# #     if not match:
# #         match = re.search(pattern_alt, text, re.IGNORECASE)
    
# #     if match:
# #         title = match.group(1).strip()
# #         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
# #             date = match.group(2).strip()
# #             return f"{title} on {date}"
# #         else:
# #             # Check for unwanted "Author" or other text in the title
# #             if "Author" in title:
# #                 title = title.split("Author")[0].strip()
# #             return title
# #     else:
# #         return "Title and date not found"

# def extract_court_name(text):
#     # Define a comprehensive pattern for court names, including spaces between letters
#     comprehensive_pattern = (
#         r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T\s*'
#         r'O\s*F\s*I\s*N\s*D\s*I\s*A|'
#         r'Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
#         r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
#     )
    
#     # Search for the comprehensive pattern in the text
#     match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
#     if match:
#         return match.group(0).strip()
#     else:
#         # Define a fallback pattern for court names, including spaces between letters
#         fallback_pattern = r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T|Supreme|High|District) Court'
        
#         # Search for the fallback pattern in the text
#         match = re.search(fallback_pattern, text, re.IGNORECASE)
        
#         if match:
#             return match.group(0).strip()
#         else:
#             return "Court name not found"

# def extract_articles_sections(text):
#     # Comprehensive pattern for articles
#     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Comprehensive pattern for sections
#     section_pattern = re.compile(r'\b(?:Section|Sec\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for clauses
#     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for sub-sections
#     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

#     unique_references = set()  # Using a set to remove duplicates

#     # Function to process matches
#     def process_matches(pattern, prefix):
#         for match in pattern.finditer(text):
#             reference = match.group().strip()
#             if prefix not in reference.lower():
#                 reference = f"{prefix} {reference}"
#             unique_references.add(reference)

#     # Processing all patterns
#     process_matches(article_pattern, "Article :")
#     process_matches(section_pattern, "Section :")
#     process_matches(clause_pattern, "Clause :")
#     process_matches(subsection_pattern, "Sub-section :")

#     if unique_references:
#         return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
#     else:
#         return "No articles, sections, clauses, or sub-sections found."  
    
# def sanitize_text(text):
#     # Remove unwanted symbols using regular expressions
#     sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)
#     return sanitized_text

# # Updated `resolve_coreferences` function
# def resolve_coreferences(text):
#     doc = nlp(text)
#     resolved_text = []
#     for token in doc:
#         if token.dep_ == 'pronoun':
#             antecedent = token.head.text
#             resolved_text.append(antecedent)
#         else:
#             resolved_text.append(token.text)
    
#     return ' '.join(resolved_text)

# # Function to preprocess text with coreference resolution
# def preprocess_text_with_coref_resolution(text):
#     text = resolve_coreferences(text)
#     text = preprocess_text(text)
#     return text

# def print_all_information(text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     print("Case Number:", extract_case_number(text))
#     print("Governing Law:", extract_governing_law(text))
#     print("Final Verdict:", extract_final_verdict(text))
#     print("Parties:", extract_parties(text))
#     print("Date:", extract_date(text))
#     print("Title of the Case:", extract_case_title(text))
#     print("Name of the Court:", extract_court_name(text))
#     print("Articles:", extract_articles_sections(text))
#     print("Summary of the Case:", generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors))

# # Main interaction loop
# try:
#     print("Bot: I will provide information about the legal document.")
#     print_all_information(raw_text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model)
# except KeyboardInterrupt:
#     print("\nBot: Thanks for talking, Bye!")


























# import re
# import nltk
# import contractions
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from autocorrect import Speller
# import emoji
# import regex 
# import gensim.downloader as api
# import fitz  # PyMuPDF
# from nltk import pos_tag
# import warnings
# import numpy as np

# warnings.simplefilter(action='ignore', category=UserWarning)

# #nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Load the English language model
# nlp = spacy.load('en_core_web_sm')

# # Set max_length to a value that accommodates your text length
# nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

# # Load pre-trained Word2Vec model
# w2v_model = api.load('word2vec-google-news-300')

# # Initialize spell checker
# spell = Speller()

# # Define stopword2Vec
# stop_words = set(stopwords.words('english'))


# # Preprocessing function with lemmatization, spell checking, and NER tagging
# def preprocess_text(text):
    
#     # Correct spelling errors
#     text = spell(text)
    
#     # Remove HTML tags, URLs, and special characters
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Expand contractions
#     text = contractions.fix(text)
    
#     # Remove citations
#     text = re.sub(r'\[[0-9]+\]', '', text)
    
#     # Remove extra whitespaces
#     text = re.sub(r'\s+', ' ', text)
    
#     # Tokenization and NER tagging using spaCy
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if token.ent_type_:
#             tokens.append(token.ent_type_)
#         else:
#             tokens.append(token.text)
    
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Remove repeated characters
#     tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]
    
#     # Remove single characters and numeric tokens
#     tokens = [word for word in tokens if len(word) > 1 and not word.isdigit()]
    
#     # Handle emojis
#     text = emoji.demojize(text)
#     text = text.replace(":", "")
    
#     # Handle emoticons
#     emoticons = regex.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = regex.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
#     text = text + ' '.join(emoticons)
    
#     # Join tokens back into a string
#     preprocessed_text = ' '.join(tokens)
    
#     return preprocessed_text

# def calculate_similarity(sentence1, sentence2):
#     # Tokenize the sentences
#     tokens1 = word_tokenize(sentence1)
#     tokens2 = word_tokenize(sentence2)
    
#     # Filter out stopwords
#     tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
#     tokens2 = [word for word in tokens2 if word.lower() not in stop_words]
    
#     # Get the Word2Vec vectors for each word
#     vectors1 = [w2v_model[word] for word in tokens1 if word in w2v_model]
#     vectors2 = [w2v_model[word] for word in tokens2 if word in w2v_model]

#     # Calculate the average vectors for each sentence
#     if vectors1 and vectors2:
#         avg_vector1 = np.mean(vectors1, axis=0)
#         avg_vector2 = np.mean(vectors2, axis=0)
        
#         # Calculate the cosine similarity between the average vectors
#         similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
#         return similarity
#     else:
#         return 0.0  # Return 0 if no vectors are found or all words are OOV

# # Keep track of previous questions and responses
# previous_questions = []
# previous_responses = []

# def generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, top_k=15, similarity_threshold=0.5):
#     global previous_questions, previous_responses

#     bot_response = ''
    
#     # Preprocess user input
#     processed_input = preprocess_text(user_input)
    
#     # Check if the processed input is empty or contains only stopwords
#     if not processed_input or all(word in stop_words for word in processed_input.split()):
#         return "I am sorry, I don't understand."
    
#     # Check if the current question is the same as a previous one
#     if processed_input in previous_questions:
#         index = previous_questions.index(processed_input)
#         return previous_responses[index]
    
#     # If not, continue with Word2Vec processing
#     similarities = []
#     for sent in sent_tokens:
#         similarity = calculate_similarity(processed_input, sent)
#         similarities.append(similarity)
    
#     # Convert similarities to a NumPy array for easier processing
#     similarities = np.array(similarities)
    
#     # Sort the similarities in descending order
#     sorted_indices = np.argsort(similarities)[::-1]
    
#     # Find the top k most similar sentences that are not in previous responses
#     top_k_sentences = []
#     for index in sorted_indices:
#         if len(top_k_sentences) < top_k and similarities[index] >= similarity_threshold:
#             top_k_sentences.append(sent_tokens[index])
    
#     # Assign the top k sentences to bot_response
#     if top_k_sentences:
#         bot_response = '\n'.join(top_k_sentences)
#         # Filter out URLs and unwanted tags from the response
#         bot_response = re.sub(r'http\S+', '', bot_response)  # Remove URLs
        
#         # Track previous questions and responses
#         previous_questions.append(processed_input)
#         previous_responses.append(bot_response)
#     else:
#         bot_response = "I am sorry, I don't get enough details."
    
#     return bot_response  # Return the top k sentences separated by newline characters

# # Read text from the local PDF file using PyMuPDF
# file_path = 'C:/Users/Viswajith/Downloads/P_C_Rajan_vs_The_Inspector_General_Of_Registration_on_9_March_2023.PDF'
# try:
#     pdf_document = fitz.open(file_path)
#     raw_text = ''
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         raw_text += page.get_text()
#     pdf_document.close()
#     print("PDF file read successfully.")
# except Exception as e:
#     print("Error reading PDF file:", e)
#     exit()

# # Tokenize the raw text to obtain sentence tokens
# sent_tokens = nltk.sent_tokenize(raw_text)

# # Preprocess the text
# preprocessed_sent_tokens = [preprocess_text(sent) for sent in sent_tokens]

# # Define TF-IDF vectorizer with optimized parameters and TF-IDF weighting schemes
# word_vectorizer = TfidfVectorizer(
#     tokenizer=word_tokenize,     # Use NLTK's word tokenizer
#     stop_words='english',        # Use English stopwords
#     ngram_range=(1, 3),          # Use unigrams and bigrams
#     max_features=15000,           # Limit the vocabulary size to the top 5000 features
#     token_pattern=r'\b\w+\b',    # Use words as tokens
#     sublinear_tf=True,           # Apply sublinear term frequency scaling
#     smooth_idf=True,             # Smooth IDF weights by adding one to document frequencies
#     norm='l2'                    # Normalize TF-IDF vectors to unit length
# )

# # Apply TF-IDF vectorization on the preprocessed text
# word_vectors = word_vectorizer.fit_transform(preprocessed_sent_tokens)


# # Generate response function
# def generate_response(user_input, text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     # Check if the user is asking for specific information
#     if 'case number' in user_input or 'case no' in user_input:
#         return extract_case_number(text)
#     elif 'governing law' in user_input:
#         return extract_governing_law(text)
#     elif 'final verdict' in user_input:
#         return extract_final_verdict(text)
#     elif 'party' in user_input:
#         return extract_parties(text)
#     elif 'date' in user_input:
#         return extract_date(text)
#     elif 'title of the case' in user_input or 'case title' in user_input:
#         return extract_case_title(text)
#     elif 'summary of the case' in user_input or 'case summary' in user_input:
#         return generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
#     elif 'name of the court' in user_input or 'court name' in user_input:
#         return extract_court_name(text)
#     elif 'article' in user_input:
#         return extract_articles_sections(text)
#     else:
#         # Handle other types of questions
#         return "I'm sorry, I don't know."
    
# def extract_case_number(text):
#     # Regular expression pattern for matching case numbers
#     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?\d{1,}\s*(?:of|OF)\s*\d{4}\b'
    
#     # Find all matches of the pattern in the text
#     case_numbers = re.findall(pattern, text)
    
#     # Return a list of unique case numbers
#     return list(set(case_numbers))

# def extract_governing_law(text):
#     # Define keywords for criminal law and civil law
#     criminal_law_keywords = [
#         'criminal', 'crime', 'offense', 'prosecution', 'defendant', 'felony', 'misdemeanor'
#     ]
#     civil_law_keywords = [
#         'civil', 'plaintiff', 'defendant', 'contract', 'tort', 'property', 'family', 'probate'
#     ]
    
#     # Tokenize the text and tag the parts of speech
#     tokens = word_tokenize(text.lower())
#     tagged_tokens = pos_tag(tokens)
    
#     # Extract nouns and adjectives from the tagged tokens
#     nouns_adjectives = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]
    
#     # Check for criminal law keywords
#     for keyword in criminal_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Criminal Law"
    
#     # Check for civil law keywords
#     for keyword in civil_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Civil Law"
    
#     return "Governing law not identified"

# def extract_final_verdict(text):
#     # Define regular expressions to match common patterns for final verdicts and dates
#     verdict_patterns = [
#         r'final\s*verdict[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'final\s*verdict[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         #r'DATED[:\s]*\d{2}\.\d{2}\.\d{4}',  # Pattern for DATED: dd.mm.yyyy
#     ]
    
#     # Search for the patterns in the text
#     final_verdict = None
#     for pattern in verdict_patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             final_verdict = match.group(0).strip()
#             break
    
#     if not final_verdict:
#         # If no final verdict date is found, try to find a date in the title of the case
#         title_pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
#         title_match = re.search(title_pattern, text, re.IGNORECASE)
#         if title_match:
#             final_verdict = title_match.group(2).strip()
    
#     return final_verdict

# # def extract_parties(text):
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Check for new_pattern_2_revised match
# #     match = new_pattern_2_revised.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # New patterns provided
# #     new_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_2 = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_3 = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     new_pattern_4 = re.compile(r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', re.IGNORECASE | re.DOTALL)
     
# #     # Additional new pattern to be added
# #     additional_new_pattern = re.compile(r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
# #     # New pattern to add (provided by user)
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     # Existing patterns
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.DOTALL | re.MULTILINE)
# #     pattern_ellipsis = re.compile(r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     pattern_dots = re.compile(r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondent\s*\(s\)', re.IGNORECASE | re.DOTALL)
# #     pattern_provided = re.compile(r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     pattern_appellant = re.compile(r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)

# #     # Try matching the specific pattern first
# #     specific_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = specific_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
     
# #     # Try matching the new pattern 6
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_6.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Try matching the new pattern next
# #     match = new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
 
# #     # Try matching the new pattern next
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_7.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Try matching the new pattern 2
# #     match = new_pattern_2.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }

# #     # Try matching the new pattern 3
# #     match = new_pattern_3.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     # Try matching the new pattern 4
# #     match = new_pattern_4.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

# #     # Try matching the additional new pattern
# #     match = additional_new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     # Try matching the new pattern 5 (provided by user)
# #     match = new_pattern_5.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

# #     # Try matching the new petitioner and respondents pattern
# #     petitioner_match = new_petitioner_pattern.search(text)
# #     respondents_match = new_respondents_pattern.search(text)

# #     if petitioner_match and respondents_match:
# #         petitioner = petitioner_match.group(1).strip()
# #         respondents_text = respondents_match.group(1).strip()
# #         respondents_list = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents_list if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"

# #     # Try matching the standard pattern
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the ellipses pattern
# #     matches = pattern_ellipsis.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the dots pattern
# #     matches = pattern_dots.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the provided specific format
# #     matches = pattern_provided.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the appellant pattern
# #     matches = pattern_appellant.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # Try matching the specific parties pattern
# #     matches = pattern_specific_parties.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     new_pattern_8 = re.compile(
# #           r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     ) 
# #     match = new_pattern_8.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('AND')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     return "Parties not found."


# #9/7/24
# # #continue tommorrow

# #as of now ok

# # def extract_parties(text):
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     petitioner_match = new_petitioner_pattern.search(text)
# #     respondents_match = new_respondents_pattern.search(text)

# #     if petitioner_match and respondents_match:
# #         petitioner = petitioner_match.group(1).strip()
# #         respondents_text = respondents_match.group(1).strip()
# #         respondents_list = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents_list if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_7.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_6.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Check for new_pattern_2_revised match
# #     match = new_pattern_2_revised.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     specific_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = specific_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     # New patterns provided
# #     new_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     new_pattern_2 = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_2.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('and')
# #         respondents = [resp.strip() for resp in respondents]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
    
# #     new_pattern_3 = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_3.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = respondents_text.split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents = "\n".join([f"{resp}" for resp in respondents])
# #         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
# #     new_pattern_4 = re.compile(r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', re.IGNORECASE | re.DOTALL)
# #     match = new_pattern_4.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
     
# #     # Additional new pattern to be added
# #     additional_new_pattern = re.compile(r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     match = additional_new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents_text = match.group(2).strip()
# #         respondents = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
# #         return {
# #             'petitioner': petitioner,
# #             'respondents': respondents
# #         }
  
# #     # New pattern to add (provided by user)
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )
# #     match = new_pattern_5.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondent = match.group(2).strip()
# #         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
# #     # Existing patterns
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.DOTALL | re.MULTILINE)
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_ellipsis = re.compile(r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_ellipsis.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_dots = re.compile(r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_dots.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_dashes = re.compile(r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_dashes.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_provided = re.compile(r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_provided.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_appellant = re.compile(r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_appellant.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties.findall(text)
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # new_pattern_8 = re.compile(
# #     #       r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     # ) 
# #     # match = new_pattern_8.search(text)
# #     # if match:
# #     #     petitioner = match.group(1).strip()
# #     #     respondents_text = match.group(2).strip()
# #     #     respondents = respondents_text.split('AND')
# #     #     respondents = [resp.strip() for resp in respondents]
# #     #     return {
# #     #         'petitioner': petitioner,
# #     #         'respondents': respondents
# #     #     }
    
# #     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties_1.findall(text) 
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             petitioner = petitioner.strip()
# #             respondent = respondent.strip()
# #             # Check if both petitioner and respondent are in uppercase
# #             if petitioner.isupper() and respondent.isupper():
# #                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
# #         # Join and return the result
# #         return "\n\n".join(parties)
    
# #     # If no patterns match, return "Parties not found."
# #     return "Parties not found."


# # # # # ##want to change every patterns to findall
# def extract_parties(text):            
#     # New patterns for petitioner and respondents
#     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
#     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

#     petitioner_matches = new_petitioner_pattern.findall(text)
#     respondents_matches = new_respondents_pattern.findall(text)

#     if petitioner_matches and respondents_matches:
#         petitioners = [match.strip() for match in petitioner_matches]
#         respondents = []
#         for match in respondents_matches:
#             respondents_list = match.strip().split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
#         petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
#         respondents = "\n".join([f"{respondent}" for respondent in respondents])
#         return f"Petitioners:\n{petitioners}\nRespondents:\n{respondents}"
       
#     new_pattern_7 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_7.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
    
    
#     new_pattern_6 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_6.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
     
    
#     new_pattern_2_revised = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_2_revised.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
#     specific_pattern = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = specific_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
#     new_pattern = re.compile(
#         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('and')
#             respondents.extend([resp.strip() for resp in respondents_list])
        
#         return {
#             'petitioners': petitioners,
#             'respondents': respondents
#         }
    
#     new_pattern_2 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_2.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('and')
#             respondents.extend([resp.strip() for resp in respondents_list])
        
#         return {
#             'petitioners': petitioners,
#             'respondents': respondents
#         }
    
#     new_pattern_3 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_3.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
    
    
#     new_pattern_4 = re.compile(
#         r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_4.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
     
#     additional_new_pattern = re.compile(
#         r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = additional_new_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents_text = match[1].strip()
#             respondents_list = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
#             petitioners.append(petitioner)
#             respondents.extend(respondents_list)
        
#         return {
#             'petitioners': petitioners,
#             'respondents': respondents
#         }
  
#     new_pattern_5 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_5.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
#     # Existing patterns
#     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
#     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
#     petitioners = petitioner_pattern.findall(text)
#     respondents = respondent_pattern.findall(text)
    
#     if petitioners and respondents:
#         petitioners = [p.strip() for p in petitioners]
#         respondents = [r.strip() for r in respondents]
#         parties = []
#         for petitioner, respondent in zip(petitioners, respondents):
#             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_11 = re.compile(
#         r'([^\n\r]+)\s*\.{3,}\s*Applicant\s*.*?VERSUS\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_11.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Petitioner:{petitioner}\nRespondents:{respondents_text}")
#         return "\n\n".join(parties)
    
#     pattern_12 = re.compile(
#         r'([^\n\r]+)\s*\.{1,}\s*Appellant\s*.*?Versus\s*([^\n\r]+)\s*\.{1,}\s*Respondent',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_12.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_17 = re.compile(
#         r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
#         re.IGNORECASE
#     )
    
#     matches = pattern_17.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_19 = re.compile(
#         r'([A-Z][A-Z\s&]+\.)\s*…\s*PETITIONERS\s*VERSUS\s*([A-Z][A-Z\s&]+\.)\s*…\s*RESPONDENTS',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_19.search(text)
#     if matches:
#         petitioner = matches.group(1).strip()
#         respondents = matches.group(2).strip().split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents_text = "\n".join(respondents)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_20 = re.compile(
#         r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     match = pattern_20.search(text)
#     if match:
#         appellants = match.group(1).strip()
#         respondents = match.group(2).strip()
#         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
    
#     pattern = re.compile(
#         r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
#         re.IGNORECASE
#     )
    
#     match = pattern.search(text)
#     if match:
#         appellants = match.group(1).strip()
#         respondents = match.group(2).strip()
#         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"

    
        
#     # pattern_9 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*Petitio\s*.*?Vs\.\s*([\s\S]*?)\s*\.{3,}\s*Respondent',re.IGNORECASE | re.DOTALL)

#     # matches = pattern_9.search(text)
#     # if matches:
#     #     petitioner = matches.group(1).strip()
#     #     respondent = matches.group(2).strip()
#     #     return f"Petitioners:\n{petitioner}\nRespondents:\n{respondent}"
    
#     #if pattern 9 activate means pattern 10 not working and most of the pattern is not work
    
#     pattern_10 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',re.IGNORECASE | re.DOTALL)
    
#     matches = pattern_10.search(text)
#     if matches:
#         petitioner = matches.group(1).strip()
#         respondents = matches.group(2).strip().split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents_text = "\n".join(respondents)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_13 = re.compile(
#         r'([^\n\r]+)\s*\.{2,}\s*Petitione\s*.*?vs\s*([\s\S]*?)\s*\.{2,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Search for the pattern in the text
#     match = pattern_13.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents = match.group(2).strip().split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents_text = "\n".join(respondents)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_14 = re.compile(
#         r'([^\n\r]+)\s*:::\s*Petitioner\s*-\s*Vs\s*-\s*([\s\S]+?)\s*:::\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Search for the pattern in the text
#     match = pattern_14.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents = match.group(2).strip().split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents_text = "\n".join(respondents)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_15 = re.compile(
#         r'([^\n\r]+)\s*\.{3,}\s*PETITIONER\s*AND\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT\S*',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Search for the pattern in the text
#     match = pattern_15.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents = match.group(2).strip().split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents_text = "\n".join(respondents)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_16 = re.compile(
#         r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*([\s\S]+?)\s*\.\.\s*Responden',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Search for the pattern in the text
#     match = pattern_16.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents = match.group(2).strip().split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents_text = "\n".join(respondents)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_18 = re.compile(
#         r'((?:\d+\.[A-Z][.\w\s]+\s*)+)\.\.Pet.*?vs\s+((?:\d+\..*?\.\s*)+)\.\.\s*Respo',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_18.search(text)
#     if matches:
#         petitioner = matches.group(1).strip()
#         respondents = matches.group(2).strip().split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents_text = "\n".join(respondents)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     # pattern = re.compile(
#     #     r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
#     #     re.IGNORECASE
#     # )
    
#     # match = pattern.search(text)
#     # if match:
#     #     appellants = match.group(1).strip()
#     #     respondents = match.group(2).strip()
#     #     return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"

    

        
#     #return "No matches found."

# #2    
#     pattern_ellipsis = re.compile(
#         r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_ellipsis.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_dots = re.compile(
#         r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dots.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
#     pattern_dashes = re.compile(
#         r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dashes.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_provided = re.compile(
#         r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_provided.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_appellant = re.compile(
#         r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_appellant.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
#     matches = pattern_specific_parties.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
      
#     #return "No matches found."
        
#     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
#     matches = pattern_specific_parties_1.findall(text) 
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioner = petitioner.strip()
#             respondent = respondent.strip()
#             # Check if both petitioner and respondent are in uppercase
#             if petitioner.isupper() and respondent.isupper():
#                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
#         # Join and return the result
#         return "\n\n".join(parties)
    
         
#     # new_pattern_8 = re.compile(
#     #       r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
#     # ) 
#     # match = new_pattern_8.search(text)
#     # if match:
#     #     petitioner = match.group(1).strip()
#     #     respondents_text = match.group(2).strip()
#     #     respondents = respondents_text.split('AND')
#     #     respondents = [resp.strip() for resp in respondents]
#     #     return {
#     #         'petitioner': petitioner,
#     #         'respondents': respondents
#     #     }
    
#     # If no patterns match, return "Parties not found."
#     return "Parties not found."

# # def extract_parties(text):
# #     pattern = re.compile(
# #         r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
# #         re.IGNORECASE
# #     )
    
# #     match = pattern.search(text)
# #     if match:
# #         appellants = match.group(1).strip()
# #         respondents = match.group(2).strip()
# #         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
# #     else:
# #         return "No match found."

# # def extract_parties(text):
# #     pattern = re.compile(
# #         r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     match = pattern.search(text)
# #     if match:
# #         appellants = match.group(1).strip()
# #         respondents = match.group(2).strip()
# #         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
# #     else:
# #         return "No match found."
    
# # def extract_parties(text):
# #     # Define the regex pattern
# #     pattern = re.compile(
# #         r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
# #         re.IGNORECASE
# #     )
    
# #     matches = pattern.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondent = match[1].strip()
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
# #     else:
# #         return "No match found."
    
# # def extract_parties(text):
# #     # Define the regex pattern
# #     pattern = re.compile(
# #         r'([^\n\r]+)\s*\.{3,}\s*PETITIONER\s*AND\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT\S*',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
# #     else:
# #         return "No match found."
   


     
# def extract_date(text):
#     # Define regex pattern to match dates in various formats
#     date_pattern = (
#         r'(\b(?:[12][0-9]|3[01]|0?[1-9])(st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s\d{4}\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}\b)'
#     )
    
#     # Find all matches of the date pattern in the text
#     matches = re.findall(date_pattern, text)
    
#     # Flatten the list of tuples and remove empty strings
#     dates = [date for match in matches for date in match if date]
    
#     # Remove duplicates by converting to set and back to list
#     unique_dates = list(set(dates))
    
#     return unique_dates

# def extract_case_title(text):
#     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
#     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/0-9]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
  
#     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
#     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-0-9]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-0-9]+))'
    
#     # First try matching the original pattern
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     # If no match, try the alternative pattern
#     if not match:
#         match = re.search(pattern_alt, text, re.IGNORECASE)
    
#     if match:
#         title = match.group(1).strip()
#         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
#             date = match.group(2).strip()
#             return f"{title} on {date}"
#         else:
#             # Check for unwanted "Author" or other text in the title
#             if "Author" in title:
#                 title = title.split("Author")[0].strip()
#             return title
#     else:
#         return "Title and date not found"

# def extract_court_name(text):
#     # Define a comprehensive pattern for court names, including spaces between letters
#     comprehensive_pattern = (
#         r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T\s*'
#         r'O\s*F\s*I\s*N\s*D\s*I\s*A|'
#         r'Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
#         r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
#     )
    
#     # Search for the comprehensive pattern in the text
#     match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
#     if match:
#         return match.group(0).strip()
#     else:
#         # Define a fallback pattern for court names, including spaces between letters
#         fallback_pattern = r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T|Supreme|High|District) Court'
        
#         # Search for the fallback pattern in the text
#         match = re.search(fallback_pattern, text, re.IGNORECASE)
        
#         if match:
#             return match.group(0).strip()
#         else:
#             return "Court name not found"

# def extract_articles_sections(text):
#     # Comprehensive pattern for articles
#     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Comprehensive pattern for sections
#     section_pattern = re.compile(r'\b(?:Section|Sec\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for clauses
#     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for sub-sections
#     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

#     unique_references = set()  # Using a set to remove duplicates

#     # Function to process matches
#     def process_matches(pattern, prefix):
#         for match in pattern.finditer(text):
#             reference = match.group().strip()
#             if prefix not in reference.lower():
#                 reference = f"{prefix} {reference}"
#             unique_references.add(reference)

#     # Processing all patterns
#     process_matches(article_pattern, "Article :")
#     process_matches(section_pattern, "Section :")
#     process_matches(clause_pattern, "Clause :")
#     process_matches(subsection_pattern, "Sub-section :")

#     if unique_references:
#         return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
#     else:
#         return "No articles, sections, clauses, or sub-sections found."  
    
# def sanitize_text(text):
#     # Remove unwanted symbols using regular expressions
#     sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)
#     return sanitized_text

# # Updated `resolve_coreferences` function
# def resolve_coreferences(text):
#     doc = nlp(text)
#     resolved_text = []
#     for token in doc:
#         if token.dep_ == 'pronoun':
#             antecedent = token.head.text
#             resolved_text.append(antecedent)
#         else:
#             resolved_text.append(token.text)
    
#     return ' '.join(resolved_text)

# # Function to preprocess text with coreference resolution
# def preprocess_text_with_coref_resolution(text):
#     text = resolve_coreferences(text)
#     text = preprocess_text(text)
#     return text

# def print_all_information(text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     print("Case Number:", extract_case_number(text))
#     print("Governing Law:", extract_governing_law(text))
#     print("Final Verdict:", extract_final_verdict(text))
#     print("Parties:", extract_parties(text))
#     print("Date:", extract_date(text))
#     print("Title of the Case:", extract_case_title(text))
#     print("Name of the Court:", extract_court_name(text))
#     print("Articles:", extract_articles_sections(text))
#     print("Summary of the Case:", generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors))

# # Main interaction loop
# try:
#     print("Bot: I will provide information about the legal document.")
#     print_all_information(raw_text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model)
# except KeyboardInterrupt:
#     print("\nBot: Thanks for talking, Bye!")




















# import re
# import nltk
# import contractions
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from autocorrect import Speller
# import emoji
# import regex 
# import gensim.downloader as api
# import fitz  # PyMuPDF
# from nltk import pos_tag
# import warnings
# import numpy as np

# warnings.simplefilter(action='ignore', category=UserWarning)

# #nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Load the English language model
# nlp = spacy.load('en_core_web_sm')

# # Set max_length to a value that accommodates your text length
# nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

# # Load pre-trained Word2Vec model
# w2v_model = api.load('word2vec-google-news-300')

# # Initialize spell checker
# spell = Speller()

# # Define stopword2Vec
# stop_words = set(stopwords.words('english'))


# # Preprocessing function with lemmatization, spell checking, and NER tagging
# def preprocess_text(text):
    
#     # Correct spelling errors
#     text = spell(text)
    
#     # Remove HTML tags, URLs, and special characters
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Expand contractions
#     text = contractions.fix(text)
    
#     # Remove citations
#     text = re.sub(r'\[[0-9]+\]', '', text)
    
#     # Remove extra whitespaces
#     text = re.sub(r'\s+', ' ', text)
    
#     # Tokenization and NER tagging using spaCy
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if token.ent_type_:
#             tokens.append(token.ent_type_)
#         else:
#             tokens.append(token.text)
    
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Remove repeated characters
#     tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]
    
#     # Remove single characters and numeric tokens
#     tokens = [word for word in tokens if len(word) > 1 and not word.isdigit()]
    
#     # Handle emojis
#     text = emoji.demojize(text)
#     text = text.replace(":", "")
    
#     # Handle emoticons
#     emoticons = regex.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = regex.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
#     text = text + ' '.join(emoticons)
    
#     # Join tokens back into a string
#     preprocessed_text = ' '.join(tokens)
    
#     return preprocessed_text

# def calculate_similarity(sentence1, sentence2):
#     # Tokenize the sentences
#     tokens1 = word_tokenize(sentence1)
#     tokens2 = word_tokenize(sentence2)
    
#     # Filter out stopwords
#     tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
#     tokens2 = [word for word in tokens2 if word.lower() not in stop_words]
    
#     # Get the Word2Vec vectors for each word
#     vectors1 = [w2v_model[word] for word in tokens1 if word in w2v_model]
#     vectors2 = [w2v_model[word] for word in tokens2 if word in w2v_model]

#     # Calculate the average vectors for each sentence
#     if vectors1 and vectors2:
#         avg_vector1 = np.mean(vectors1, axis=0)
#         avg_vector2 = np.mean(vectors2, axis=0)
        
#         # Calculate the cosine similarity between the average vectors
#         similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
#         return similarity
#     else:
#         return 0.0  # Return 0 if no vectors are found or all words are OOV

# # Keep track of previous questions and responses
# previous_questions = []
# previous_responses = []

# def generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, top_k=15, similarity_threshold=0.5):
#     global previous_questions, previous_responses

#     bot_response = ''
    
#     # Preprocess user input
#     processed_input = preprocess_text(user_input)
    
#     # Check if the processed input is empty or contains only stopwords
#     if not processed_input or all(word in stop_words for word in processed_input.split()):
#         return "I am sorry, I don't understand."
    
#     # Check if the current question is the same as a previous one
#     if processed_input in previous_questions:
#         index = previous_questions.index(processed_input)
#         return previous_responses[index]
    
#     # If not, continue with Word2Vec processing
#     similarities = []
#     for sent in sent_tokens:
#         similarity = calculate_similarity(processed_input, sent)
#         similarities.append(similarity)
    
#     # Convert similarities to a NumPy array for easier processing
#     similarities = np.array(similarities)
    
#     # Sort the similarities in descending order
#     sorted_indices = np.argsort(similarities)[::-1]
    
#     # Find the top k most similar sentences that are not in previous responses
#     top_k_sentences = []
#     for index in sorted_indices:
#         if len(top_k_sentences) < top_k and similarities[index] >= similarity_threshold:
#             top_k_sentences.append(sent_tokens[index])
    
#     # Assign the top k sentences to bot_response
#     if top_k_sentences:
#         bot_response = '\n'.join(top_k_sentences)
#         # Filter out URLs and unwanted tags from the response
#         bot_response = re.sub(r'http\S+', '', bot_response)  # Remove URLs
        
#         # Track previous questions and responses
#         previous_questions.append(processed_input)
#         previous_responses.append(bot_response)
#     else:
#         bot_response = "I am sorry, I don't get enough details."
    
#     return bot_response  # Return the top k sentences separated by newline characters

# # Read text from the local PDF file using PyMuPDF
# file_path = 'C:/legal keywords summarization/type 2 case documents/Vishaka_Ors_vs_State_Of_Rajasthan_Ors_on_13_August_1997.PDF'
# try:
#     pdf_document = fitz.open(file_path)
#     raw_text = ''
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         raw_text += page.get_text()
#     pdf_document.close()
#     print("PDF file read successfully.")
# except Exception as e:
#     print("Error reading PDF file:", e)
#     exit()

# # Tokenize the raw text to obtain sentence tokens
# sent_tokens = nltk.sent_tokenize(raw_text)

# # Preprocess the text
# preprocessed_sent_tokens = [preprocess_text(sent) for sent in sent_tokens]

# # Define TF-IDF vectorizer with optimized parameters and TF-IDF weighting schemes
# word_vectorizer = TfidfVectorizer(
#     tokenizer=word_tokenize,     # Use NLTK's word tokenizer
#     stop_words='english',        # Use English stopwords
#     ngram_range=(1, 3),          # Use unigrams and bigrams
#     max_features=15000,           # Limit the vocabulary size to the top 5000 features
#     token_pattern=r'\b\w+\b',    # Use words as tokens
#     sublinear_tf=True,           # Apply sublinear term frequency scaling
#     smooth_idf=True,             # Smooth IDF weights by adding one to document frequencies
#     norm='l2'                    # Normalize TF-IDF vectors to unit length
# )

# # Apply TF-IDF vectorization on the preprocessed text
# word_vectors = word_vectorizer.fit_transform(preprocessed_sent_tokens)


# # Generate response function
# def generate_response(user_input, text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     # Check if the user is asking for specific information
#     if 'case number' in user_input or 'case no' in user_input:
#         return extract_case_number(text)
#     elif 'governing law' in user_input:
#         return extract_governing_law(text)
#     elif 'final verdict' in user_input:
#         return extract_final_verdict(text)
#     elif 'party' in user_input:
#         return extract_parties(text)
#     elif 'date' in user_input:
#         return extract_date(text)
#     elif 'title of the case' in user_input or 'case title' in user_input:
#         return extract_case_title(text)
#     elif 'summary of the case' in user_input or 'case summary' in user_input:
#         return generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
#     elif 'name of the court' in user_input or 'court name' in user_input:
#         return extract_court_name(text)
#     elif 'article' in user_input:
#         return extract_articles_sections(text)
#     else:
#         # Handle other types of questions
#         return "I'm sorry, I don't know."
    
# def extract_case_number(text):
#     # Regular expression pattern for matching case numbers
#     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?\d{1,}\s*(?:of|OF)\s*\d{4}\b'
    
#     # Find all matches of the pattern in the text
#     case_numbers = re.findall(pattern, text)
    
#     # Return a list of unique case numbers
#     return list(set(case_numbers))

# def extract_governing_law(text):
#     # Define keywords for criminal law and civil law
#     criminal_law_keywords = [
#         'criminal', 'crime', 'offense', 'prosecution', 'defendant', 'felony', 'misdemeanor'
#     ]
#     civil_law_keywords = [
#         'civil', 'plaintiff', 'defendant', 'contract', 'tort', 'property', 'family', 'probate'
#     ]
    
#     # Tokenize the text and tag the parts of speech
#     tokens = word_tokenize(text.lower())
#     tagged_tokens = pos_tag(tokens)
    
#     # Extract nouns and adjectives from the tagged tokens
#     nouns_adjectives = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]
    
#     # Check for criminal law keywords
#     for keyword in criminal_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Criminal Law"
    
#     # Check for civil law keywords
#     for keyword in civil_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Civil Law"
    
#     return "Governing law not identified"

# def extract_final_verdict(text):
#     # Define regular expressions to match common patterns for final verdicts and dates
#     verdict_patterns = [
#         r'final\s*verdict[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'final\s*verdict[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         #r'DATED[:\s]*\d{2}\.\d{2}\.\d{4}',  # Pattern for DATED: dd.mm.yyyy
#     ]
    
#     # Search for the patterns in the text
#     final_verdict = None
#     for pattern in verdict_patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             final_verdict = match.group(0).strip()
#             break
    
#     if not final_verdict:
#         # If no final verdict date is found, try to find a date in the title of the case
#         title_pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
#         title_match = re.search(title_pattern, text, re.IGNORECASE)
#         if title_match:
#             final_verdict = title_match.group(2).strip()
    
#     return final_verdict


# # # # # # ##want to change every patterns to findall
# # def extract_parties(text):            
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

# #     petitioner_matches = new_petitioner_pattern.findall(text)
# #     respondents_matches = new_respondents_pattern.findall(text)

# #     if petitioner_matches and respondents_matches:
# #         petitioners = [match.strip() for match in petitioner_matches]
# #         respondents = []
# #         for match in respondents_matches:
# #             respondents_list = match.strip().split('\n')
# #             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
# #         petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
# #         respondents = "\n".join([f"{respondent}" for respondent in respondents])
# #         return f"Petitioners:\n{petitioners}\nRespondents:\n{respondents}"
       
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_7.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
    
    
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_6.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
     
    
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_2_revised.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
# #     specific_pattern = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = specific_pattern.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents_text = match[1].strip()
# #             respondents_list = respondents_text.split('\n')
# #             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
# #     new_pattern = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents_text = match[1].strip()
# #             respondents_list = respondents_text.split('and')
# #             respondents.extend([resp.strip() for resp in respondents_list])
        
# #         return {
# #             'petitioners': petitioners,
# #             'respondents': respondents
# #         }
    
# #     new_pattern_2 = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_2.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents_text = match[1].strip()
# #             respondents_list = respondents_text.split('and')
# #             respondents.extend([resp.strip() for resp in respondents_list])
        
# #         return {
# #             'petitioners': petitioners,
# #             'respondents': respondents
# #         }
    
# #     new_pattern_3 = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_3.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents_text = match[1].strip()
# #             respondents_list = respondents_text.split('\n')
# #             respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
    
    
# #     new_pattern_4 = re.compile(
# #         r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_4.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
     
# #     additional_new_pattern = re.compile(
# #         r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = additional_new_pattern.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents_text = match[1].strip()
# #             respondents_list = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
# #             petitioners.append(petitioner)
# #             respondents.extend(respondents_list)
        
# #         return {
# #             'petitioners': petitioners,
# #             'respondents': respondents
# #         }
  
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_5.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
# #     # Existing patterns
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_11 = re.compile(
# #         r'([^\n\r]+)\s*\.{3,}\s*Applicant\s*.*?VERSUS\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_11.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
# #             parties.append(f"Petitioner:{petitioner}\nRespondents:{respondents_text}")
# #         return "\n\n".join(parties)
    
# #     pattern_12 = re.compile(
# #         r'([^\n\r]+)\s*\.{1,}\s*Appellant\s*.*?Versus\s*([^\n\r]+)\s*\.{1,}\s*Respondent',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_12.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondent = match[1].strip()
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_17 = re.compile(
# #         r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
# #         re.IGNORECASE
# #     )
    
# #     matches = pattern_17.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondent = match[1].strip()
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_19 = re.compile(
# #         r'([A-Z][A-Z\s&]+\.)\s*…\s*PETITIONERS\s*VERSUS\s*([A-Z][A-Z\s&]+\.)\s*…\s*RESPONDENTS',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_19.search(text)
# #     if matches:
# #         petitioner = matches.group(1).strip()
# #         respondents = matches.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_20 = re.compile(
# #         r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     match = pattern_20.search(text)
# #     if match:
# #         appellants = match.group(1).strip()
# #         respondents = match.group(2).strip()
# #         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
    
# #     pattern_21 = re.compile(
# #         r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
# #         re.IGNORECASE
# #     )
    
# #     match = pattern_21.search(text)
# #     if match:
# #         appellants = match.group(1).strip()
# #         respondents = match.group(2).strip()
# #         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"

    
        
# #     # pattern_9 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*Petitio\s*.*?Vs\.\s*([\s\S]*?)\s*\.{3,}\s*Respondent',re.IGNORECASE | re.DOTALL)

# #     # matches = pattern_9.search(text)
# #     # if matches:
# #     #     petitioner = matches.group(1).strip()
# #     #     respondent = matches.group(2).strip()
# #     #     return f"Petitioners:\n{petitioner}\nRespondents:\n{respondent}"
    
# #     #if pattern 9 activate means pattern 10 not working and most of the pattern is not work
    
# #     pattern_10 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',re.IGNORECASE | re.DOTALL)
    
# #     matches = pattern_10.search(text)
# #     if matches:
# #         petitioner = matches.group(1).strip()
# #         respondents = matches.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_13 = re.compile(
# #         r'([^\n\r]+)\s*\.{2,}\s*Petitione\s*.*?vs\s*([\s\S]*?)\s*\.{2,}\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = pattern_13.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_14 = re.compile(
# #         r'([^\n\r]+)\s*:::\s*Petitioner\s*-\s*Vs\s*-\s*([\s\S]+?)\s*:::\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = pattern_14.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_15 = re.compile(
# #         r'([^\n\r]+)\s*\.{3,}\s*PETITIONER\s*AND\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT\S*',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = pattern_15.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_16 = re.compile(
# #         r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*([\s\S]+?)\s*\.\.\s*Responden',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = pattern_16.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_18 = re.compile(
# #         r'((?:\d+\.[A-Z][.\w\s]+\s*)+)\.\.Pet.*?vs\s+((?:\d+\..*?\.\s*)+)\.\.\s*Respo',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_18.search(text)
# #     if matches:
# #         petitioner = matches.group(1).strip()
# #         respondents = matches.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"

# # #2    
# #     pattern_ellipsis = re.compile(
# #         r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_ellipsis.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_dots = re.compile(
# #         r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_dots.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
    
# #     pattern_dashes = re.compile(
# #         r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_dashes.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_provided = re.compile(
# #         r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_provided.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_appellant = re.compile(
# #         r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_appellant.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
# #     matches = pattern_specific_parties.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
           
# #     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties_1.findall(text) 
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             petitioner = petitioner.strip()
# #             respondent = respondent.strip()
# #             # Check if both petitioner and respondent are in uppercase
# #             if petitioner.isupper() and respondent.isupper():
# #                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
# #         # Join and return the result
# #         return "\n\n".join(parties)
    
         
# #     # new_pattern_8 = re.compile(
# #     #       r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     # ) 
# #     # match = new_pattern_8.search(text)
# #     # if match:
# #     #     petitioner = match.group(1).strip()
# #     #     respondents_text = match.group(2).strip()
# #     #     respondents = respondents_text.split('AND')
# #     #     respondents = [resp.strip() for resp in respondents]
# #     #     return {
# #     #         'petitioner': petitioner,
# #     #         'respondents': respondents
# #     #     }
    
# #     # If no patterns match, return "Parties not found."
# #     return "Parties not found."


# ###---------------------------------------------------------------------------------------------------------
# #type 2

# # def extract_parties(text):            
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

# #     petitioner_matches = new_petitioner_pattern.findall(text)
# #     respondents_matches = new_respondents_pattern.findall(text)

# #     if petitioner_matches and respondents_matches:
# #         petitioners = [match.strip() for match in petitioner_matches]
# #         respondents = []
# #         for match in respondents_matches:
# #             respondents_list = match.strip().split('\n')
# #             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
# #         petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
# #         respondents = "\n".join([f"{respondent}" for respondent in respondents])
# #         return f"Petitioners:\n{petitioners}\nRespondents:\n{respondents}"
       
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_7.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
    
    
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_6.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
     
    
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_2_revised.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
# #     specific_pattern = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = specific_pattern.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents_text = match[1].strip()
# #             respondents_list = respondents_text.split('\n')
# #             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
# #     new_pattern = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     new_pattern_2 = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = new_pattern_2.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     new_pattern_3 = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_3.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents_text = match[1].strip()
# #             respondents_list = respondents_text.split('\n')
# #             respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
    
    
# #     new_pattern_4 = re.compile(
# #         r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_4.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
     
# #     additional_new_pattern = re.compile(
# #         r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = additional_new_pattern.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
  
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_5.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\nRespondents:\n{respondents_text}"
    
# #     # Existing patterns
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_11 = re.compile(
# #         r'([^\n\r]+)\s*\.{3,}\s*Applicant\s*.*?VERSUS\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_11.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
# #             parties.append(f"Petitioner:{petitioner}\nRespondents:{respondents_text}")
# #         return "\n\n".join(parties)
    
# #     pattern_12 = re.compile(
# #         r'([^\n\r]+)\s*\.{1,}\s*Appellant\s*.*?Versus\s*([^\n\r]+)\s*\.{1,}\s*Respondent',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_12.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondent = match[1].strip()
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_17 = re.compile(
# #         r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
# #         re.IGNORECASE
# #     )
    
# #     matches = pattern_17.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondent = match[1].strip()
# #             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_19 = re.compile(
# #         r'([A-Z][A-Z\s&]+\.)\s*…\s*PETITIONERS\s*VERSUS\s*([A-Z][A-Z\s&]+\.)\s*…\s*RESPONDENTS',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_19.search(text)
# #     if matches:
# #         petitioner = matches.group(1).strip()
# #         respondents = matches.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_20 = re.compile(
# #         r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     match = pattern_20.search(text)
# #     if match:
# #         appellants = match.group(1).strip()
# #         respondents = match.group(2).strip()
# #         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
    
# #     pattern_21 = re.compile(
# #         r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
# #         re.IGNORECASE
# #     )
    
# #     match = pattern_21.search(text)
# #     if match:
# #         appellants = match.group(1).strip()
# #         respondents = match.group(2).strip()
# #         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"

    
        
# #     # pattern_9 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*Petitio\s*.*?Vs\.\s*([\s\S]*?)\s*\.{3,}\s*Respondent',re.IGNORECASE | re.DOTALL)

# #     # matches = pattern_9.search(text)
# #     # if matches:
# #     #     petitioner = matches.group(1).strip()
# #     #     respondent = matches.group(2).strip()
# #     #     return f"Petitioners:\n{petitioner}\nRespondents:\n{respondent}"
    
# #     #if pattern 9 activate means pattern 10 not working and most of the pattern is not work
    
# #     pattern_10 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',re.IGNORECASE | re.DOTALL)
    
# #     matches = pattern_10.search(text)
# #     if matches:
# #         petitioner = matches.group(1).strip()
# #         respondents = matches.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_13 = re.compile(
# #         r'([^\n\r]+)\s*\.{2,}\s*Petitione\s*.*?vs\s*([\s\S]*?)\s*\.{2,}\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = pattern_13.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_14 = re.compile(
# #         r'([^\n\r]+)\s*:::\s*Petitioner\s*-\s*Vs\s*-\s*([\s\S]+?)\s*:::\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = pattern_14.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_15 = re.compile(
# #         r'([^\n\r]+)\s*\.{3,}\s*PETITIONER\s*AND\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT\S*',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = pattern_15.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_16 = re.compile(
# #         r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*([\s\S]+?)\s*\.\.\s*Responden',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Search for the pattern in the text
# #     match = pattern_16.search(text)
# #     if match:
# #         petitioner = match.group(1).strip()
# #         respondents = match.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_18 = re.compile(
# #         r'((?:\d+\.[A-Z][.\w\s]+\s*)+)\.\.Pet.*?vs\s+((?:\d+\..*?\.\s*)+)\.\.\s*Respo',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_18.search(text)
# #     if matches:
# #         petitioner = matches.group(1).strip()
# #         respondents = matches.group(2).strip().split('\n')
# #         respondents = [resp.strip() for resp in respondents if resp.strip()]
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"

# # #2    
# #     pattern_ellipsis = re.compile(
# #         r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_ellipsis.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_dots = re.compile(
# #         r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_dots.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
    
# #     pattern_dashes = re.compile(
# #         r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_dashes.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_provided = re.compile(
# #         r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_provided.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_appellant = re.compile(
# #         r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_appellant.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
# #     matches = pattern_specific_parties.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
           
# #     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties_1.findall(text) 
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             petitioner = petitioner.strip()
# #             respondent = respondent.strip()
# #             # Check if both petitioner and respondent are in uppercase
# #             if petitioner.isupper() and respondent.isupper():
# #                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
# #         # Join and return the result
# #         return "\n\n".join(parties)
    
         
# #     # new_pattern_8 = re.compile(
# #     #       r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     # ) 
# #     # match = new_pattern_8.search(text)
# #     # if match:
# #     #     petitioner = match.group(1).strip()
# #     #     respondents_text = match.group(2).strip()
# #     #     respondents = respondents_text.split('AND')
# #     #     respondents = [resp.strip() for resp in respondents]
# #     #     return {
# #     #         'petitioner': petitioner,
# #     #         'respondents': respondents
# #     #     }
    
# #     # If no patterns match, return "Parties not found."
# #     return "Parties not found."


# ###----------------------------------------------------------------------------------
# # #type 3
# def extract_parties(text):
#     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
#     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
#     petitioners = petitioner_pattern.findall(text)
#     respondents = respondent_pattern.findall(text)
    
#     if petitioners and respondents:
#         petitioners = [p.strip() for p in petitioners]
#         respondents = [r.strip() for r in respondents]
#         parties = []
#         for petitioner, respondent in zip(petitioners, respondents):
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
         
#     # New patterns for petitioner and respondents
#     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
#     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

#     petitioner_matches = new_petitioner_pattern.findall(text)
#     respondents_matches = new_respondents_pattern.findall(text)

#     if petitioner_matches and respondents_matches:
#         petitioners = [match.strip() for match in petitioner_matches]
#         respondents = []
#         for match in respondents_matches:
#             respondents_list = match.strip().split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
#         petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
#         respondents = "\n".join([f"{respondent}" for respondent in respondents])
#         return f"Petitioners:\n{petitioners}\n\nRespondents:\n{respondents}"
       
#     new_pattern_7 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_7.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
    
#     new_pattern_6 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_6.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
    
#     new_pattern_2_revised = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_2_revised.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     specific_pattern = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = specific_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     new_pattern = re.compile(
#     r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = new_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
   
#     new_pattern_2 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = new_pattern_2.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     new_pattern_3 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_3.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
    
#     new_pattern_4 = re.compile(
#         r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_4.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
#     additional_new_pattern = re.compile(
#         r'\[.*?\]\s*(.*?)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = additional_new_pattern.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
            
#             # Extract only the relevant petitioner information and remove leading spaces
#             petitioner_parts = petitioner.split('\n')
#             relevant_petitioner = '\n'.join(part.strip() for part in petitioner_parts if '.' not in part[:3])
            
#             petitioners_list.append(relevant_petitioner)
#             respondents_list.append("\n".join(respondents))
        
#         petitioners_text = "\n\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    

  
#     new_pattern_5 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_5.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     # Existing patterns
#     # petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
#     # respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
#     # petitioners = petitioner_pattern.findall(text)
#     # respondents = respondent_pattern.findall(text)
    
#     # if petitioners and respondents:
#     #     petitioners = [p.strip() for p in petitioners]
#     #     respondents = [r.strip() for r in respondents]
#     #     parties = []
#     #     for petitioner, respondent in zip(petitioners, respondents):
#     #         parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#     #     return "\n\n".join(parties)
    
#     pattern_11_1 = re.compile(
#         r'\[.*?\]\s*(.*?)\s*\.{3,}\s*Applicant\s*.*?Versus\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_11_1.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             # Remove leading/trailing spaces and align to left
#             applicant_lines = [line.strip() for line in match[0].split('\n') if line.strip()]
#             applicant = '\n'.join(applicant_lines)
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Applicant:\n{applicant}\n\nRespondents:\n{respondents_text}")
#         return "\n\n".join(parties)
    
    
#     pattern_11 = re.compile(
#         r'([^\n\r]+)\s*\.{3,}\s*Applicant\s*.*?VERSUS\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_11.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Petitioner:{petitioner}\n\nRespondents:{respondents_text}")
#         return "\n\n".join(parties)
    
    
    
#     pattern_12 = re.compile(
#         r'([^\n\r]+)\s*\.{1,}\s*Appellant\s*.*?Versus\s*([^\n\r]+)\s*\.{1,}\s*Respondent',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_12.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_17 = re.compile(
#         r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
#         re.IGNORECASE
#     )
    
#     matches = pattern_17.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_19 = re.compile(
#     r'([A-Z][A-Z\s&]+\.)\s*…\s*PETITIONERS\s*VERSUS\s*([A-Z][A-Z\s&]+\.)\s*…\s*RESPONDENTS',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_19.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_20 = re.compile(
#     r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_20.findall(text)
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip()
#             respondents = match[1].strip()
            
#             appellants_list.append(appellants)
#             respondents_list.append(respondents)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
#         return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents}"
    
#     pattern_21 = re.compile(
#     r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
#     re.IGNORECASE
#     )
    
#     # Find all matches in the text
#     matches = pattern_21.findall(text)
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip()
#             respondents = match[1].strip()
            
#             appellants_list.append(appellants)
#             respondents_list.append(respondents)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
#         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"

    
        
#     # pattern_9 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*Petitio\s*.*?Vs\.\s*([\s\S]*?)\s*\.{3,}\s*Respondent',re.IGNORECASE | re.DOTALL)

#     # matches = pattern_9.search(text)
#     # if matches:
#     #     petitioner = matches.group(1).strip()
#     #     respondent = matches.group(2).strip()
#     #     return f"Petitioners:\n{petitioner}\nRespondents:\n{respondent}"
    
#     #if pattern 9 activate means pattern 10 not working and most of the pattern is not work
    
#     pattern_10 = re.compile(
#     r'([^\n\r]+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_10.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"

    
#     pattern_13 = re.compile(
#     r'([^\n\r]+)\s*\.{2,}\s*Petitione\s*.*?vs\s*([\s\S]*?)\s*\.{2,}\s*Respondents',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_13.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_14 = re.compile(
#         r'(?:.*?WP\(C\)\sNo\.\s\d+\sof\s\d+\s+Date\sof\sDecision:.*?\n)(.*?)\s*:::\s*Petitioner\s*-\s*Vs\s*-\s*([\s\S]+?)\s*:::\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_14.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             # Split petitioner info into lines, strip each line, and rejoin
#             petitioner_lines = match[0].strip().split('\n')
#             petitioner = '\n'.join(line.strip() for line in petitioner_lines if line.strip())
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"

    
#     pattern_15 = re.compile(
#     r'([^\n\r]+)\s*\.{3,}\s*PETITIONER\s*AND\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT\S*',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_15.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_16 = re.compile(
#     r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*([\s\S]+?)\s*\.\.\s*Responden',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_16.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_18 = re.compile(
#     r'((?:\d+\.[A-Z][.\w\s]+\s*)+)\.\.Pet.*?vs\s+((?:\d+\..*?\.\s*)+)\.\.\s*Respo',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_18.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"

# #2    
#     pattern_ellipsis = re.compile(
#         r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_ellipsis.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_dots = re.compile(
#         r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dots.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\n\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
#     pattern_dashes = re.compile(
#         r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dashes.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_provided = re.compile(
#         r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_provided.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_appellant = re.compile(
#         r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_appellant.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
#     matches = pattern_specific_parties.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
           
#     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
#     matches = pattern_specific_parties_1.findall(text) 
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioner = petitioner.strip()
#             respondent = respondent.strip()
#             # Check if both petitioner and respondent are in uppercase
#             if petitioner.isupper() and respondent.isupper():
#                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
#         # Join and return the result
#         return "\n\n".join(parties)
    
         
#     # new_pattern_8 = re.compile(
#     #       r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
#     # ) 
#     # match = new_pattern_8.search(text)
#     # if match:
#     #     petitioner = match.group(1).strip()
#     #     respondents_text = match.group(2).strip()
#     #     respondents = respondents_text.split('AND')
#     #     respondents = [resp.strip() for resp in respondents]
#     #     return {
#     #         'petitioner': petitioner,
#     #         'respondents': respondents
#     #     }
    
#     # If no patterns match, return "Parties not found."
#     return "Parties not found."





# # def extract_parties(text):
        
# #     pattern_15 = re.compile(
# #     r'\.{3,}\s*PETITIONER\s*AND\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT\S*',

# #     re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_15.findall(text)
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"

# def extract_date(text):
#     # Define regex pattern to match dates in various formats
#     date_pattern = (
#         r'(\b(?:[12][0-9]|3[01]|0?[1-9])(st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s\d{4}\b)|'
#         r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s(?:[12][0-9]|3[01]),?\s\d{4}\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}\b)'
#     )
    
#     # Find all matches of the date pattern in the text
#     matches = re.findall(date_pattern, text)
    
#     # Flatten the list of tuples and remove empty strings
#     dates = [date for match in matches for date in match if date]
    
#     # Remove duplicates by converting to set and back to list
#     unique_dates = list(set(dates))
    
#     return unique_dates
    
# # def extract_date(text):
# #     # Define regex pattern to match dates in various formats
# #     date_pattern = (
# #         r'(\b(?:[12][0-9]|3[01]|0?[1-9])(st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s\d{4}\b)|'
# #         r'(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b)|'
# #         r'(\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}\b)'
# #     )
    
# #     # Find all matches of the date pattern in the text
# #     matches = re.findall(date_pattern, text)
    
# #     # Flatten the list of tuples and remove empty strings
# #     dates = [date for match in matches for date in match if date]
    
# #     # Remove duplicates by converting to set and back to list
# #     unique_dates = list(set(dates))
    
# #     return unique_dates

# def extract_case_title(text):
#     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
#     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/0-9]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
  
#     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
#     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-0-9]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-0-9]+))'
    
#     # First try matching the original pattern
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     # If no match, try the alternative pattern
#     if not match:
#         match = re.search(pattern_alt, text, re.IGNORECASE)
    
#     if match:
#         title = match.group(1).strip()
#         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
#             date = match.group(2).strip()
#             return f"{title} on {date}"
#         else:
#             # Check for unwanted "Author" or other text in the title
#             if "Author" in title:
#                 title = title.split("Author")[0].strip()
#             return title
#     else:
#         return "Title and date not found"

# def extract_court_name(text):
#     # Define a comprehensive pattern for court names, including spaces between letters
#     comprehensive_pattern = (
#         r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T\s*'
#         r'O\s*F\s*I\s*N\s*D\s*I\s*A|'
#         r'Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
#         r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
#     )
    
#     # Search for the comprehensive pattern in the text
#     match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
#     if match:
#         return match.group(0).strip()
#     else:
#         # Define a fallback pattern for court names, including spaces between letters
#         fallback_pattern = r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T|Supreme|High|District) Court'
        
#         # Search for the fallback pattern in the text
#         match = re.search(fallback_pattern, text, re.IGNORECASE)
        
#         if match:
#             return match.group(0).strip()
#         else:
#             return "Court name not found"

# def extract_articles_sections(text):
#     # Comprehensive pattern for articles
#     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Comprehensive pattern for sections
#     section_pattern = re.compile(r'\b(?:Section|Sec\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for clauses
#     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for sub-sections
#     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

#     unique_references = set()  # Using a set to remove duplicates

#     # Function to process matches
#     def process_matches(pattern, prefix):
#         for match in pattern.finditer(text):
#             reference = match.group().strip()
#             if prefix not in reference.lower():
#                 reference = f"{prefix} {reference}"
#             unique_references.add(reference)

#     # Processing all patterns
#     process_matches(article_pattern, "Article :")
#     process_matches(section_pattern, "Section :")
#     process_matches(clause_pattern, "Clause :")
#     process_matches(subsection_pattern, "Sub-section :")

#     if unique_references:
#         return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
#     else:
#         return "No articles, sections, clauses, or sub-sections found."  
    
# def sanitize_text(text):
#     # Remove unwanted symbols using regular expressions
#     sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)
#     return sanitized_text

# # Updated `resolve_coreferences` function
# def resolve_coreferences(text):
#     doc = nlp(text)
#     resolved_text = []
#     for token in doc:
#         if token.dep_ == 'pronoun':
#             antecedent = token.head.text
#             resolved_text.append(antecedent)
#         else:
#             resolved_text.append(token.text)
    
#     return ' '.join(resolved_text)

# # Function to preprocess text with coreference resolution
# def preprocess_text_with_coref_resolution(text):
#     text = resolve_coreferences(text)
#     text = preprocess_text(text)
#     return text

# def print_all_information(text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     print("Case Number:", extract_case_number(text))
#     print("Governing Law:", extract_governing_law(text))
#     print("Final Verdict:", extract_final_verdict(text))
#     print("Parties:", extract_parties(text))
#     print("Date:", extract_date(text))
#     print("Title of the Case:", extract_case_title(text))
#     print("Name of the Court:", extract_court_name(text))
#     print("Articles:", extract_articles_sections(text))
#     print("Summary of the Case:", generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors))

# # Main interaction loop
# try:
#     print("Bot: I will provide information about the legal document.")
#     print_all_information(raw_text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model)
# except KeyboardInterrupt:
#     print("\nBot: Thanks for talking, Bye!")






































# import re
# import nltk
# import contractions
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from autocorrect import Speller
# import emoji
# import regex 
# import gensim.downloader as api
# import fitz  # PyMuPDF
# from nltk import pos_tag
# import warnings
# import numpy as np

# warnings.simplefilter(action='ignore', category=UserWarning)

# #nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Load the English language model
# nlp = spacy.load('en_core_web_sm')

# # Set max_length to a value that accommodates your text length
# nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

# # Load pre-trained Word2Vec model
# w2v_model = api.load('word2vec-google-news-300')

# # Initialize spell checker
# spell = Speller()

# # Define stopword2Vec
# stop_words = set(stopwords.words('english'))


# # Preprocessing function with lemmatization, spell checking, and NER tagging
# def preprocess_text(text):
    
#     # Correct spelling errors
#     text = spell(text)
    
#     # Remove HTML tags, URLs, and special characters
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Expand contractions
#     text = contractions.fix(text)
    
#     # Remove citations
#     text = re.sub(r'\[[0-9]+\]', '', text)
    
#     # Remove extra whitespaces
#     text = re.sub(r'\s+', ' ', text)
    
#     # Tokenization and NER tagging using spaCy
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if token.ent_type_:
#             tokens.append(token.ent_type_)
#         else:
#             tokens.append(token.text)
    
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Remove repeated characters
#     tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]
    
#     # Remove single characters and numeric tokens
#     tokens = [word for word in tokens if len(word) > 1 and not word.isdigit()]
    
#     # Handle emojis
#     text = emoji.demojize(text)
#     text = text.replace(":", "")
    
#     # Handle emoticons
#     emoticons = regex.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = regex.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
#     text = text + ' '.join(emoticons)
    
#     # Join tokens back into a string
#     preprocessed_text = ' '.join(tokens)
    
#     return preprocessed_text

# def calculate_similarity(sentence1, sentence2):
#     # Tokenize the sentences
#     tokens1 = word_tokenize(sentence1)
#     tokens2 = word_tokenize(sentence2)
    
#     # Filter out stopwords
#     tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
#     tokens2 = [word for word in tokens2 if word.lower() not in stop_words]
    
#     # Get the Word2Vec vectors for each word
#     vectors1 = [w2v_model[word] for word in tokens1 if word in w2v_model]
#     vectors2 = [w2v_model[word] for word in tokens2 if word in w2v_model]

#     # Calculate the average vectors for each sentence
#     if vectors1 and vectors2:
#         avg_vector1 = np.mean(vectors1, axis=0)
#         avg_vector2 = np.mean(vectors2, axis=0)
        
#         # Calculate the cosine similarity between the average vectors
#         similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
#         return similarity
#     else:
#         return 0.0  # Return 0 if no vectors are found or all words are OOV

# # Keep track of previous questions and responses
# previous_questions = []
# previous_responses = []

# def generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, top_k=15, similarity_threshold=0.5):
#     global previous_questions, previous_responses

#     bot_response = ''
    
#     # Preprocess user input
#     processed_input = preprocess_text(user_input)
    
#     # Check if the processed input is empty or contains only stopwords
#     if not processed_input or all(word in stop_words for word in processed_input.split()):
#         return "I am sorry, I don't understand."
    
#     # Check if the current question is the same as a previous one
#     if processed_input in previous_questions:
#         index = previous_questions.index(processed_input)
#         return previous_responses[index]
    
#     # If not, continue with Word2Vec processing
#     similarities = []
#     for sent in sent_tokens:
#         similarity = calculate_similarity(processed_input, sent)
#         similarities.append(similarity)
    
#     # Convert similarities to a NumPy array for easier processing
#     similarities = np.array(similarities)
    
#     # Sort the similarities in descending order
#     sorted_indices = np.argsort(similarities)[::-1]
    
#     # Find the top k most similar sentences that are not in previous responses
#     top_k_sentences = []
#     for index in sorted_indices:
#         if len(top_k_sentences) < top_k and similarities[index] >= similarity_threshold:
#             top_k_sentences.append(sent_tokens[index])
    
#     # Assign the top k sentences to bot_response
#     if top_k_sentences:
#         bot_response = '\n'.join(top_k_sentences)
#         # Filter out URLs and unwanted tags from the response
#         bot_response = re.sub(r'http\S+', '', bot_response)  # Remove URLs
        
#         # Track previous questions and responses
#         previous_questions.append(processed_input)
#         previous_responses.append(bot_response)
#     else:
#         bot_response = "I am sorry, I don't get enough details."
    
#     return bot_response  # Return the top k sentences separated by newline characters

# # Read text from the local PDF file using PyMuPDF
# file_path = 'C:/Users/Viswajith/Downloads/Sitharthan_vs_The_Deputy_Inspector_General_Of_Police_on_11_July_2024.PDF'
# try:
#     pdf_document = fitz.open(file_path)
#     raw_text = ''
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         raw_text += page.get_text()
#     pdf_document.close()
#     print("PDF file read successfully.")
# except Exception as e:
#     print("Error reading PDF file:", e)
#     exit()

# # Tokenize the raw text to obtain sentence tokens
# sent_tokens = nltk.sent_tokenize(raw_text)

# # Preprocess the text
# preprocessed_sent_tokens = [preprocess_text(sent) for sent in sent_tokens]

# # Define TF-IDF vectorizer with optimized parameters and TF-IDF weighting schemes
# word_vectorizer = TfidfVectorizer(
#     tokenizer=word_tokenize,     # Use NLTK's word tokenizer
#     stop_words='english',        # Use English stopwords
#     ngram_range=(1, 3),          # Use unigrams and bigrams
#     max_features=15000,           # Limit the vocabulary size to the top 5000 features
#     token_pattern=r'\b\w+\b',    # Use words as tokens
#     sublinear_tf=True,           # Apply sublinear term frequency scaling
#     smooth_idf=True,             # Smooth IDF weights by adding one to document frequencies
#     norm='l2'                    # Normalize TF-IDF vectors to unit length
# )

# # Apply TF-IDF vectorization on the preprocessed text
# word_vectors = word_vectorizer.fit_transform(preprocessed_sent_tokens)


# # Generate response function
# def generate_response(user_input, text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     # Check if the user is asking for specific information
#     if 'case number' in user_input or 'case no' in user_input:
#         return extract_case_number(text)
#     elif 'governing law' in user_input:
#         return extract_governing_law(text)
#     elif 'final verdict' in user_input:
#         return extract_final_verdict(text)
#     elif 'party' in user_input:
#         return extract_parties(text)
#     elif 'date' in user_input:
#         return extract_date(text)
#     elif 'title of the case' in user_input or 'case title' in user_input:
#         return extract_case_title(text)
#     elif 'summary of the case' in user_input or 'case summary' in user_input:
#         return generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
#     elif 'name of the court' in user_input or 'court name' in user_input:
#         return extract_court_name(text)
#     elif 'article' in user_input:
#         return extract_articles_sections(text)
#     else:
#         # Handle other types of questions
#         return "I'm sorry, I don't know."
    
# def extract_case_number(text):
#     # Regular expression pattern for matching case numbers
#     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?\d{1,}\s*(?:of|OF)\s*\d{4}\b'
    
#     # Find all matches of the pattern in the text
#     case_numbers = re.findall(pattern, text)
    
#     # Return a list of unique case numbers
#     return list(set(case_numbers))

# def extract_governing_law(text):
#     # Define keywords for criminal law and civil law
#     criminal_law_keywords = [
#         'criminal', 'crime', 'offense', 'prosecution', 'defendant', 'felony', 'misdemeanor'
#     ]
#     civil_law_keywords = [
#         'civil', 'plaintiff', 'defendant', 'contract', 'tort', 'property', 'family', 'probate'
#     ]
    
#     # Tokenize the text and tag the parts of speech
#     tokens = word_tokenize(text.lower())
#     tagged_tokens = pos_tag(tokens)
    
#     # Extract nouns and adjectives from the tagged tokens
#     nouns_adjectives = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]
    
#     # Check for criminal law keywords
#     for keyword in criminal_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Criminal Law"
    
#     # Check for civil law keywords
#     for keyword in civil_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Civil Law"
    
#     return "Governing law not identified"

# def extract_final_verdict(text):
#     # Define regular expressions to match common patterns for final verdicts and dates
#     verdict_patterns = [
#         r'final\s*verdict[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'final\s*verdict[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         #r'DATED[:\s]*\d{2}\.\d{2}\.\d{4}',  # Pattern for DATED: dd.mm.yyyy
#     ]
    
#     # Search for the patterns in the text
#     final_verdict = None
#     for pattern in verdict_patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             final_verdict = match.group(0).strip()
#             break
    
#     if not final_verdict:
#         # If no final verdict date is found, try to find a date in the title of the case
#         title_pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
#         title_match = re.search(title_pattern, text, re.IGNORECASE)
#         if title_match:
#             final_verdict = title_match.group(2).strip()
    
#     return final_verdict

# # def extract_parties(text):        
# #     # New patterns for petitioner and respondents
# #     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
# #     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

# #     petitioner_matches = new_petitioner_pattern.findall(text)
# #     respondents_matches = new_respondents_pattern.findall(text)

# #     if petitioner_matches and respondents_matches:
# #         petitioners = [match.strip() for match in petitioner_matches]
# #         respondents = []
# #         for match in respondents_matches:
# #             respondents_list = match.strip().split('\n')
# #             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
# #         petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
# #         respondents = "\n".join([f"{respondent}" for respondent in respondents])
# #         return f"Petitioners:\n{petitioners}\n\nRespondents:\n{respondents}"

       
# #     new_pattern_7 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_7.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
        
# #     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
# #     petitioners = petitioner_pattern.findall(text)
# #     respondents = respondent_pattern.findall(text)
    
# #     if petitioners and respondents:
# #         petitioners = [p.strip() for p in petitioners]
# #         respondents = [r.strip() for r in respondents]
# #         parties = []
# #         for petitioner, respondent in zip(petitioners, respondents):
# #             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
    
    
# #     new_pattern_6 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
# #         r'\s*(?:\n*AND\n*|\s+AND\s+)'
# #         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_6.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
    
# #     new_pattern_2_revised = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_2_revised.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     specific_pattern = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = specific_pattern.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents_text = match[1].strip()
# #             respondents_list = respondents_text.split('\n')
# #             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     new_pattern = re.compile(
# #     r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
# #     re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = new_pattern.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n\n".join(respondents_list)
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
   
# #     new_pattern_2 = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = new_pattern_2.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n\n".join(respondents_list)
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     new_pattern_3 = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_3.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents_text = match[1].strip()
# #             respondents_list = respondents_text.split('\n')
# #             respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
    
# #     new_pattern_4 = re.compile(
# #         r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_4.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
# #     additional_new_pattern = re.compile(
# #         r'\[.*?\]\s*(.*?)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = additional_new_pattern.findall(text)
    
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
            
# #             # Extract only the relevant petitioner information and remove leading spaces
# #             petitioner_parts = petitioner.split('\n')
# #             relevant_petitioner = '\n'.join(part.strip() for part in petitioner_parts if '.' not in part[:3])
            
# #             petitioners_list.append(relevant_petitioner)
# #             respondents_list.append("\n".join(respondents))
        
# #         petitioners_text = "\n\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    

  
# #     new_pattern_5 = re.compile(
# #         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = new_pattern_5.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents.append(match[1].strip())
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     # #Existing patterns
# #     # petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
# #     # respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
# #     # petitioners = petitioner_pattern.findall(text)
# #     # respondents = respondent_pattern.findall(text)
    
# #     # if petitioners and respondents:
# #     #     petitioners = [p.strip() for p in petitioners]
# #     #     respondents = [r.strip() for r in respondents]
# #     #     parties = []
# #     #     for petitioner, respondent in zip(petitioners, respondents):
# #     #         parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
# #     #     return "\n\n".join(parties)
    
# #     pattern_11_1 = re.compile(
# #         r'\[.*?\]\s*(.*?)\s*\.{3,}\s*Applicant\s*.*?Versus\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_11_1.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             # Remove leading/trailing spaces and align to left
# #             applicant_lines = [line.strip() for line in match[0].split('\n') if line.strip()]
# #             applicant = '\n'.join(applicant_lines)
            
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
# #             parties.append(f"Applicant:\n{applicant}\n\nRespondents:\n{respondents_text}")
# #         return "\n\n".join(parties)
    
    
# #     pattern_11 = re.compile(
# #         r'([^\n\r]+)\s*\.{3,}\s*Applicant\s*.*?VERSUS\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_11.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
# #             parties.append(f"Petitioner:{petitioner}\n\nRespondents:{respondents_text}")
# #         return "\n\n".join(parties)
    
    
    
# #     pattern_12 = re.compile(
# #         r'([^\n\r]+)\s*\.{1,}\s*Appellant\s*.*?Versus\s*([^\n\r]+)\s*\.{1,}\s*Respondent',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_12.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondent = match[1].strip()
# #             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_17 = re.compile(
# #         r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
# #         re.IGNORECASE
# #     )
    
# #     matches = pattern_17.findall(text)
# #     if matches:
# #         parties = []
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondent = match[1].strip()
# #             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
# #         return "\n\n".join(parties)
    
# #     pattern_19 = re.compile(
# #     r'([A-Z][A-Z\s&]+\.)\s*…\s*PETITIONERS\s*VERSUS\s*([A-Z][A-Z\s&]+\.)\s*…\s*RESPONDENTS',
# #     re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_19.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n\n".join(respondents_list)
# #         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_20 = re.compile(
# #     r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
# #     re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_20.findall(text)
# #     if matches:
# #         appellants_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             appellants = match[0].strip()
# #             respondents = match[1].strip()
            
# #             appellants_list.append(appellants)
# #             respondents_list.append(respondents)
        
# #         appellants_text = "\n".join(appellants_list)
# #         respondents_text = "\n".join(respondents_list)
# #         return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents}"
    
# #     pattern_21 = re.compile(
# #     r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
# #     re.IGNORECASE
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_21.findall(text)
# #     if matches:
# #         appellants_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             appellants = match[0].strip()
# #             respondents = match[1].strip()
            
# #             appellants_list.append(appellants)
# #             respondents_list.append(respondents)
        
# #         appellants_text = "\n".join(appellants_list)
# #         respondents_text = "\n".join(respondents_list)
# #         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
    
# #     pattern_appellant = re.compile(
# #         r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_appellant.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_23 = re.compile(
# #     r'(?:\d+\s+of\s+\d+\s+)?(?P<petitioner>[\w\s]+?)\s*…\s*Petitioner\s+versus\s+(?P<respondent>[\w\s]+?\s+and\s+others)\s*…\s*Respondents',
# #     re.IGNORECASE
# #     )
    
# #     matches = pattern_23.findall(text)
    
# #     if matches:
# #         seen = set()
# #         results = []
# #         for match in matches:
# #             petitioner, respondent = match
# #             # Remove all numbers and 'of' from petitioner name
# #             petitioner = re.sub(r'\d+\s+of\s+\d+\s+', '', petitioner).strip()
# #             # Remove any leading/trailing whitespace and newlines
# #             petitioner = ' '.join(petitioner.split())
# #             if petitioner not in seen:
# #                 seen.add(petitioner)
# #                 results.append(f"Petitioner: {petitioner}\nRespondent: {respondent.strip()}")
        
# #         return "\n\n".join(results)

    
        
# #     # pattern_9 = re.compile(r'([^\n\r]+)\s*\.{3,}\s*Petitio\s*.*?Vs\.\s*([\s\S]*?)\s*\.{3,}\s*Respondent',re.IGNORECASE | re.DOTALL)

# #     # matches = pattern_9.search(text)
# #     # if matches:
# #     #     petitioner = matches.group(1).strip()
# #     #     respondent = matches.group(2).strip()
# #     #     return f"Petitioners:\n{petitioner}\nRespondents:\n{respondent}"
    
# #     #if pattern 9 activate means pattern 10 not working and most of the pattern is not work
    
# #     pattern_10 = re.compile(
# #     r'([^\n\r]+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
# #     re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_10.findall(text)
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     # pattern_appellant = re.compile(
# #     #     r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
# #     #     re.IGNORECASE | re.DOTALL
# #     # )
    
# #     # matches = pattern_appellant.findall(text)
    
# #     # if matches:
# #     #     parties = []
# #     #     for petitioner, respondent in matches:
# #     #         parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #     #     return "\n\n".join(parties)

    
# #     pattern_13 = re.compile(
# #     r'([^\n\r]+)\s*\.{2,}\s*Petitione\s*.*?vs\s*([\s\S]*?)\s*\.{2,}\s*Respondents',
# #     re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_13.findall(text)
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_14 = re.compile(
# #         r'(?:.*?WP\(C\)\sNo\.\s\d+\sof\s\d+\s+Date\sof\sDecision:.*?\n)(.*?)\s*:::\s*Petitioner\s*-\s*Vs\s*-\s*([\s\S]+?)\s*:::\s*Respondents',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_14.findall(text)
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             # Split petitioner info into lines, strip each line, and rejoin
# #             petitioner_lines = match[0].strip().split('\n')
# #             petitioner = '\n'.join(line.strip() for line in petitioner_lines if line.strip())
            
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
# #     pattern_15_1 = re.compile(
# #         r'\b(?:Between:|Petitioner:|Petitioners:)\s*(.*?)\s*\.\.\.\s*Petitioner\s*(?:Versus|AND|Vs\.|v\.)\s*(.*?)\s*\.\.\.\s*(?:Respondents?|Respondent)',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_15_1.findall(text)
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         # Ensure no unnecessary leading or trailing whitespace in the final output
# #         petitioners_text = "\n".join([line.strip() for line in petitioners_text.split('\n')])
# #         respondents_text = "\n".join([line.strip() for line in respondents_text.split('\n')])
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"

    
# #     pattern_15 = re.compile(
# #     r'([^\n\r]+)\s*\.{3,}\s*PETITIONER\s*AND\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT\S*',
# #     re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_15.findall(text)
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_16 = re.compile(
# #     r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*([\s\S]+?)\s*\.\.\s*Responden',
# #     re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_16.findall(text)
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_18 = re.compile(
# #     r'((?:\d+\.[A-Z][.\w\s]+\s*)+)\.\.Pet.*?vs\s+((?:\d+\..*?\.\s*)+)\.\.\s*Respo',
# #     re.IGNORECASE | re.DOTALL
# #     )
    
# #     # Find all matches in the text
# #     matches = pattern_18.findall(text)
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     pattern_22 = re.compile(
# #         r'BETWEEN:-\s*(?P<petitioner>.*?)\s*\(BY.*?\)\s*AND\s*(?P<respondents>(?:\d+\..*?)+)(?:\(BY|$)',
# #         re.DOTALL | re.IGNORECASE
# #     )
    
# #     match = pattern_22.search(text)
# #     if match:
# #         petitioner = match.group('petitioner').strip()
# #         respondents_text = match.group('respondents').strip()
        
# #         # Format the petitioner
# #         petitioner_formatted = "\n".join(line.strip() for line in petitioner.split('\n'))
        
# #         # Format the respondents
# #         respondents = re.findall(r'\d+\.(.*?)(?=\d+\.|\Z)', respondents_text, re.DOTALL)
# #         respondents_formatted = "\n".join(f"{i+1}. {' '.join(line.strip() for line in resp.split())}" 
# #                                           for i, resp in enumerate(respondents))
        
# #         return f"Petitioner:\n{petitioner_formatted}\n\nRespondents:\n{respondents_formatted}"
    
# #     # pattern_23 = re.compile(
# #     # r'(?:\d+\s+of\s+\d+\s+)?(?P<petitioner>[\w\s]+?)\s*…\s*Petitioner\s+versus\s+(?P<respondent>[\w\s]+?\s+and\s+others)\s*…\s*Respondents',
# #     # re.IGNORECASE
# #     # )
    
# #     # matches = pattern_23.findall(text)
    
# #     # if matches:
# #     #     seen = set()
# #     #     results = []
# #     #     for match in matches:
# #     #         petitioner, respondent = match
# #     #         # Remove all numbers and 'of' from petitioner name
# #     #         petitioner = re.sub(r'\d+\s+of\s+\d+\s+', '', petitioner).strip()
# #     #         # Remove any leading/trailing whitespace and newlines
# #     #         petitioner = ' '.join(petitioner.split())
# #     #         if petitioner not in seen:
# #     #             seen.add(petitioner)
# #     #             results.append(f"Petitioner: {petitioner}\nRespondent: {respondent.strip()}")
        
# #     #     return "\n\n".join(results)
    

# # #2    
# #     pattern_ellipsis = re.compile(
# #         r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_ellipsis.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     pattern_dots = re.compile(
# #         r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_dots.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\n\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
    
    
# #     pattern_dashes = re.compile(
# #         r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_dashes.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
    
# #     pattern_provided = re.compile(
# #         r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_provided.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
    
# #     # pattern_appellant = re.compile(
# #     #     r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
# #     #     re.IGNORECASE | re.DOTALL
# #     # )
    
# #     # matches = pattern_appellant.findall(text)
    
# #     # if matches:
# #     #     parties = []
# #     #     for petitioner, respondent in matches:
# #     #         parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #     #     return "\n\n".join(parties)
    
# #     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
# #     matches = pattern_specific_parties.findall(text)
    
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
# #         return "\n\n".join(parties)
           
# #     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
# #     matches = pattern_specific_parties_1.findall(text) 
# #     if matches:
# #         parties = []
# #         for petitioner, respondent in matches:
# #             petitioner = petitioner.strip()
# #             respondent = respondent.strip()
# #             # Check if both petitioner and respondent are in uppercase
# #             if petitioner.isupper() and respondent.isupper():
# #                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
# #         # Join and return the result
# #         return "\n\n".join(parties)
         
# #     # new_pattern_8 = re.compile(
# #     #       r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
# #     # ) 
# #     # match = new_pattern_8.search(text)
# #     # if match:
# #     #     petitioner = match.group(1).strip()
# #     #     respondents_text = match.group(2).strip()
# #     #     respondents = respondents_text.split('AND')
# #     #     respondents = [resp.strip() for resp in respondents]
# #     #     return {
# #     #         'petitioner': petitioner,
# #     #         'respondents': respondents
# #     #     }
    
# #     # If no patterns match, return "Parties not found."
# #     return "Parties not found."


# def extract_parties(text):        
#     # New patterns for petitioner and respondents
#     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
#     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

#     petitioner_matches = new_petitioner_pattern.findall(text)
#     respondents_matches = new_respondents_pattern.findall(text)

#     if petitioner_matches and respondents_matches:
#         petitioners = [match.strip() for match in petitioner_matches]
#         respondents = []
#         for match in respondents_matches:
#             respondents_list = match.strip().split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
#         petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
#         respondents = "\n".join([f"{respondent}" for respondent in respondents])
#         return f"Petitioners:\n{petitioners}\n\nRespondents:\n{respondents}"

       
#     new_pattern_7 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_7.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
        
#     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
#     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
#     petitioners = petitioner_pattern.findall(text)
#     respondents = respondent_pattern.findall(text)
    
#     if petitioners and respondents:
#         petitioners = [p.strip() for p in petitioners]
#         respondents = [r.strip() for r in respondents]
#         parties = []
#         for petitioner, respondent in zip(petitioners, respondents):
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
    
    
#     new_pattern_6 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_6.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
    
#     new_pattern_2_revised = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_2_revised.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     specific_pattern = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = specific_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     new_pattern = re.compile(
#     r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = new_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
   
#     new_pattern_2 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = new_pattern_2.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     new_pattern_3 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_3.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
    
#     new_pattern_4 = re.compile(
#         r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_4.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
#     additional_new_pattern = re.compile(
#         r'\[.*?\]\s*(.*?)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = additional_new_pattern.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
            
#             # Extract only the relevant petitioner information and remove leading spaces
#             petitioner_parts = petitioner.split('\n')
#             relevant_petitioner = '\n'.join(part.strip() for part in petitioner_parts if '.' not in part[:3])
            
#             petitioners_list.append(relevant_petitioner)
#             respondents_list.append("\n".join(respondents))
        
#         petitioners_text = "\n\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    

  
#     new_pattern_5 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_5.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
        
#     pattern_11_1 = re.compile(
#         r'\[.*?\]\s*(.*?)\s*\.{3,}\s*Applicant\s*.*?Versus\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_11_1.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             # Remove leading/trailing spaces and align to left
#             applicant_lines = [line.strip() for line in match[0].split('\n') if line.strip()]
#             applicant = '\n'.join(applicant_lines)
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Applicant:\n{applicant}\n\nRespondents:\n{respondents_text}")
#         return "\n\n".join(parties)
    
    
#     pattern_11 = re.compile(
#         r'([^\n\r]+)\s*\.{3,}\s*Applicant\s*.*?VERSUS\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_11.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Petitioner:{petitioner}\n\nRespondents:{respondents_text}")
#         return "\n\n".join(parties)
    
    
    
#     pattern_12 = re.compile(
#         r'([^\n\r]+)\s*\.{1,}\s*Appellant\s*.*?Versus\s*([^\n\r]+)\s*\.{1,}\s*Respondent',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_12.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_17 = re.compile(
#         r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
#         re.IGNORECASE
#     )
    
#     matches = pattern_17.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_19 = re.compile(
#     r'([A-Z][A-Z\s&]+\.)\s*…\s*PETITIONERS\s*VERSUS\s*([A-Z][A-Z\s&]+\.)\s*…\s*RESPONDENTS',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_19.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_20 = re.compile(
#     r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_20.findall(text)
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip()
#             respondents = match[1].strip()
            
#             appellants_list.append(appellants)
#             respondents_list.append(respondents)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
#         return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents}"
    
#     pattern_21 = re.compile(
#     r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
#     re.IGNORECASE
#     )
    
#     # Find all matches in the text
#     matches = pattern_21.findall(text)
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip()
#             respondents = match[1].strip()
            
#             appellants_list.append(appellants)
#             respondents_list.append(respondents)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
#         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
    
#     pattern_appellant = re.compile(
#         r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_appellant.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_23 = re.compile(
#     r'(?:\d+\s+of\s+\d+\s+)?(?P<petitioner>[\w\s]+?)\s*…\s*Petitioner\s+versus\s+(?P<respondent>[\w\s]+?\s+and\s+others)\s*…\s*Respondents',
#     re.IGNORECASE
#     )
    
#     matches = pattern_23.findall(text)
    
#     if matches:
#         seen = set()
#         results = []
#         for match in matches:
#             petitioner, respondent = match
#             # Remove all numbers and 'of' from petitioner name
#             petitioner = re.sub(r'\d+\s+of\s+\d+\s+', '', petitioner).strip()
#             # Remove any leading/trailing whitespace and newlines
#             petitioner = ' '.join(petitioner.split())
#             if petitioner not in seen:
#                 seen.add(petitioner)
#                 results.append(f"Petitioner: {petitioner}\nRespondent: {respondent.strip()}")
        
#         return "\n\n".join(results)

#     pattern_10_1 = re.compile(
#         r'((?:\d+\.[^\n\r]+\s*)+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_10_1.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip().split('\n')
#             petitioner = [pet.strip() for pet in petitioner if pet.strip()]
#             petitioner_text = "\n".join(petitioner)
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner_text)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_10 = re.compile(
#     r'([^\n\r]+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_10.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
        
#     pattern_13 = re.compile(
#     r'([^\n\r]+)\s*\.{2,}\s*Petitione\s*.*?vs\s*([\s\S]*?)\s*\.{2,}\s*Respondents',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_13.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_14 = re.compile(
#         r'(?:.*?WP\(C\)\sNo\.\s\d+\sof\s\d+\s+Date\sof\sDecision:.*?\n)(.*?)\s*:::\s*Petitioner\s*-\s*Vs\s*-\s*([\s\S]+?)\s*:::\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_14.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             # Split petitioner info into lines, strip each line, and rejoin
#             petitioner_lines = match[0].strip().split('\n')
#             petitioner = '\n'.join(line.strip() for line in petitioner_lines if line.strip())
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_15_3 = re.compile(
#         r'((?:\d+\.\s?[^\n\r]+\s*)+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:versus)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15_3.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             petitioners = petitioner.split('\n')
#             petitioners = [pet.strip() for pet in petitioners if pet.strip()]
#             petitioners_text = "\n".join(petitioners)

#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioners_text)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
#     pattern_15_2 = re.compile(
#         r'Between:\s*(.*?)\s*\.{3,}\s*APPELLANT(?:\(S\))?\s*(?:AND)?\s*(.*?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15_2.findall(text)
    
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip().split('\n')
#             respondents = match[1].strip().split('\n')
            
#             appellants = [app.strip() for app in appellants if app.strip()]
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
            
#             appellants_text = "\n".join(appellants)
#             respondents_text = "\n".join(respondents)
            
#             appellants_list.append(appellants_text)
#             respondents_list.append(respondents_text)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
        
#         return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents_text}"
    
    
#     pattern_15_1 = re.compile(
#         r'\b(?:Between:|Petitioner:|Petitioners:)\s*(.*?)\s*\.\.\.\s*Petitioner\s*(?:Versus|AND|Vs\.|v\.)\s*(.*?)\s*\.\.\.\s*(?:Respondents?|Respondent)',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_15_1.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         # Ensure no unnecessary leading or trailing whitespace in the final output
#         petitioners_text = "\n".join([line.strip() for line in petitioners_text.split('\n')])
#         respondents_text = "\n".join([line.strip() for line in respondents_text.split('\n')])
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"

    
#     pattern_15 = re.compile(
#         r'([^\n\r]+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_16 = re.compile(
#     r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*([\s\S]+?)\s*\.\.\s*Responden',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_16.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_18 = re.compile(
#     r'((?:\d+\.[A-Z][.\w\s]+\s*)+)\.\.Pet.*?vs\s+((?:\d+\..*?\.\s*)+)\.\.\s*Respo',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_18.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_22 = re.compile(
#         r'BETWEEN:-\s*(?P<petitioner>.*?)\s*\(BY.*?\)\s*AND\s*(?P<respondents>(?:\d+\..*?)+)(?:\(BY|$)',
#         re.DOTALL | re.IGNORECASE
#     )
    
#     match = pattern_22.search(text)
#     if match:
#         petitioner = match.group('petitioner').strip()
#         respondents_text = match.group('respondents').strip()
        
#         # Format the petitioner
#         petitioner_formatted = "\n".join(line.strip() for line in petitioner.split('\n'))
        
#         # Format the respondents
#         respondents = re.findall(r'\d+\.(.*?)(?=\d+\.|\Z)', respondents_text, re.DOTALL)
#         respondents_formatted = "\n".join(f"{i+1}. {' '.join(line.strip() for line in resp.split())}" 
#                                           for i, resp in enumerate(respondents))
        
#         return f"Petitioner:\n{petitioner_formatted}\n\nRespondents:\n{respondents_formatted}"
    
#     specific_pattern_1 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Pe\s*versus\s*([\s\S]+?)\s*\.\.\.\s*Re', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = specific_pattern_1.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Parties:\nPetitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #2    
#     pattern_ellipsis = re.compile(
#         r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_ellipsis.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_dots = re.compile(
#         r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dots.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\n\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
    
#     pattern_dashes = re.compile(
#         r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dashes.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
#     pattern_provided = re.compile(
#         r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_provided.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
#     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
#     matches = pattern_specific_parties.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
           
#     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
#     matches = pattern_specific_parties_1.findall(text) 
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioner = petitioner.strip()
#             respondent = respondent.strip()
#             # Check if both petitioner and respondent are in uppercase
#             if petitioner.isupper() and respondent.isupper():
#                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
#         # Join and return the result
#         return "\n\n".join(parties)
         
#     return "Parties not found."


# # def extract_parties(text):
# #     specific_pattern_1 = re.compile(
# #         r'([^\n\r]+?)\s*\.\.\.\s*Pe\s*versus\s*([\s\S]+?)\s*\.\.\.\s*Re', 
# #         re.IGNORECASE | re.DOTALL
# #     )

# #     matches = specific_pattern_1.findall(text)
# #     if matches:
# #         petitioners = []
# #         respondents = []
# #         for match in matches:
# #             petitioners.append(match[0].strip())
# #             respondents_text = match[1].strip()
# #             respondents_list = respondents_text.split('\n')
# #             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
# #         petitioners_text = "\n".join(petitioners)
# #         respondents_text = "\n".join(respondents)
# #         return f"Parties:\nPetitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"

# #     return "Parties not found."



# # def extract_parties(text):
# #     pattern_15 = re.compile(
# #         r'((?:\d+\.[^\n\r]+\s*)+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_15.findall(text)
    
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# # def extract_parties(text):
# #     pattern_15_2 = re.compile(
# #         r'((?:\d+\.\s?[^\n\r]+\s*)+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:versus)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_15_2.findall(text)
    
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             petitioners = petitioner.split('\n')
# #             petitioners = [pet.strip() for pet in petitioners if pet.strip()]
# #             petitioners_text = "\n".join(petitioners)

# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioners_text)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #     return "Parties not found."



# # def extract_parties(text):
# #     pattern_15_2 = re.compile(
# #         r'Between:\s*(.*?)\s*\.{3,}\s*APPELLANT(?:\(S\))?\s*(?:AND)?\s*(.*?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_15_2.findall(text)
    
# #     if matches:
# #         appellants_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             appellants = match[0].strip().split('\n')
# #             respondents = match[1].strip().split('\n')
            
# #             appellants = [app.strip() for app in appellants if app.strip()]
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
            
# #             appellants_text = "\n".join(appellants)
# #             respondents_text = "\n".join(respondents)
            
# #             appellants_list.append(appellants_text)
# #             respondents_list.append(respondents_text)
        
# #         appellants_text = "\n".join(appellants_list)
# #         respondents_text = "\n".join(respondents_list)
        
# #         return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents_text}"
    
# #     return "No matches found"




# def extract_date(text):
#     # Define regex pattern to match dates in various formats
#     date_pattern = (
#         r'(\b(?:[12][0-9]|3[01]|0?[1-9])(st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s\d{4}\b)|'
#         r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s(?:[12][0-9]|3[01]),?\s\d{4}\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}\b)'
#     )
    
#     # Find all matches of the date pattern in the text
#     matches = re.findall(date_pattern, text)
    
#     # Flatten the list of tuples and remove empty strings
#     dates = [date for match in matches for date in match if date]
    
#     # Remove duplicates by converting to set and back to list
#     unique_dates = list(set(dates))
    
#     return unique_dates
    
# def extract_case_title(text):
#     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
#     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/0-9]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
  
#     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
#     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-0-9]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-0-9]+))'
    
#     # First try matching the original pattern
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     # If no match, try the alternative pattern
#     if not match:
#         match = re.search(pattern_alt, text, re.IGNORECASE)
    
#     if match:
#         title = match.group(1).strip()
#         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
#             date = match.group(2).strip()
#             return f"{title} on {date}"
#         else:
#             # Check for unwanted "Author" or other text in the title
#             if "Author" in title:
#                 title = title.split("Author")[0].strip()
#             return title
#     else:
#         return "Title and date not found"

# def extract_court_name(text):
#     # Define a comprehensive pattern for court names, including spaces between letters
#     comprehensive_pattern = (
#         r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T\s*'
#         r'O\s*F\s*I\s*N\s*D\s*I\s*A|'
#         r'Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
#         r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
#     )
    
#     # Search for the comprehensive pattern in the text
#     match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
#     if match:
#         return match.group(0).strip()
#     else:
#         # Define a fallback pattern for court names, including spaces between letters
#         fallback_pattern = r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T|Supreme|High|District) Court'
        
#         # Search for the fallback pattern in the text
#         match = re.search(fallback_pattern, text, re.IGNORECASE)
        
#         if match:
#             return match.group(0).strip()
#         else:
#             return "Court name not found"

# def extract_articles_sections(text):
#     # Comprehensive pattern for articles
#     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Comprehensive pattern for sections
#     section_pattern = re.compile(r'\b(?:Section|Sec\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for clauses
#     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for sub-sections
#     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

#     unique_references = set()  # Using a set to remove duplicates

#     # Function to process matches
#     def process_matches(pattern, prefix):
#         for match in pattern.finditer(text):
#             reference = match.group().strip()
#             if prefix not in reference.lower():
#                 reference = f"{prefix} {reference}"
#             unique_references.add(reference)

#     # Processing all patterns
#     process_matches(article_pattern, "Article :")
#     process_matches(section_pattern, "Section :")
#     process_matches(clause_pattern, "Clause :")
#     process_matches(subsection_pattern, "Sub-section :")

#     if unique_references:
#         return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
#     else:
#         return "No articles, sections, clauses, or sub-sections found."  
    
# def sanitize_text(text):
#     # Remove unwanted symbols using regular expressions
#     sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)
#     return sanitized_text

# # Updated `resolve_coreferences` function
# def resolve_coreferences(text):
#     doc = nlp(text)
#     resolved_text = []
#     for token in doc:
#         if token.dep_ == 'pronoun':
#             antecedent = token.head.text
#             resolved_text.append(antecedent)
#         else:
#             resolved_text.append(token.text)
    
#     return ' '.join(resolved_text)

# # Function to preprocess text with coreference resolution
# def preprocess_text_with_coref_resolution(text):
#     text = resolve_coreferences(text)
#     text = preprocess_text(text)
#     return text

# def print_all_information(text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     print("Case Number:", extract_case_number(text))
#     print("Governing Law:", extract_governing_law(text))
#     print("Final Verdict:", extract_final_verdict(text))
#     print("Parties:", extract_parties(text))
#     print("Date:", extract_date(text))
#     print("Title of the Case:", extract_case_title(text))
#     print("Name of the Court:", extract_court_name(text))
#     print("Articles:", extract_articles_sections(text))
#     print("Summary of the Case:", generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors))

# # Main interaction loop
# try:
#     print("Bot: I will provide information about the legal document.")
#     print_all_information(raw_text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model)
# except KeyboardInterrupt:
#     print("\nBot: Thanks for talking, Bye!")






































# import re
# import nltk
# import contractions
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from autocorrect import Speller
# import emoji
# import regex 
# import gensim.downloader as api
# import fitz  # PyMuPDF
# from nltk import pos_tag
# import warnings
# import numpy as np

# warnings.simplefilter(action='ignore', category=UserWarning)

# #nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Load the English language model
# nlp = spacy.load('en_core_web_sm')

# # Set max_length to a value that accommodates your text length
# nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

# # Load pre-trained Word2Vec model
# w2v_model = api.load('word2vec-google-news-300')

# # Initialize spell checker
# spell = Speller()

# # Define stopword2Vec
# stop_words = set(stopwords.words('english'))


# # Preprocessing function with lemmatization, spell checking, and NER tagging
# def preprocess_text(text):
    
#     # Correct spelling errors
#     text = spell(text)
    
#     # Remove HTML tags, URLs, and special characters
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Expand contractions
#     text = contractions.fix(text)
    
#     # Remove citations
#     text = re.sub(r'\[[0-9]+\]', '', text)
    
#     # Remove extra whitespaces
#     text = re.sub(r'\s+', ' ', text)
    
#     # Tokenization and NER tagging using spaCy
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if token.ent_type_:
#             tokens.append(token.ent_type_)
#         else:
#             tokens.append(token.text)
    
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Remove repeated characters
#     tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]
    
#     # Remove single characters and numeric tokens
#     tokens = [word for word in tokens if len(word) > 1 and not word.isdigit()]
    
#     # Handle emojis
#     text = emoji.demojize(text)
#     text = text.replace(":", "")
    
#     # Handle emoticons
#     emoticons = regex.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = regex.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
#     text = text + ' '.join(emoticons)
    
#     # Join tokens back into a string
#     preprocessed_text = ' '.join(tokens)
    
#     return preprocessed_text


# def calculate_similarity(sentence1, sentence2):
#     # Tokenize the sentences
#     tokens1 = word_tokenize(sentence1)
#     tokens2 = word_tokenize(sentence2)
    
#     # Filter out stopwords
#     tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
#     tokens2 = [word for word in tokens2 if word.lower() not in stop_words]
    
#     # Get the Word2Vec vectors for each word
#     vectors1 = [w2v_model[word] for word in tokens1 if word in w2v_model]
#     vectors2 = [w2v_model[word] for word in tokens2 if word in w2v_model]

#     # Calculate the average vectors for each sentence
#     if vectors1 and vectors2:
#         avg_vector1 = np.mean(vectors1, axis=0)
#         avg_vector2 = np.mean(vectors2, axis=0)
        
#         # Calculate the cosine similarity between the average vectors
#         similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
#         return similarity
#     else:
#         return 0.0  # Return 0 if no vectors are found or all words are OOV

# # Keep track of previous questions and responses
# previous_questions = []
# previous_responses = []




# def generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, top_k=15, similarity_threshold=0.5):
#     global previous_questions, previous_responses

#     bot_response = ''
    
#     # Preprocess user input
#     processed_input = preprocess_text(user_input)
    
#     # Check if the processed input is empty or contains only stopwords
#     if not processed_input or all(word in stop_words for word in processed_input.split()):
#         return "I am sorry, I don't understand."
    
#     # Check if the current question is the same as a previous one
#     if processed_input in previous_questions:
#         index = previous_questions.index(processed_input)
#         return previous_responses[index]
    
#     # If not, continue with Word2Vec processing
#     similarities = []
#     for sent in sent_tokens:
#         similarity = calculate_similarity(processed_input, sent)
#         similarities.append(similarity)
    
#     # Convert similarities to a NumPy array for easier processing
#     similarities = np.array(similarities)
    
#     # Sort the similarities in descending order
#     sorted_indices = np.argsort(similarities)[::-1]
    
#     # Find the top k most similar sentences that are not in previous responses
#     top_k_sentences = []
#     for index in sorted_indices:
#         if len(top_k_sentences) < top_k and similarities[index] >= similarity_threshold:
#             top_k_sentences.append(sent_tokens[index])
    
#     # Assign the top k sentences to bot_response
#     if top_k_sentences:
#         bot_response = '\n'.join(top_k_sentences)
#         # Filter out URLs and unwanted tags from the response
#         bot_response = re.sub(r'http\S+', '', bot_response)  # Remove URLs
        
#         # Track previous questions and responses
#         previous_questions.append(processed_input)
#         previous_responses.append(bot_response)
#     else:
#         bot_response = "I am sorry, I don't get enough details."
    
#     return bot_response  # Return the top k sentences separated by newline characters

# # Read text from the local PDF file using PyMuPDF
# file_path = 'C:/legal keywords summarization/case documents 1/Sri_Sankari_Prasad_Singh_Deo_vs_Union_Of_India_And_State_Of_Bihar_And_on_5_October_1951.PDF'
# try:
#     pdf_document = fitz.open(file_path)
#     raw_text = ''
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         raw_text += page.get_text()
#     pdf_document.close()
#     print("PDF file read successfully.")
# except Exception as e:
#     print("Error reading PDF file:", e)
#     exit()

# # Tokenize the raw text to obtain sentence tokens
# sent_tokens = nltk.sent_tokenize(raw_text)

# # Preprocess the text
# preprocessed_sent_tokens = [preprocess_text(sent) for sent in sent_tokens]

# # Define TF-IDF vectorizer with optimized parameters and TF-IDF weighting schemes
# word_vectorizer = TfidfVectorizer(
#     tokenizer=word_tokenize,     # Use NLTK's word tokenizer
#     stop_words='english',        # Use English stopwords
#     ngram_range=(1, 3),          # Use unigrams and bigrams
#     max_features=15000,           # Limit the vocabulary size to the top 5000 features
#     token_pattern=r'\b\w+\b',    # Use words as tokens
#     sublinear_tf=True,           # Apply sublinear term frequency scaling
#     smooth_idf=True,             # Smooth IDF weights by adding one to document frequencies
#     norm='l2'                    # Normalize TF-IDF vectors to unit length
# )

# # Apply TF-IDF vectorization on the preprocessed text
# word_vectors = word_vectorizer.fit_transform(preprocessed_sent_tokens)


# # Generate response function
# def generate_response(user_input, text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     # Check if the user is asking for specific information
#     if 'case number' in user_input or 'case no' in user_input:
#         return extract_case_number(text)
#     elif 'governing law' in user_input:
#         return extract_governing_law(text)
#     elif 'final verdict' in user_input:
#         return extract_final_verdict(text)
#     elif 'party' in user_input:
#         return extract_parties(text)
#     elif 'date' in user_input:
#         return extract_date(text)
#     elif 'title of the case' in user_input or 'case title' in user_input:
#         return extract_case_title(text)
#     elif 'summary of the case' in user_input or 'case summary' in user_input:
#         return generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
#     elif 'name of the court' in user_input or 'court name' in user_input:
#         return extract_court_name(text)
#     elif 'article' in user_input:
#         return extract_articles_sections(text)
#     else:
#         # Handle other types of questions
#         return "I'm sorry, I don't know."


# def extract_case_number(text):
#     # Regular expression pattern for matching case numbers, including ranges
#     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?(?<!\d\/)\b\d{2,}(?:-\d{2,})?(?:\s*(?:and|,)\s*\d{2,})*(?:\s*(?:of|OF)\s*|\s*\/\s*)\d{4}\b(?!\/\d{2})\b'
    
#     # Find all matches of the pattern in the text
#     case_numbers = re.findall(pattern, text)
    
#     # Return a list of unique case numbers
#     return list(set(case_numbers))

# # def extract_case_number(text):
# #     # Regular expression pattern for matching case numbers, excluding dates
# #     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?(?<!\d\/)\b\d{2,}(?:\s*(?:and|,)\s*\d{2,})*(?:\s*(?:of|OF)\s*|\s*\/\s*)\d{4}\b(?!\/\d{2})\b'
    
# #     # Find all matches of the pattern in the text
# #     case_numbers = re.findall(pattern, text)
    
# #     # Return a list of unique case numbers
# #     return list(set(case_numbers))



# # def extract_case_number(text):
# #     # Regular expression pattern for matching case numbers
# #     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?\d{1,}(?:\s*(?:and|,)\s*\d{1,})*(?:\s*(?:of|OF)\s*|\s*\/\s*)\d{4}\b'
    
# #     # Find all matches of the pattern in the text
# #     case_numbers = re.findall(pattern, text)
    
# #     # Return a list of unique case numbers
# #     return list(set(case_numbers))

# # def extract_case_number(text):
# #     # Regular expression pattern for matching case numbers
# #     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?\d{1,}(?:\s*(?:of|OF)\s*|\s*\/\s*)\d{4}\b'
    
# #     # Find all matches of the pattern in the text
# #     case_numbers = re.findall(pattern, text)
    
# #     # Return a list of unique case numbers
# #     return list(set(case_numbers))
    
# # def extract_case_number(text):
# #     # Regular expression pattern for matching case numbers
# #     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?\d{1,}\s*(?:of|OF)\s*\d{4}\b'
    
# #     # Find all matches of the pattern in the text
# #     case_numbers = re.findall(pattern, text)
    
# #     # Return a list of unique case numbers
# #     return list(set(case_numbers))

# def extract_governing_law(text):
#     # Define keywords for criminal law and civil law
#     criminal_law_keywords = [
#         'criminal', 'crime', 'offense', 'prosecution', 'defendant', 'felony', 'misdemeanor'
#     ]
#     civil_law_keywords = [
#         'civil', 'plaintiff', 'defendant', 'contract', 'tort', 'property', 'family', 'probate'
#     ]
    
#     # Tokenize the text and tag the parts of speech
#     tokens = word_tokenize(text.lower())
#     tagged_tokens = pos_tag(tokens)
    
#     # Extract nouns and adjectives from the tagged tokens
#     nouns_adjectives = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]
    
#     # Check for criminal law keywords
#     for keyword in criminal_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Criminal Law"
    
#     # Check for civil law keywords
#     for keyword in civil_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Civil Law"
    
#     return "Governing law not identified"

# def extract_final_verdict(text):
#     # Define regular expressions to match common patterns for final verdicts and dates
#     verdict_patterns = [
#         r'final\s*verdict[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'final\s*verdict[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         #r'DATED[:\s]*\d{2}\.\d{2}\.\d{4}',  # Pattern for DATED: dd.mm.yyyy
#     ]
    
#     # Search for the patterns in the text
#     final_verdict = None
#     for pattern in verdict_patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             final_verdict = match.group(0).strip()
#             break
    
#     if not final_verdict:
#         # If no final verdict date is found, try to find a date in the title of the case
#         title_pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
#         title_match = re.search(title_pattern, text, re.IGNORECASE)
#         if title_match:
#             final_verdict = title_match.group(2).strip()
    
#     return final_verdict



# def extract_parties(text):        
#     # New patterns for petitioner and respondents
#     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
#     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

#     petitioner_matches = new_petitioner_pattern.findall(text)
#     respondents_matches = new_respondents_pattern.findall(text)

#     if petitioner_matches and respondents_matches:
#         petitioners = [match.strip() for match in petitioner_matches]
#         respondents = []
#         for match in respondents_matches:
#             respondents_list = match.strip().split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
#         petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
#         respondents = "\n".join([f"{respondent}" for respondent in respondents])
#         return f"Petitioners:\n{petitioners}\n\nRespondents:\n{respondents}"

       
#     new_pattern_7 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_7.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
        
#     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
#     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
#     petitioners = petitioner_pattern.findall(text)
#     respondents = respondent_pattern.findall(text)
    
#     if petitioners and respondents:
#         petitioners = [p.strip() for p in petitioners]
#         respondents = [r.strip() for r in respondents]
#         parties = []
#         for petitioner, respondent in zip(petitioners, respondents):
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
    
    
#     new_pattern_6 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_6.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
    
#     new_pattern_2_revised = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_2_revised.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     specific_pattern = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = specific_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     new_pattern = re.compile(
#     r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = new_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
   
#     new_pattern_2 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = new_pattern_2.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     new_pattern_3 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_3.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
    
#     new_pattern_4 = re.compile(
#         r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_4.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
#     additional_new_pattern = re.compile(
#         r'\[.*?\]\s*(.*?)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = additional_new_pattern.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
            
#             # Extract only the relevant petitioner information and remove leading spaces
#             petitioner_parts = petitioner.split('\n')
#             relevant_petitioner = '\n'.join(part.strip() for part in petitioner_parts if '.' not in part[:3])
            
#             petitioners_list.append(relevant_petitioner)
#             respondents_list.append("\n".join(respondents))
        
#         petitioners_text = "\n\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    

  
#     new_pattern_5 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_5.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
        
#     pattern_11_1 = re.compile(
#         r'\[.*?\]\s*(.*?)\s*\.{3,}\s*Applicant\s*.*?Versus\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_11_1.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             # Remove leading/trailing spaces and align to left
#             applicant_lines = [line.strip() for line in match[0].split('\n') if line.strip()]
#             applicant = '\n'.join(applicant_lines)
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Applicant:\n{applicant}\n\nRespondents:\n{respondents_text}")
#         return "\n\n".join(parties)
    
    
#     pattern_11 = re.compile(
#         r'([^\n\r]+)\s*\.{3,}\s*Applicant\s*.*?VERSUS\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_11.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Petitioner:{petitioner}\n\nRespondents:{respondents_text}")
#         return "\n\n".join(parties)
    
    
    
#     pattern_12 = re.compile(
#         r'([^\n\r]+)\s*\.{1,}\s*Appellant\s*.*?Versus\s*([^\n\r]+)\s*\.{1,}\s*Respondent',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_12.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_17 = re.compile(
#         r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
#         re.IGNORECASE
#     )
    
#     matches = pattern_17.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_19 = re.compile(
#     r'([A-Z][A-Z\s&]+\.)\s*…\s*PETITIONERS\s*VERSUS\s*([A-Z][A-Z\s&]+\.)\s*…\s*RESPONDENTS',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_19.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_20 = re.compile(
#     r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_20.findall(text)
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip()
#             respondents = match[1].strip()
            
#             appellants_list.append(appellants)
#             respondents_list.append(respondents)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
#         return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents}"
    
#     pattern_21 = re.compile(
#     r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
#     re.IGNORECASE
#     )
    
#     # Find all matches in the text
#     matches = pattern_21.findall(text)
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip()
#             respondents = match[1].strip()
            
#             appellants_list.append(appellants)
#             respondents_list.append(respondents)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
#         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
    
#     pattern_appellant = re.compile(
#         r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_appellant.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_23 = re.compile(
#     r'(?:\d+\s+of\s+\d+\s+)?(?P<petitioner>[\w\s]+?)\s*…\s*Petitioner\s+versus\s+(?P<respondent>[\w\s]+?\s+and\s+others)\s*…\s*Respondents',
#     re.IGNORECASE
#     )
    
#     matches = pattern_23.findall(text)
    
#     if matches:
#         seen = set()
#         results = []
#         for match in matches:
#             petitioner, respondent = match
#             # Remove all numbers and 'of' from petitioner name
#             petitioner = re.sub(r'\d+\s+of\s+\d+\s+', '', petitioner).strip()
#             # Remove any leading/trailing whitespace and newlines
#             petitioner = ' '.join(petitioner.split())
#             if petitioner not in seen:
#                 seen.add(petitioner)
#                 results.append(f"Petitioner: {petitioner}\nRespondent: {respondent.strip()}")
        
#         return "\n\n".join(results)

#     pattern_10_1 = re.compile(
#         r'((?:\d+\.[^\n\r]+\s*)+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_10_1.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip().split('\n')
#             petitioner = [pet.strip() for pet in petitioner if pet.strip()]
#             petitioner_text = "\n".join(petitioner)
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner_text)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_10 = re.compile(
#     r'([^\n\r]+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_10.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
        
#     pattern_13 = re.compile(
#     r'([^\n\r]+)\s*\.{2,}\s*Petitione\s*.*?vs\s*([\s\S]*?)\s*\.{2,}\s*Respondents',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_13.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_14 = re.compile(
#         r'(?:.*?WP\(C\)\sNo\.\s\d+\sof\s\d+\s+Date\sof\sDecision:.*?\n)(.*?)\s*:::\s*Petitioner\s*-\s*Vs\s*-\s*([\s\S]+?)\s*:::\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_14.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             # Split petitioner info into lines, strip each line, and rejoin
#             petitioner_lines = match[0].strip().split('\n')
#             petitioner = '\n'.join(line.strip() for line in petitioner_lines if line.strip())
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_15_3 = re.compile(
#         r'((?:\d+\.\s?[^\n\r]+\s*)+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:versus)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15_3.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             petitioners = petitioner.split('\n')
#             petitioners = [pet.strip() for pet in petitioners if pet.strip()]
#             petitioners_text = "\n".join(petitioners)

#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioners_text)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
#     pattern_15_2 = re.compile(
#         r'Between:\s*(.*?)\s*\.{3,}\s*APPELLANT(?:\(S\))?\s*(?:AND)?\s*(.*?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15_2.findall(text)
    
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip().split('\n')
#             respondents = match[1].strip().split('\n')
            
#             appellants = [app.strip() for app in appellants if app.strip()]
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
            
#             appellants_text = "\n".join(appellants)
#             respondents_text = "\n".join(respondents)
            
#             appellants_list.append(appellants_text)
#             respondents_list.append(respondents_text)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
        
#         return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents_text}"
    
    
#     pattern_15_1 = re.compile(
#         r'\b(?:Between:|Petitioner:|Petitioners:)\s*(.*?)\s*\.\.\.\s*Petitioner\s*(?:Versus|AND|Vs\.|v\.)\s*(.*?)\s*\.\.\.\s*(?:Respondents?|Respondent)',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_15_1.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         # Ensure no unnecessary leading or trailing whitespace in the final output
#         petitioners_text = "\n".join([line.strip() for line in petitioners_text.split('\n')])
#         respondents_text = "\n".join([line.strip() for line in respondents_text.split('\n')])
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_15_4 = re.compile(
#         r'Between:\s*(.*?)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15_4.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"

    
#     pattern_15 = re.compile(
#         r'([^\n\r]+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_16 = re.compile(
#     r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*([\s\S]+?)\s*\.\.\s*Responden',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_16.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_18 = re.compile(
#     r'((?:\d+\.[A-Z][.\w\s]+\s*)+)\.\.Pet.*?vs\s+((?:\d+\..*?\.\s*)+)\.\.\s*Respo',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_18.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_22 = re.compile(
#         r'BETWEEN:-\s*(?P<petitioner>.*?)\s*\(BY.*?\)\s*AND\s*(?P<respondents>(?:\d+\..*?)+)(?:\(BY|$)',
#         re.DOTALL | re.IGNORECASE
#     )
    
#     match = pattern_22.search(text)
#     if match:
#         petitioner = match.group('petitioner').strip()
#         respondents_text = match.group('respondents').strip()
        
#         # Format the petitioner
#         petitioner_formatted = "\n".join(line.strip() for line in petitioner.split('\n'))
        
#         # Format the respondents
#         respondents = re.findall(r'\d+\.(.*?)(?=\d+\.|\Z)', respondents_text, re.DOTALL)
#         respondents_formatted = "\n".join(f"{i+1}. {' '.join(line.strip() for line in resp.split())}" 
#                                           for i, resp in enumerate(respondents))
        
#         return f"Petitioner:\n{petitioner_formatted}\n\nRespondents:\n{respondents_formatted}"
    
#     specific_pattern_1 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Pe\s*versus\s*([\s\S]+?)\s*\.\.\.\s*Re', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = specific_pattern_1.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Parties:\nPetitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #2    
#     pattern_ellipsis = re.compile(
#         r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_ellipsis.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_dots = re.compile(
#         r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dots.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\n\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
    
#     pattern_dashes = re.compile(
#         r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dashes.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
#     pattern_provided = re.compile(
#         r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_provided.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
#     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
#     matches = pattern_specific_parties.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
           
#     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
#     matches = pattern_specific_parties_1.findall(text) 
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioner = petitioner.strip()
#             respondent = respondent.strip()
#             # Check if both petitioner and respondent are in uppercase
#             if petitioner.isupper() and respondent.isupper():
#                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
#         # Join and return the result
#         return "\n\n".join(parties)
         
#     return "Parties not found."


# # def extract_parties(text):    
# #     pattern_15_4 = re.compile(
# #         r'Between:\s*(.*?)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_15_4.findall(text)
    
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"


    


# def extract_date(text):
#     # Define regex pattern to match dates in various formats
#     date_pattern = (
#         r'(\b(?:[12][0-9]|3[01]|0?[1-9])(st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s\d{4}\b)|'
#         r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s(?:[12][0-9]|3[01]),?\s\d{4}\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}\b)'
#     )
    
#     # Find all matches of the date pattern in the text
#     matches = re.findall(date_pattern, text)
    
#     # Flatten the list of tuples and remove empty strings
#     dates = [date for match in matches for date in match if date]
    
#     # Remove duplicates by converting to set and back to list
#     unique_dates = list(set(dates))
    
#     return unique_dates
    
# def extract_case_title(text):
#     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
#     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/0-9]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
  
#     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
#     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-0-9]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-0-9]+))'
    
#     # First try matching the original pattern
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     # If no match, try the alternative pattern
#     if not match:
#         match = re.search(pattern_alt, text, re.IGNORECASE)
    
#     if match:
#         title = match.group(1).strip()
#         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
#             date = match.group(2).strip()
#             return f"{title} on {date}"
#         else:
#             # Check for unwanted "Author" or other text in the title
#             if "Author" in title:
#                 title = title.split("Author")[0].strip()
#             return title
#     else:
#         return "Title and date not found"

# def extract_court_name(text):
#     # Define a comprehensive pattern for court names, including spaces between letters
#     comprehensive_pattern = (
#         r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T\s*'
#         r'O\s*F\s*I\s*N\s*D\s*I\s*A|'
#         r'Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
#         r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
#     )
    
#     # Search for the comprehensive pattern in the text
#     match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
#     if match:
#         return match.group(0).strip()
#     else:
#         # Define a fallback pattern for court names, including spaces between letters
#         fallback_pattern = r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T|Supreme|High|District) Court'
        
#         # Search for the fallback pattern in the text
#         match = re.search(fallback_pattern, text, re.IGNORECASE)
        
#         if match:
#             return match.group(0).strip()
#         else:
#             return "Court name not found"

# def extract_articles_sections(text):
#     # Comprehensive pattern for articles
#     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Comprehensive pattern for sections (including plural 'Sections')
#     section_pattern = re.compile(r'\b(?:Section|Sec\.?|Sections|Secs\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for clauses
#     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for sub-sections
#     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

#     unique_references = set()  # Using a set to remove duplicates

#     # Function to process matches
#     def process_matches(pattern, prefix):
#         for match in pattern.finditer(text):
#             reference = match.group().strip()
#             if prefix not in reference.lower():
#                 reference = f"{prefix} {reference}"
#             unique_references.add(reference)

#     # Processing all patterns
#     process_matches(article_pattern, "Article :")
#     process_matches(section_pattern, "Section :")
#     process_matches(clause_pattern, "Clause :")
#     process_matches(subsection_pattern, "Sub-section :")

#     if unique_references:
#         return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
#     else:
#         return "No articles, sections, clauses, or sub-sections found."


# # def extract_articles_sections(text):
# #     # Comprehensive pattern for articles
# #     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
# #     # Comprehensive pattern for sections
# #     section_pattern = re.compile(r'\b(?:Section|Sec\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
# #     # Pattern for clauses
# #     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
# #     # Pattern for sub-sections
# #     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

# #     unique_references = set()  # Using a set to remove duplicates

# #     # Function to process matches
# #     def process_matches(pattern, prefix):
# #         for match in pattern.finditer(text):
# #             reference = match.group().strip()
# #             if prefix not in reference.lower():
# #                 reference = f"{prefix} {reference}"
# #             unique_references.add(reference)

# #     # Processing all patterns
# #     process_matches(article_pattern, "Article :")
# #     process_matches(section_pattern, "Section :")
# #     process_matches(clause_pattern, "Clause :")
# #     process_matches(subsection_pattern, "Sub-section :")

# #     if unique_references:
# #         return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
# #     else:
# #         return "No articles, sections, clauses, or sub-sections found."  
    
# def sanitize_text(text):
#     # Remove unwanted symbols using regular expressions
#     sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)
#     return sanitized_text

# # Updated `resolve_coreferences` function
# def resolve_coreferences(text):
#     doc = nlp(text)
#     resolved_text = []
#     for token in doc:
#         if token.dep_ == 'pronoun':
#             antecedent = token.head.text
#             resolved_text.append(antecedent)
#         else:
#             resolved_text.append(token.text)
    
#     return ' '.join(resolved_text)

# # Function to preprocess text with coreference resolution
# def preprocess_text_with_coref_resolution(text):
#     text = resolve_coreferences(text)
#     text = preprocess_text(text)
#     return text

# def print_all_information(text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     print("Case Number:", extract_case_number(text))
#     print("Governing Law:", extract_governing_law(text))
#     print("Final Verdict:", extract_final_verdict(text))
#     print("Parties:", extract_parties(text))
#     print("Date:", extract_date(text))
#     print("Title of the Case:", extract_case_title(text))
#     print("Name of the Court:", extract_court_name(text))
#     print("Articles:", extract_articles_sections(text))
#     print("Summary of the Case:", generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors))

# # Main interaction loop
# try:
#     print("Bot: I will provide information about the legal document.")
#     print_all_information(raw_text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model)
# except KeyboardInterrupt:
#     print("\nBot: Thanks for talking, Bye!")






























# import re
# import nltk
# import contractions
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from autocorrect import Speller
# import emoji
# import regex 
# import gensim.downloader as api
# import fitz  # PyMuPDF
# from nltk import pos_tag
# import warnings
# import numpy as np

# warnings.simplefilter(action='ignore', category=UserWarning)

# #nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Load the English language model
# nlp = spacy.load('en_core_web_sm')

# # Set max_length to a value that accommodates your text length
# nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

# # Load pre-trained Word2Vec model
# w2v_model = api.load('word2vec-google-news-300')

# # Initialize spell checker
# spell = Speller()

# # Define stopword2Vec
# stop_words = set(stopwords.words('english'))


# # Preprocessing function with lemmatization, spell checking, and NER tagging
# def preprocess_text(text):
    
#     # Correct spelling errors
#     text = spell(text)
    
#     # Remove HTML tags, URLs, and special characters
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Expand contractions
#     text = contractions.fix(text)
    
#     # Remove citations
#     text = re.sub(r'\[[0-9]+\]', '', text)
    
#     # Remove extra whitespaces
#     text = re.sub(r'\s+', ' ', text)
    
#     # Tokenization and NER tagging using spaCy
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if token.ent_type_:
#             tokens.append(token.ent_type_)
#         else:
#             tokens.append(token.text)
    
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Remove repeated characters
#     tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]
    
#     # Remove single characters and numeric tokens
#     tokens = [word for word in tokens if len(word) > 1 and not word.isdigit()]
    
#     # Handle emojis
#     text = emoji.demojize(text)
#     text = text.replace(":", "")
    
#     # Handle emoticons
#     emoticons = regex.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = regex.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
#     text = text + ' '.join(emoticons)
    
#     # Join tokens back into a string
#     preprocessed_text = ' '.join(tokens)
    
#     return preprocessed_text

# def calculate_similarity(sentence1, sentence2):
#     # Tokenize the sentences
#     tokens1 = word_tokenize(sentence1)
#     tokens2 = word_tokenize(sentence2)
    
#     # Filter out stopwords
#     tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
#     tokens2 = [word for word in tokens2 if word.lower() not in stop_words]
    
#     # Get the Word2Vec vectors for each word
#     vectors1 = [w2v_model[word] for word in tokens1 if word in w2v_model]
#     vectors2 = [w2v_model[word] for word in tokens2 if word in w2v_model]

#     # Calculate the average vectors for each sentence
#     if vectors1 and vectors2:
#         avg_vector1 = np.mean(vectors1, axis=0)
#         avg_vector2 = np.mean(vectors2, axis=0)
        
#         # Calculate the cosine similarity between the average vectors
#         similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
#         return similarity
#     else:
#         return 0.0  # Return 0 if no vectors are found or all words are OOV

# # Keep track of previous questions and responses
# previous_questions = []
# previous_responses = []

# def generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, top_k=15, similarity_threshold=0.5):
#     global previous_questions, previous_responses

#     bot_response = ''
    
#     # Preprocess user input
#     processed_input = preprocess_text(user_input)
    
#     # Check if the processed input is empty or contains only stopwords
#     if not processed_input or all(word in stop_words for word in processed_input.split()):
#         return "I am sorry, I don't understand."
    
#     # Check if the current question is the same as a previous one
#     if processed_input in previous_questions:
#         index = previous_questions.index(processed_input)
#         return previous_responses[index]
    
#     # If not, continue with Word2Vec processing
#     similarities = []
#     for sent in sent_tokens:
#         similarity = calculate_similarity(processed_input, sent)
#         similarities.append(similarity)
    
#     # Convert similarities to a NumPy array for easier processing
#     similarities = np.array(similarities)
    
#     # Sort the similarities in descending order
#     sorted_indices = np.argsort(similarities)[::-1]
    
#     # Find the top k most similar sentences that are not in previous responses
#     top_k_sentences = []
#     for index in sorted_indices:
#         if len(top_k_sentences) < top_k and similarities[index] >= similarity_threshold:
#             top_k_sentences.append(sent_tokens[index])
    
#     # Assign the top k sentences to bot_response
#     if top_k_sentences:
#         bot_response = '\n'.join(top_k_sentences)
#         # Filter out URLs and unwanted tags from the response
#         bot_response = re.sub(r'http\S+', '', bot_response)  # Remove URLs
        
#         # Track previous questions and responses
#         previous_questions.append(processed_input)
#         previous_responses.append(bot_response)
#     else:
#         bot_response = "I am sorry, I don't get enough details."
    
#     return bot_response  # Return the top k sentences separated by newline characters

# # Read text from the local PDF file using PyMuPDF
# file_path = 'C:/legal keywords summarization/case documents 1/In_Re__The_Berubari_Union_Andexchange_Of_vs_Reference_Under_Article_143_1_Ofthe_on_1_April_1959.PDF'
# try:
#     pdf_document = fitz.open(file_path)
#     raw_text = ''
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         raw_text += page.get_text()
#     pdf_document.close()
#     print("PDF file read successfully.")
# except Exception as e:
#     print("Error reading PDF file:", e)
#     exit()

# # Tokenize the raw text to obtain sentence tokens
# sent_tokens = nltk.sent_tokenize(raw_text)

# # Preprocess the text
# preprocessed_sent_tokens = [preprocess_text(sent) for sent in sent_tokens]

# # Define TF-IDF vectorizer with optimized parameters and TF-IDF weighting schemes
# word_vectorizer = TfidfVectorizer(
#     tokenizer=word_tokenize,     # Use NLTK's word tokenizer
#     stop_words='english',        # Use English stopwords
#     ngram_range=(1, 3),          # Use unigrams and bigrams
#     max_features=15000,           # Limit the vocabulary size to the top 5000 features
#     token_pattern=r'\b\w+\b',    # Use words as tokens
#     sublinear_tf=True,           # Apply sublinear term frequency scaling
#     smooth_idf=True,             # Smooth IDF weights by adding one to document frequencies
#     norm='l2'                    # Normalize TF-IDF vectors to unit length
# )

# # Apply TF-IDF vectorization on the preprocessed text
# word_vectors = word_vectorizer.fit_transform(preprocessed_sent_tokens)


# # Generate response function
# def generate_response(user_input, text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     # Check if the user is asking for specific information
#     if 'case number' in user_input or 'case no' in user_input:
#         return extract_case_number(text)
#     elif 'governing law' in user_input:
#         return extract_governing_law(text)
#     elif 'final verdict' in user_input:
#         return extract_final_verdict(text)
#     elif 'party' in user_input:
#         return extract_parties(text)
#     elif 'date' in user_input:
#         return extract_date(text)
#     elif 'title of the case' in user_input or 'case title' in user_input:
#         return extract_case_title(text)
#     elif 'summary of the case' in user_input or 'case summary' in user_input:
#         return generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
#     elif 'name of the court' in user_input or 'court name' in user_input:
#         return extract_court_name(text)
#     elif 'article' in user_input:
#         return extract_articles_sections(text)
#     else:
#         # Handle other types of questions
#         return "I'm sorry, I don't know."




# def extract_case_number(text):
#     # Regular expression pattern for matching case numbers, including ranges
#     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?(?<!\d\/)\b\d{2,}(?:-\d{2,})?(?:\s*(?:and|,)\s*\d{2,})*(?:\s*(?:of|OF)\s*|\s*\/\s*)\d{4}\b(?!\/\d{2})\b'
    
#     # Find all matches of the pattern in the text
#     case_numbers = re.findall(pattern, text)
    
#     # Return a list of unique case numbers
#     return list(set(case_numbers))

# def extract_governing_law(text):
#     # Define keywords for criminal law and civil law
#     criminal_law_keywords = [
#         'criminal', 'crime', 'offense', 'prosecution', 'defendant', 'felony', 'misdemeanor'
#     ]
#     civil_law_keywords = [
#         'civil', 'plaintiff', 'defendant', 'contract', 'tort', 'property', 'family', 'probate'
#     ]
    
#     # Tokenize the text and tag the parts of speech
#     tokens = word_tokenize(text.lower())
#     tagged_tokens = pos_tag(tokens)
    
#     # Extract nouns and adjectives from the tagged tokens
#     nouns_adjectives = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]
    
#     # Check for criminal law keywords
#     for keyword in criminal_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Criminal Law"
    
#     # Check for civil law keywords
#     for keyword in civil_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Civil Law"
    
#     return "Governing law not identified"

# def extract_final_verdict(text):
#     # Define regular expressions to match common patterns for final verdicts and dates
#     verdict_patterns = [
#         r'final\s*verdict[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'final\s*verdict[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         #r'DATED[:\s]*\d{2}\.\d{2}\.\d{4}',  # Pattern for DATED: dd.mm.yyyy
#     ]
    
#     # Search for the patterns in the text
#     final_verdict = None
#     for pattern in verdict_patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             final_verdict = match.group(0).strip()
#             break
    
#     if not final_verdict:
#         # If no final verdict date is found, try to find a date in the title of the case
#         title_pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
#         title_match = re.search(title_pattern, text, re.IGNORECASE)
#         if title_match:
#             final_verdict = title_match.group(2).strip()
    
#     return final_verdict


# def extract_parties(text):        
#     # New patterns for petitioner and respondents
#     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
#     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)

#     petitioner_matches = new_petitioner_pattern.findall(text)
#     respondents_matches = new_respondents_pattern.findall(text)

#     if petitioner_matches and respondents_matches:
#         petitioners = [match.strip() for match in petitioner_matches]
#         respondents = []
#         for match in respondents_matches:
#             respondents_list = match.strip().split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
#         petitioners = "\n".join([f"{petitioner}" for petitioner in petitioners])
#         respondents = "\n".join([f"{respondent}" for respondent in respondents])
#         return f"Petitioners:\n{petitioners}\n\nRespondents:\n{respondents}"

       
#     new_pattern_7 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_7.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
        
#     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
#     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)\s*(?=$|\n)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
#     petitioners = petitioner_pattern.findall(text)
#     respondents = respondent_pattern.findall(text)
    
#     if petitioners and respondents:
#         petitioners = [p.strip() for p in petitioners]
#         respondents = [r.strip() for r in respondents]
#         parties = []
#         for petitioner, respondent in zip(petitioners, respondents):
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
    
    
#     new_pattern_6 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_6.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
    
#     new_pattern_2_revised = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*(?:Appellants|Petitioners|Complainants|Plaintiffs)\s*.*?\n\s*[-versus-]+\n\s*([^\n\r]+?)\s*\.\.\.\s*(?:Respondent|Defendant|Accused|Appellees)',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_2_revised.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     specific_pattern = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = specific_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     new_pattern = re.compile(
#     r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = new_pattern.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
   
#     new_pattern_2 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = new_pattern_2.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     new_pattern_3 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_3.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.append("\n".join([resp.strip() for resp in respondents_list if resp.strip()]))
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
    
#     new_pattern_4 = re.compile(
#         r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_4.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
     
#     additional_new_pattern = re.compile(
#         r'\[.*?\]\s*(.*?)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = additional_new_pattern.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
            
#             # Extract only the relevant petitioner information and remove leading spaces
#             petitioner_parts = petitioner.split('\n')
#             relevant_petitioner = '\n'.join(part.strip() for part in petitioner_parts if '.' not in part[:3])
            
#             petitioners_list.append(relevant_petitioner)
#             respondents_list.append("\n".join(respondents))
        
#         petitioners_text = "\n\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    

  
#     new_pattern_5 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = new_pattern_5.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents.append(match[1].strip())
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
        
#     pattern_11_1 = re.compile(
#         r'\[.*?\]\s*(.*?)\s*\.{3,}\s*Applicant\s*.*?Versus\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_11_1.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             # Remove leading/trailing spaces and align to left
#             applicant_lines = [line.strip() for line in match[0].split('\n') if line.strip()]
#             applicant = '\n'.join(applicant_lines)
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Applicant:\n{applicant}\n\nRespondents:\n{respondents_text}")
#         return "\n\n".join(parties)
    
    
#     pattern_11 = re.compile(
#         r'([^\n\r]+)\s*\.{3,}\s*Applicant\s*.*?VERSUS\s*([\s\S]*?)\s*\.{3,}\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_11.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Petitioner:{petitioner}\n\nRespondents:{respondents_text}")
#         return "\n\n".join(parties)
    
    
    
#     pattern_12 = re.compile(
#         r'([^\n\r]+)\s*\.{1,}\s*Appellant\s*.*?Versus\s*([^\n\r]+)\s*\.{1,}\s*Respondent',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_12.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_17 = re.compile(
#         r'([^\n\r]+?)\s*\.{4,}\s*PETITIONER\s*Versus\s*([\s\S]+?)\s*\.{4,}\s*RESPONDENTS',
#         re.IGNORECASE
#     )
    
#     matches = pattern_17.findall(text)
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondent = match[1].strip()
#             parties.append(f"Petitioner: {petitioner}\n\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     pattern_19 = re.compile(
#     r'([A-Z][A-Z\s&]+\.)\s*…\s*PETITIONERS\s*VERSUS\s*([A-Z][A-Z\s&]+\.)\s*…\s*RESPONDENTS',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_19.findall(text)
#     if matches:
#         petitioners = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n\n".join(respondents_list)
#         return f"Petitioner:\n{petitioner}\n\nRespondents:\n{respondents_text}"
    
#     pattern_20 = re.compile(
#     r'([A-Z\s’&.]+)\s*…+\s*APPELLANTS\s*VERSUS\s*([A-Z\s’&.]+)\s*\.+\s*RESPONDENTS',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_20.findall(text)
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip()
#             respondents = match[1].strip()
            
#             appellants_list.append(appellants)
#             respondents_list.append(respondents)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
#         return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents}"
    
#     pattern_21 = re.compile(
#     r'([A-Z][A-Za-z\s&.]+\.)\s*…\s*Appellants\s*Versus\s*([A-Z][A-Za-z\s&.]+\.)\s*…\s*Respondents',
#     re.IGNORECASE
#     )
    
#     # Find all matches in the text
#     matches = pattern_21.findall(text)
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip()
#             respondents = match[1].strip()
            
#             appellants_list.append(appellants)
#             respondents_list.append(respondents)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
#         return f"Appellants:\n{appellants}\n\nRespondents:\n{respondents}"
    
#     pattern_appellant = re.compile(
#         r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_appellant.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_23 = re.compile(
#     r'(?:\d+\s+of\s+\d+\s+)?(?P<petitioner>[\w\s]+?)\s*…\s*Petitioner\s+versus\s+(?P<respondent>[\w\s]+?\s+and\s+others)\s*…\s*Respondents',
#     re.IGNORECASE
#     )
    
#     matches = pattern_23.findall(text)
    
#     if matches:
#         seen = set()
#         results = []
#         for match in matches:
#             petitioner, respondent = match
#             # Remove all numbers and 'of' from petitioner name
#             petitioner = re.sub(r'\d+\s+of\s+\d+\s+', '', petitioner).strip()
#             # Remove any leading/trailing whitespace and newlines
#             petitioner = ' '.join(petitioner.split())
#             if petitioner not in seen:
#                 seen.add(petitioner)
#                 results.append(f"Petitioner: {petitioner}\nRespondent: {respondent.strip()}")
        
#         return "\n\n".join(results)

#     pattern_10_1 = re.compile(
#         r'((?:\d+\.[^\n\r]+\s*)+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_10_1.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip().split('\n')
#             petitioner = [pet.strip() for pet in petitioner if pet.strip()]
#             petitioner_text = "\n".join(petitioner)
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner_text)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_10 = re.compile(
#     r'([^\n\r]+)\s*\.{3,}\s*.*?Vs\s*([\s\S]*?)\s*\.{3,}\s*R',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_10.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
        
#     pattern_13 = re.compile(
#     r'([^\n\r]+)\s*\.{2,}\s*Petitione\s*.*?vs\s*([\s\S]*?)\s*\.{2,}\s*Respondents',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_13.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_14 = re.compile(
#         r'(?:.*?WP\(C\)\sNo\.\s\d+\sof\s\d+\s+Date\sof\sDecision:.*?\n)(.*?)\s*:::\s*Petitioner\s*-\s*Vs\s*-\s*([\s\S]+?)\s*:::\s*Respondents',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_14.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             # Split petitioner info into lines, strip each line, and rejoin
#             petitioner_lines = match[0].strip().split('\n')
#             petitioner = '\n'.join(line.strip() for line in petitioner_lines if line.strip())
            
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_15_3 = re.compile(
#         r'((?:\d+\.\s?[^\n\r]+\s*)+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:versus)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15_3.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             petitioners = petitioner.split('\n')
#             petitioners = [pet.strip() for pet in petitioners if pet.strip()]
#             petitioners_text = "\n".join(petitioners)

#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioners_text)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
    
#     pattern_15_2 = re.compile(
#         r'Between:\s*(.*?)\s*\.{3,}\s*APPELLANT(?:\(S\))?\s*(?:AND)?\s*(.*?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15_2.findall(text)
    
#     if matches:
#         appellants_list = []
#         respondents_list = []
        
#         for match in matches:
#             appellants = match[0].strip().split('\n')
#             respondents = match[1].strip().split('\n')
            
#             appellants = [app.strip() for app in appellants if app.strip()]
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
            
#             appellants_text = "\n".join(appellants)
#             respondents_text = "\n".join(respondents)
            
#             appellants_list.append(appellants_text)
#             respondents_list.append(respondents_text)
        
#         appellants_text = "\n".join(appellants_list)
#         respondents_text = "\n".join(respondents_list)
        
#         return f"Appellants:\n{appellants_text}\n\nRespondents:\n{respondents_text}"
    
    
#     pattern_15_1 = re.compile(
#         r'\b(?:Between:|Petitioner:|Petitioners:)\s*(.*?)\s*\.\.\.\s*Petitioner\s*(?:Versus|AND|Vs\.|v\.)\s*(.*?)\s*\.\.\.\s*(?:Respondents?|Respondent)',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_15_1.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         # Ensure no unnecessary leading or trailing whitespace in the final output
#         petitioners_text = "\n".join([line.strip() for line in petitioners_text.split('\n')])
#         respondents_text = "\n".join([line.strip() for line in respondents_text.split('\n')])
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_15_4 = re.compile(
#         r'Between:\s*(.*?)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15_4.findall(text)
    
#     if matches:
#         parties = []
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
#             parties.append(f"Petitioner:{petitioner}\n\nRespondents:{respondents_text}")
#         return "\n\n".join(parties)
    
#     # pattern_15_4 = re.compile(
#     #     r'Between:\s*(.*?)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
#     #     re.IGNORECASE | re.DOTALL
#     # )
    
#     # matches = pattern_15_4.findall(text)
    
#     # if matches:
#     #     petitioners_list = []
#     #     respondents_list = []
        
#     #     for match in matches:
#     #         petitioner = match[0].strip()
#     #         respondents = match[1].strip().split('\n')
#     #         respondents = [resp.strip() for resp in respondents if resp.strip()]
#     #         respondents_text = "\n".join(respondents)
            
#     #         petitioners_list.append(petitioner)
#     #         respondents_list.append(respondents_text)
        
#     #     petitioners_text = "\n".join(petitioners_list)
#     #     respondents_text = "\n\n".join(respondents_list)
        
#     #     return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"

    
#     pattern_15 = re.compile(
#         r'([^\n\r]+)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_15.findall(text)
    
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_16 = re.compile(
#     r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*([\s\S]+?)\s*\.\.\s*Responden',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_16.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_18 = re.compile(
#     r'((?:\d+\.[A-Z][.\w\s]+\s*)+)\.\.Pet.*?vs\s+((?:\d+\..*?\.\s*)+)\.\.\s*Respo',
#     re.IGNORECASE | re.DOTALL
#     )
    
#     # Find all matches in the text
#     matches = pattern_18.findall(text)
#     if matches:
#         petitioners_list = []
#         respondents_list = []
        
#         for match in matches:
#             petitioner = match[0].strip()
#             respondents = match[1].strip().split('\n')
#             respondents = [resp.strip() for resp in respondents if resp.strip()]
#             respondents_text = "\n".join(respondents)
            
#             petitioners_list.append(petitioner)
#             respondents_list.append(respondents_text)
        
#         petitioners_text = "\n".join(petitioners_list)
#         respondents_text = "\n\n".join(respondents_list)
        
#         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
#     pattern_22 = re.compile(
#         r'BETWEEN:-\s*(?P<petitioner>.*?)\s*\(BY.*?\)\s*AND\s*(?P<respondents>(?:\d+\..*?)+)(?:\(BY|$)',
#         re.DOTALL | re.IGNORECASE
#     )
    
#     match = pattern_22.search(text)
#     if match:
#         petitioner = match.group('petitioner').strip()
#         respondents_text = match.group('respondents').strip()
        
#         # Format the petitioner
#         petitioner_formatted = "\n".join(line.strip() for line in petitioner.split('\n'))
        
#         # Format the respondents
#         respondents = re.findall(r'\d+\.(.*?)(?=\d+\.|\Z)', respondents_text, re.DOTALL)
#         respondents_formatted = "\n".join(f"{i+1}. {' '.join(line.strip() for line in resp.split())}" 
#                                           for i, resp in enumerate(respondents))
        
#         return f"Petitioner:\n{petitioner_formatted}\n\nRespondents:\n{respondents_formatted}"
    
#     specific_pattern_1 = re.compile(
#         r'([^\n\r]+?)\s*\.\.\.\s*Pe\s*versus\s*([\s\S]+?)\s*\.\.\.\s*Re', 
#         re.IGNORECASE | re.DOTALL
#     )

#     matches = specific_pattern_1.findall(text)
#     if matches:
#         petitioners = []
#         respondents = []
#         for match in matches:
#             petitioners.append(match[0].strip())
#             respondents_text = match[1].strip()
#             respondents_list = respondents_text.split('\n')
#             respondents.extend([resp.strip() for resp in respondents_list if resp.strip()])
        
#         petitioners_text = "\n".join(petitioners)
#         respondents_text = "\n".join(respondents)
#         return f"Parties:\nPetitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"
    
# #2    
#     pattern_ellipsis = re.compile(
#         r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_ellipsis.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     pattern_dots = re.compile(
#         r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dots.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\n\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
    
#     pattern_dashes = re.compile(
#         r'([^\n\r]+?)\s*-\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*-\s*Respondents', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_dashes.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
#     pattern_provided = re.compile(
#         r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     matches = pattern_provided.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
    
#     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
    
#     matches = pattern_specific_parties.findall(text)
    
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
           
#     pattern_specific_parties_1 = re.compile(r'([^\n\r]+?)\s*VS\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)
#     matches = pattern_specific_parties_1.findall(text) 
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             petitioner = petitioner.strip()
#             respondent = respondent.strip()
#             # Check if both petitioner and respondent are in uppercase
#             if petitioner.isupper() and respondent.isupper():
#                 parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
        
#         # Join and return the result
#         return "\n\n".join(parties)
         
#     return "Parties not found."

# # def extract_parties(text):    
# #     pattern_15_4 = re.compile(
# #         r'Between:\s*(.*?)\s*\.{3,}\s*PETITIONER(?:\(S\))?\s*(?:AND)?\s*([\s\S]+?)\s*\.{3,}\s*RESPONDENT(?:\(S\))?\S*',
# #         re.IGNORECASE | re.DOTALL
# #     )
    
# #     matches = pattern_15_4.findall(text)
    
# #     if matches:
# #         petitioners_list = []
# #         respondents_list = []
        
# #         for match in matches:
# #             petitioner = match[0].strip()
# #             respondents = match[1].strip().split('\n')
# #             respondents = [resp.strip() for resp in respondents if resp.strip()]
# #             respondents_text = "\n".join(respondents)
            
# #             petitioners_list.append(petitioner)
# #             respondents_list.append(respondents_text)
        
# #         petitioners_text = "\n".join(petitioners_list)
# #         respondents_text = "\n\n".join(respondents_list)
        
# #         return f"Petitioners:\n{petitioners_text}\n\nRespondents:\n{respondents_text}"


# def extract_date(text):
#     # Define regex pattern to match dates in various formats
#     date_pattern = (
#         r'(\b(?:[12][0-9]|3[01]|0?[1-9])(st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s\d{4}\b)|'
#         r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s(?:[12][0-9]|3[01]),?\s\d{4}\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b)|'
#         r'(\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.\d{4}\b)'
#     )
    
#     # Find all matches of the date pattern in the text
#     matches = re.findall(date_pattern, text)
    
#     # Flatten the list of tuples and remove empty strings
#     dates = [date for match in matches for date in match if date]
    
#     # Remove duplicates by converting to set and back to list
#     unique_dates = list(set(dates))
    
#     return unique_dates

# def extract_case_title(text):
#     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
#     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/0-9@]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/0-9]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/0-9@]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
    
#     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
#     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-0-9@]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-0-9@]+))'
    
#     # First try matching the original pattern
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     # If no match, try the alternative pattern
#     if not match:
#         match = re.search(pattern_alt, text, re.IGNORECASE)
    
#     if match:
#         title = match.group(1).strip()
#         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
#             date = match.group(2).strip()
#             return f"{title} on {date}"
#         else:
#             # Check for unwanted "Author" or other text in the title
#             if "Author" in title:
#                 title = title.split("Author")[0].strip()
#             return title
#     else:
#         return "Title and date not found"
    
# # def extract_case_title(text):
# #     # Updated pattern to match case titles with varying formats, including dots, colons, and additional parts
# #     pattern = r'((?:In\sRe:\s)?[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s+Through\s+[A-Za-z\s.,&()\'’\-/0-9]+)?(?:\s+\.\.\.\s+)?(?:vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-/0-9]+(?:\s*[:,]\s*[a-zA-Z\s.,&()\'’]*)*)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
  
# #     # Alternative pattern to match titles like "Aum Capital Market Pvt. Ltd vs Union Of India"
# #     pattern_alt = r'([A-Z][a-zA-Z\s.,&()\'’\-0-9]+(?:,\s*[A-Z]\d+)?(?:\s+vs\.?\s+[A-Z][a-zA-Z\s.,&()\'’\-0-9]+))'
    
# #     # First try matching the original pattern
# #     match = re.search(pattern, text, re.IGNORECASE)
    
# #     # If no match, try the alternative pattern
# #     if not match:
# #         match = re.search(pattern_alt, text, re.IGNORECASE)
    
# #     if match:
# #         title = match.group(1).strip()
# #         if len(match.groups()) > 1 and re.search(r'\s+on\s+', text, re.IGNORECASE):
# #             date = match.group(2).strip()
# #             return f"{title} on {date}"
# #         else:
# #             # Check for unwanted "Author" or other text in the title
# #             if "Author" in title:
# #                 title = title.split("Author")[0].strip()
# #             return title
# #     else:
# #         return "Title and date not found"

# def extract_court_name(text):
#     # Define a comprehensive pattern for court names, including spaces between letters
#     comprehensive_pattern = (
#         r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T\s*'
#         r'O\s*F\s*I\s*N\s*D\s*I\s*A|'
#         r'Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
#         r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
#     )
    
#     # Search for the comprehensive pattern in the text
#     match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
#     if match:
#         return match.group(0).strip()
#     else:
#         # Define a fallback pattern for court names, including spaces between letters
#         fallback_pattern = r'(?:S\s*U\s*P\s*R\s*E\s*M\s*E\s*C\s*O\s*U\s*R\s*T|Supreme|High|District) Court'
        
#         # Search for the fallback pattern in the text
#         match = re.search(fallback_pattern, text, re.IGNORECASE)
        
#         if match:
#             return match.group(0).strip()
#         else:
#             return "Court name not found"
        
# article_details = {
#     "226": "Power of High Courts to issue certain writs.",
#     "64": "The Vice-President to be ex officio Chairman of the Council of States.",
#     "1": "Name and territory of the Union.",
#     "2": "Admission or establishment of new States.",
#     "2A": "[Repealed.]",
#     "3": "Formation of new States and alteration of areas, boundaries or names of existing States.",
#     "4": "Laws made under articles 2 and 3 to provide for the amendment of the First and the Fourth Schedules and supplemental, incidental and consequential matters.",
#     "5": "Citizenship at the commencement of the Constitution.",
#     "6": "Rights of citizenship of certain persons who have migrated to India from Pakistan.",
#     "7": "Rights of citizenship of certain migrants to Pakistan.",
#     "8": "Rights of citizenship of certain persons of Indian origin residing outside India.",
#     "9": "Persons voluntarily acquiring citizenship of a foreign State not to be citizens.",
#     "10": "Continuance of the rights of citizenship.",
#     "11": "Parliament to regulate the right of citizenship by law.",
#     "13": "Laws inconsistent with or in derogation of the fundamental rights.",
#     "14": "Equality before law.",
#     "15": "Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth.",
#     "16": "Equality of opportunity in matters of public employment.",
#     "17": "Abolition of Untouchability.",
#     "18": "Abolition of titles.",
#     "19": "Protection of certain rights regarding freedom of speech, etc.",
#     "20": "Protection in respect of conviction for offences.",
#     "21": "Protection of life and personal liberty.",
#     "21A": "Right to education.",
#     "22": "Protection against arrest and detention in certain cases.",
#     "23": "Prohibition of traffic in human beings and forced labour.",
#     "24": "Prohibition of employment of children in factories, etc.",
#     "25": "Freedom of conscience and free profession, practice and propagation of religion.",
#     "26": "Freedom to manage religious affairs.",
#     "27": "Freedom as to payment of taxes for promotion of any particular religion.",
#     "28": "Freedom as to attendance at religious instruction or religious worship in certain educational institutions.",
#     "29": "Protection of interests of minorities.",
#     "30": "Right of minorities to establish and administer educational institutions.",
#     "31": "[Repealed.]",
#     "31A": "Saving of Laws providing for the acquisition of estates, etc.",
#     "31B": "Validation of certain Acts and Regulations.",
#     "31C": "Saving of laws giving effect to certain directive principles.",
#     "31D": "[Repealed.]",
#     "32": "Remedies for enforcement of rights conferred by this Part.",
#     "32A": "[Repealed.]",
#     "33": "Power of Parliament to modify the rights conferred by this Part in their application to Forces, etc.",
#     "34": "Restriction on rights conferred by this Part while martial law is in force in any area.",
#     "35": "Legislation to give effect to the provisions of this Part.",
#     "37": "Application of the principles contained in this Part.",
#     "38": "State to secure a social order for the promotion of the welfare of the people.",
#     "39": "Certain principles of policy to be followed by the State.",
#     "39A": "Equal justice and free legal aid.",
#     "40": "The organisation of village panchayats.",
#     "41": "Right to work, to education and to public assistance in certain cases.",
#     "42": "Provision for just and humane conditions of work and maternity relief.",
#     "43": "Living wage, etc., for workers.",
#     "43A": "Participation of workers in the management of industries.",
#     "43B": "Promotion of co-operative societies.",
#     "44": "Uniform civil code for the citizens.",
#     "45": "Provision for free and compulsory education for children.",
#     "46": "Promotion of educational and economic interests of Scheduled Castes, Scheduled Tribes and other weaker sections.",
#     "47": "Duty of the State to raise the level of nutrition and the standard of living and to improve public health.",
#     "48": "The organisation of agriculture and animal husbandry.",
#     "48A": "Protection and improvement of environment and safeguarding of forests and wildlife.",
#     "49": "Protection of monuments and places and objects of national importance.",
#     "50": "Separation of judiciary from the executive.",
#     "51": "Promotion of international peace and security.",
#     "51A": "Fundamental duties.",
#     "52": "The President of India.",
#     "53": "The executive power of the Union.",
#     "54": "Election of President.",
#     "55": "Manner of election of President.",
#     "56": "Term of office of President.",
#     "57": "Eligibility for re-election.",
#     "58": "Qualifications for election as President.",
#     "59": "Conditions of the President’s office.",
#     "60": "Oath or affirmation by the President.",
#     "61": "Procedure for impeachment of the President.",
#     "62": "Time of holding the election to fill the vacancy in the office of President and the term of office of person elected to fill the casual vacancy.",
#     "63": "The Vice-President of India.",
#     "64": "The Vice-President to be ex officio Chairman of the Council of States.",
#     "65": "The Vice-President to act as President or to discharge his functions during casual vacancies in the office, or during the absence, of President.",
#     "66": "Election of Vice-President.",
#     "67": "Term of office of Vice-President.",
#     "68": "Time of holding the election to fill the vacancy in the office of Vice-President and the term of office of person elected to fill the casual vacancy.",
#     "69": "Oath or affirmation by the Vice-President.",
#     "70": "Discharge of President’s functions in other contingencies.",
#     "71": "Matters relating to, or connected with, the election of a President or Vice-President.",
#     "72": "Power of President to grant pardons, etc., and to suspend, remit or commute sentences in certain cases.",
#     "73": "The extent of executive power of the Union.",
#     "74": "Council of Ministers to aid and advise the President.",
#     "75": "Other provisions as to Ministers.",
#     "76": "Attorney-General for India.",
#     "77": "Conduct of business of the Government of India.",
#     "78": "Duties of Prime Minister as respects the furnishing of information to the President, etc.",
#     "79": "Constitution of Parliament.",
#     "80": "Composition of the Council of States.",
#     "81": "Composition of the House of the People.",
#     "82": "Readjustment after each census.",
#     "83": "Duration of Houses of Parliament.",
#     "84": "Qualification for membership of Parliament.",
#     "85": "Sessions of Parliament, prorogation and dissolution.",
#     "86": "Right of President to address and send messages to Houses.",
#     "87": "Special address by the President.",
#     "88": "Rights of Ministers and Attorney-General as respects Houses.",
#     "89": "The Chairman and Deputy Chairman of the Council of States.",
#     "90": "Vacation and resignation of, and removal from, the office of Deputy Chairman.",
#     "91": "Power of the Deputy Chairman or other person to perform the duties of the office of, or to act as, Chairman.",
#     "92": "The Chairman or the Deputy Chairman not to preside while a resolution for his removal from office is under consideration.",
#     "93": "The Speaker and Deputy Speaker of the House of the People.",
#     "94": "Vacation and resignation of, and removal from, the offices of Speaker and Deputy Speaker.",
#     "95": "Power of the Deputy Speaker or other person to perform the duties of the office of, or to act as, Speaker.",
#     "96": "The Speaker or the Deputy Speaker not to preside while a resolution for his removal from office is under consideration.",
#     "97": "Salaries and allowances of the Chairman and Deputy Chairman and the Speaker and Deputy Speaker.",
#     "98": "Secretariat of Parliament.",
#     "99": "Oath or affirmation by members.",
#     "100": "Voting in Houses, power of Houses to act notwithstanding vacancies and quorum.",
#     "101": "Vacation of seats.",
#     "102": "Disqualifications for membership.",
#     "103": "Decision on questions as to disqualifications of members.",
#     "104": "Penalty for sitting and voting before making oath or affirmation under article 99 or when not qualified or when disqualified.",
#     "105": "Powers, privileges, etc., of the Houses of Parliament and of the members and committees thereof.",
#     "106": "Salaries and allowances of members.",
#     "107": "Provisions as to introduction and passing of Bills.",
#     "108": "Joint sitting of both Houses in certain cases.",
#     "109": "Special procedure in respect of Money Bills.",
#     "110": "Definition of “Money Bills”.",
#     "111": "Assent to Bills.",
#     "112": "Annual financial statement.",
#     "113": "Procedure in Parliament with respect to estimates.",
#     "114": "Appropriation Bills.",
#     "115": "Supplementary, additional or excess grants.",
#     "116": "Votes on account, votes of credit and exceptional grants.",
#     "117": "Special provisions as to financial Bills.",
#     "118": "Rules of procedure.",
#     "119": "Regulation by law of procedure in Parliament in relation to financial business.",
#     "120": "Language to be used in Parliament.",
#     "121": "Restriction on discussion in Parliament.",
#     "122": "Courts not to inquire into proceedings of Parliament.",
#     "123": "Power of President to promulgate Ordinances during recess of Parliament.",
#     "124": "Establishment and constitution of Supreme Court.",
#     "124A": "National Judicial Appointments Commission. (Declared unconstitutional by the Supreme Court, however not repealed by the Parliament)",
#     "124B": "Functions of Commission.",
#     "124C": "Power of Parliament to make law.",
#     "125": "Salaries, etc., of Judges.",
#     "126": "Appointment of acting Chief Justice.",
#     "127": "Appointment of ad hoc judges.",
#     "128": "Attendance of retired Judges at sittings of the Supreme Court.",
#     "129": "Supreme Court to be a court of record.",
#     "130": "Seat of Supreme Court.",
#     "131": "Original jurisdiction of the Supreme Court.",
#     "131A": "[Repealed.]",
#     "132": "Appellate jurisdiction of Supreme Court in appeals from High Courts in certain cases.",
#     "133": "Appellate jurisdiction of Supreme Court in appeals from High Courts in regard to Civil matters.",
#     "134": "Appellate jurisdiction of Supreme Court in regard to criminal matters.",
#     "134A": "Certificate for appeal to the Supreme Court.",
#     "135": "Jurisdiction and powers of the Federal Court under existing law to be exercisable by the Supreme Court.",
#     "136": "Special leave to appeal by the Supreme Court.",
#     "137": "Review of judgments or orders by the Supreme Court.",
#     "138": "Enlargement of the jurisdiction of the Supreme Court.",
#     "139": "Conferment on the Supreme Court of powers to issue certain writs.",
#     "139A": "Transfer of certain cases.",
#     "140": "Ancillary powers of Supreme Court.",
#     "141": "Law declared by Supreme Court to be binding on all courts.",
#     "142": "Enforcement of decrees and orders of Supreme Court and orders as to discovery, etc.",
#     "143": "Power of President to consult Supreme Court.",
#     "144": "Civil and judicial authorities to act in aid of the Supreme Court.",
#     "144A": "[Repealed.]",
#     "145": "Rules of Court, etc.",
#     "146": "Officers and servants and the expenses of the Supreme Court.",
#     "147": "Interpretation.",
#     "148": "Comptroller and Auditor-General of India.",
#     "149": "Duties and powers of the Comptroller and Auditor-General.",
#     "150": "Form of accounts of the Union and of the States."
# }


# section_details = {
#     "1": "(Introduction) Title and extent of operation of the Code",
#     "2": "(Introduction) Punishment of offences committed within India",
#     "3": "(Introduction) Punishment of offences committed beyond, but which by law may be tried within, India",
#     "4": "(Introduction) Extension of Code to extra-territorial offences",
#     "5": "(Introduction) Certain laws not to be affected by this Act",
#     "6": "(General explanations) Definitions in the Code to be understood subject to exceptions",
#     "7": "(General explanations) Sense of expression once explained",
#     "8": "(General explanations) Gender",
#     "9": "(General explanations) Number",
#     "10": "(General explanations) Man, Woman",
#     "11": "(General explanations) Person",
#     "12": "(General explanations) Public",
#     "13": "(General explanations) Queen",
#     "14": "(General explanations) Servant of Government",
#     "15": "(General explanations) British India",
#     "16": "(General explanations) Government of India",
#     "17": "(General explanations) Government",
#     "18": "(General explanations) India",
#     "19": "(General explanations) Judge",
#     "20": "(General explanations) Court of Justice",
#     "21": "(General explanations) Public Servant",
#     "22": "(General explanations) Moveable property",
#     "23": "(General explanations) Wrongful gain",
#     "24": "(General explanations) Dishonestly",
#     "25": "(General explanations) Fraudulently",
#     "26": "(General explanations) Reason to believe",
#     "27": "(General explanations) Property in possession of wife, clerk or servant",
#     "28": "(General explanations) Counterfeit",
#     "29": "(General explanations) Document",
#     "29A": "(General explanations) Electronic record",
#     "30": "(General explanations) Valuable security",
#     "31": "(General explanations) A will",
#     "32": "(General explanations) Words referring to acts include illegal omissions",
#     "33": "(General explanations) Act Omission",
#     "34": "(General explanations) Acts done by several persons in furtherance of common intention",
#     "35": "(General explanations) When such an act is criminal by reason of its being done with a criminal knowledge or intention",
#     "36": "(General explanations) Effect caused partly by act and partly by omission",
#     "37": "(General explanations) Co-operation by doing one of several acts constituting an offence",
#     "38": "(General explanations) Persons concerned in criminal act may be guilty of different offences",
#     "39": "(General explanations) Voluntarily",
#     "40": "(General explanations) Offence",
#     "41": "(General explanations) Special law",
#     "42": "(General explanations) Local law",
#     "43": "(General explanations) Illegal, Legally bound to do",
#     "44": "(General explanations) Injury",
#     "45": "(General explanations) Life",
#     "46": "(General explanations) Death",
#     "47": "(General explanations) Animal",
#     "48": "(General explanations) Vessel",
#     "49": "(General explanations) Year, Month",
#     "50": "(General explanations) Section",
#     "51": "(General explanations) Oath",
#     "52": "(General explanations) Good faith",
#     "52A": "(General explanations) Harbour",
#     "53": "Punishment",
#     "53A": "Construction of reference to transportation",
#     "54": "Commutation of sentence of death",
#     "55": "Commutation of sentence of imprisonment for life",
#     "55A": "Definition of appropriate Government",
#     "56": "Sentence of Europeans and Americans to penal servitude",
#     "57": "Fractions of terms of punishment",
#     "58": "Offenders sentenced to transportation how dealt with until transported",
#     "59": "Transportation instead of imprisonment",
#     "60": "Sentence may be (in certain cases of imprisonment) wholly or partly rigorous or simple",
#     "61": "Sentence of forfeiture of property",
#     "62": "Forfeiture of property, in respect of offenders punishable with death, transportation or imprisonment",
#     "63": "Amount of fine",
#     "64": "Sentence of imprisonment for non-payment of fine",
#     "65": "Limit to imprisonment for non-payment of fine, when imprisonment and fine awardable",
#     "66": "Description of imprisonment for non-payment of fine",
#     "67": "Imprisonment for non-payment of fine when offence punishable with fine only",
#     "68": "Imprisonment to terminate on payment of fine",
#     "69": "Termination of imprisonment on payment of proportional part of fine",
#     "70": "Fine levied within six years, or during imprisonment- Death not to discharge property from liability",
#     "71": "Limit of punishment of offence made up of several offences",
#     "73": "Solitary confinement",
#     "74": "Limit of solitary confinement",
#     "75": " Enhanced punishment for certain offences under Chapter XII or Chapter XVII after previous conviction",
#     "76": "Act done by a person bound, or by mistake of fact believing himself bound, by law",
#     "77": "Act of Judge when acting judicially",
#     "78": "Act done pursuant to the judgment or order of Court",
#     "79": "Act done by a person justified, or by mistake of fact believing himself justified, by law",
#     "80": "Accident in doing a lawful act",
#     "81": "Act likely to cause harm, but done without criminal intent, and to prevent other harm",
#     "82": "Act of a child under seven years of age",
#     "83": "Act of a child above seven and under twelve of immature understanding",
#     "84": "Act of a person of unsound mind",
#     "85": "Act of a person incapable of judgment by reason of intoxication caused against his will",
#     "86": "Offence requiring a particular intent of knowledge committed by one who is intoxicated",
#     "87": "Act not intended and not known to be likely to cause death or grievous hurt, done by consent",
#     "88": "Act not intended to cause death, done by consent in good faith for person's benefit",
#     "89": "Act done in good faith for benefit of child or insane person, by or by consent of guardian",
#     "90": "Consent known to be given under fear or misconception",
#     "91": "Exclusion of acts which are offences independently of harm caused",
#     "92": "Act done in good faith for benefit of a person without consent",
#     "93": "Communication made in good faith",
#     "94": "Act to which a person is compelled by threats",
#     "95": "Act causing slight harm",
#     "96": "Things done in private defence",
#     "97": "Right of private defence of the body and of property",
#     "98": "Right of private defence against the act of a person of unsound mind, etc.",
#     "99": "Act against which there is no right of private defence",
#     "100": "When the right of private defence of the body extends to causing death",
#     "101": "When such right extends to causing any harm other than death",
#     "102": "Commencement and continuance of the right of private defence of the body",
#     "103": "When the right of private defence of property extends to causing death",
#     "104": "When such right extends to causing any harm other than death",
#     "105": "Commencement and continuance of the right of private defence of property",
#     "106": "Right of private defence against deadly assault when there is risk of harm to innocent person",
#     "107": "Abetment of a thing",
#     "108": "Abettor",
#     "108A": "Abetment in India of offences outside India",
#     "110": "Punishment of abetment if person abetted does act with different intention from that of abettor",
#     "111": "Liability of abettor when one act abetted and different act done",
#     "112": "Abettor when liable to cumulative punishment for act abetted and for act done",
#     "113": " Liability of abettor for an effect caused by the act abetted different from that intended by the abettor",
#     "114": "Abettor present when offence is committed",
#     "115": "Abetment of offence punishable with death or imprisonment for life-if offence not committed",
#     "116": "Abetment of offence punishable with imprisonment-if offence be not committed",
#     "117": "Abetting commission of offence by the public or by more than ten persons",
#     "118": "Concealing design to commit offence punishable with death or imprisonment for life",
#     "119": "Public servant concealing design to commit offence which it is his duty to prevent",
#     "120": "Concealing design to commit offence punishable with imprisonment",
#     "120A": "Definition of criminal conspiracy",
#     "120B": "Punishment of criminal conspiracy",
#     "121": "Waging, or attempting to wage war, or abetting waging of war, against the Government of India",
#     "121A": "Conspiracy to commit offences punishable by section 121",
#     "122": "Collecting arms, etc., with intention of waging war against the Government of India",
#     "123": "Concealing with intent to facilitate design to wage war",
#     "124": " Assaulting President, Governor, etc., with intent to compel or restrain the exercise of any lawful power",
#     "124A": "Sedition",
#     "125": "Waging war against any Asiatic Power in alliance with the Government of India",
#     "126": "Committing depredation on territories of Power at peace with the Government of India",
#     "127": "Receiving Property taken by war on depredation mention in Sections 125 and 126",
#     "128": "Public servant voluntarily allowing prisoner of State or war to escape",
#     "129": "Public servant negligently suffering such prisoner to escape",
#     "130": "Aiding escape of, rescuing or harboring such prisoner",
#     "131": "Abetting mutiny, or attempting to seduce a soldier, sailor or airman from his duty",
#     "132": "Abetment of mutiny, if mutiny is committed in consequence thereof",
#     "133": " Abetment of assault by soldier, sailor or airman on his superior officer, when in execution of his office",
#     "134": "Abetment of such assault, if the assault is committed",
#     "135": "Abetment of desertion of soldier, sailor or airman",
#     "136": "Harbouring deserter",
#     "137": "Deserter concealed on board merchant vessel through negligence of master",
#     "138": "Abetment of act of insubordination by soldier, sailor or airman",
#     "138A": "Application of foregoing sections to the Indian Marine Service",
#     "139": "Persons subject to certain Acts",
#     "140": "Wearing garb or carrying token used by soldier, sailor or airman",
#     "141": "Unlawful assembly",
#     "142": "Being member of unlawful assembly",
#     "143": "Punishment",
#     "144": "Joining unlawful assembly armed with deadly weapon",
#     "145": "Joining or continuing in unlawful assembly, knowing it has been commanded to disperse",
#     "146": "Rioting",
#     "147": "Punishment for rioting",
#     "148": "Rioting, armed with deadly weapon",
#     "149": "Every member of unlawful assembly guilty of offence committed in prosecution of common object",
#     "150": "Hiring, or conniving at hiring, of persons to join unlawful assembly",
#     "151": "Knowingly joining or continuing in assembly of five or more persons after it has been commanded to disperse",
#     "152": "Assaulting or obstructing public servant when suppressing riot, etc.",
#     "153": "Wantonly giving provocation with intent to cause riot-if rioting be committed-if not committed",
#     "153A": "Promoting enmity between different groups on grounds of religion, race, place of birth, residence, language, etc., and doing acts prejudicial to maintenance of harmony",
#     "153AA": "Punishment for knowingly carrying arms in any procession or organizing, or holding or taking part in any mass drill or mass training with arms",
#     "153B": "Imputations, assertions prejudicial to national-integration",
#     "154": "Owner or occupier of land on which an unlawful assembly is held",
#     "155": "Liability of person for whose benefit riot is committed",
#     "156": "Liability of agent of owner of occupier for whose benefit riot is committed",
#     "157": "Harbouring persons hired for an unlawful assembly",
#     "158": "Being hired to take part in an unlawful assembly or riot",
#     "159": "Affray",
#     "160": "Punishment for committing affray",
#     "161-165A": "Repealed",
#     "166": "Public servant disobeying law, with intent to cause injury to any person",
#     "167": "Public servant framing an incorrect document with intent to cause injury",
#     "168": "Public servant unlawfully engaging in trade",
#     "169": "Public servant unlawfully buying or bidding for property",
#     "170": "Personating a public servant",
#     "171": "Wearing garb or carrying token used by public servant with fraudulent intent",
#     "171A": "Candidate, Electoral right defined",
#     "171B": "Bribery",
#     "171C": "Undue influence at elections",
#     "171D": "Personation at elections",
#     "171E": "Punishment for bribery",
#     "171F": "Punishment for undue influence or personation at an election",
#     "171G": "False statement in connection with an election",
#     "171H": "Illegal payments in connection with an election",
#     "171I": "Failure to keep election accounts",
#     "172": "Absconding to avoid service of summons or other proceeding",
#     "173": "Preventing service of summons or other proceeding, or preventing publication thereof",
#     "174": "Non-attendance in obedience to an order form public servant",
#     "174A": "Non-appearance in response to a proclamation under section 82 of Act 2 of 1974",
#     "175" : ". Omission to produce document or electronic record to public servant by person legally bound to produce it.",
#     "176": "Omission to give notice or information to public servant by person legally bound to give it",
#     "177": "Furnishing false information",
#     "178": "Refusing oath or affirmation when duly required by public servant to make it",
#     "179": "Refusing to answer public servant authorised to question",
#     "180": "Refusing to sign statement",
#     "181": "False statement on oath or affirmation to public servant or person authorised to administer an oath or affirmation",
#     "182": "False information, with intent to cause public servant to use his lawful power to the injury of another person",
#     "183": "Resistance to the taking of property by the lawful authority of a public servant",
#     "184": "Obstructing sale of property offered for sale by authority of public servant",
#     "185": "Illegal purchase or bid for property offered for sale by authority of public servant",
#     "186": "Obstructing public servant in discharge of public functions",
#     "187": "Omission to assist public servant when bound by law to give assistance",
#     "188": "Disobedience to order duly promulgated by public servant",
#     "189": "Threat of injury to public servant",
#     "190": "Threat of injury to induce person to refrain from applying for protection to public servant",
#     "191": "Giving false evidence",
#     "192": "Fabricating false evidence",
#     "193": "Punishment for false evidence",
#     "194": "Giving or fabricating false evidence with intent to procure conviction of capital offence",
#     "195": "Giving or fabricating false evidence with intent to procure conviction of offence punishable with imprisonment for life or imprisonment",
#     "195A": "Threatening any person to give false evidence",
#     "196": "Using evidence known to be false",
#     "197": "Issuing or signing false certificate",
#     "198": "Using as true a certificate known to be false",
#     "199": "False statement made in declaration which is by law receivable as evidence",
#     "200": "Using as true such declaration knowing it to be false",
#     "201": "Causing disappearance of evidence of offence, or giving false information to screen offender",
#     "202": "Intentional omission to give information of offence by person bound to inform",
#     "203": "Giving false information respecting an offence committed",
#     "204": "Destruction of document or electronic record to prevent its production as evidence",
#     "205": "False personation for purpose of act or proceeding in suit or prosecution",
#     "206": "Fraudulent removal or concealment of property to prevent its seizure as forfeited or in execution",
#     "207": "Fraudulent claim to property to prevent its seizure as forfeited or in execution",
#     "208": "Fraudulently suffering decree for sum not due",
#     "209": "Dishonestly making false claim in Court",
#     "210": "Fraudulently obtaining decree for sum not due",
#     "211": "False charge of offence made with intent to injure",
#     "212": "Harbouring offender",
#     "213": "Taking gift, etc., to screen an offender from punishment",
#     "214": "Offering gift or restoration of property in consideration of screening offender",
#     "215": "Taking gift to help to recover stolen property, etc.",
#     "216": "Harbouring offender who has escaped from custody or whose apprehension has been ordered",
#     "216A": "Penalty for harbouring robbers or dacoits",
#     "216B": "Definition of 'harbour' in sections 212, 216 and 216A",
#     "217": "Public servant disobeying direction of law with intent to save person from punishment or property from forfeiture",
#     "218": "Public servant framing incorrect record or writing with intent to save person from punishment or property from forfeiture",
#     "219": "Public servant in judicial proceeding corruptly making report, etc., contrary to law",
#     "220": "Commitment for trial or confinement by person having authority who knows that he is acting contrary to law",
#     "221": "Intentional omission to apprehend on the part of public servant bound to apprehend",
#     "222": "Intentional omission to apprehend on the part of public servant bound to apprehend person under sentence or lawfully committed",
#     "223": "Escape from confinement or custody negligently suffered by public servant",
#     "224": "Resistance or obstruction by a person to his lawful apprehension",
#     "225": "Resistance or obstruction to lawful apprehension of another person",
#     "225A": "Omission to apprehend, or sufferance of escape, on part of public servant, in cases not otherwise, provided for",
#     "225B": "Resistance or obstruction to lawful apprehension, or escape or rescue in cases not otherwise provided for",
#     "226": "Unlawful return from transportation",
#     "227": "Violation of condition of remission of punishment",
#     "228": "Intentional insult or interruption to public servant sitting in judicial proceeding",
#     "228A": "Disclosure of identity of the victim of certain offences etc",
#     "229": "Personation of a juror or assessor",
#     "229A": "Failure by person released on bail or bond to appear in Court",
#     "230": "Coin defined",
#     "231": "Counterfeiting coin",
#     "232": "Counterfeiting Indian coin",
#     "233": "Making or selling instrument for counterfeiting coin",
#     "234": "Making or selling instrument for counterfeiting Indian coin",
#     "235": "Possession of instrument, or material for the purpose of using the same for counterfeiting coin",
#     "236": "Abetting in India the counterfeiting out of India of coin",
#     "237": "Import or export of counterfeit coin",
#     "238": "Import or export of counterfeits of the India coin",
#     "239": "Delivery of coin, possessed with knowledge that it is counterfeit",
#     "240": "Delivery of Indian coin, possessed with knowledge that it is counterfeit",
#     "241": "Delivery of coin as genuine, which, when first possessed, the deliverer did not know to be counterfeit",
#     "242": "Possession of counterfeit coin by person who knew it to be counterfeit when he became possessed thereof",
#     "243": "Possession of Indian coin by person who knew it to be counterfeit when he became possessed thereof",
#     "244": "Person employed in mint causing coin to be of different weight or composition from that fixed by law",
#     "245": "Unlawfully taking coining instrument from mint",
#     "246": "Fraudulently or dishonestly diminishing weight or altering composition of coin",
#     "247": "Fraudulently or dishonestly diminishing weight or altering composition of Indian coin",
#     "248": "Altering appearance of coin with intent that it shall pass as coin of different description",
#     "249": "Altering appearance of Indian coin with intent that it shall pass as coin of different description",
#     "250": "Delivery of coin, possessed with knowledge that it is altered",
#     "251": "Delivery of Indian coin, possessed with knowledge that it is altered",
#     "252": "Possession of coin by person who knew it to be altered when he became possessed thereof",
#     "253": "Possession of Indian coin by person who knew it to be altered when he became possessed thereof",
#     "254": "Delivery of coin as genuine, which, when first possessed, the deliverer did not know to be altered",
#     "255": "Counterfeiting Government stamp",
#     "256": "Having possession of instrument or material for counterfeiting Government stamp",
#     "257": "Making or selling instrument for counterfeiting Government stamp",
#     "259": "Having possession of counterfeit Government stamp",
#     "260": "Using as genuine a Government stamp known to be a counterfeit",
#     "261": "Effacing, writing from substance bearing Government stamp, or removing from document a stamp used for it, with intent to cause loss to Government",
#     "262": "Using Government stamp known to have been before used",
#     "263": "Erasure of mark denoting that stamp has been used",
#     "263A": "Prohibition of fictitious stamps",
#     "264": "Fraudulent use of false instrument for weighing",
#     "265": "Fraudulent use of false weight or measure",
#     "266": "Being in possession of false weight or measure",
#     "267": "Making or selling false weight or measure",
#     "268": "Public nuisance",
#     "269": "Negligent act likely to spread infection of disease dangerous to life",
#     "270": "Malignant act likely to spread infection of disease dangerous to life",
#     "271": "Disobedience to quarantine rule",
#     "272": "Adulteration of food or drink intended for sale",
#     "273": "Sale of noxious food or drink",
#     "274": "Adulteration of drugs",
#     "275": "Sale of adulterated drugs",
#     "276": "Sale of drug as a different drug or preparation",
#     "277": "Fouling water of public spring or reservoir",
#     "278": "Making atmosphere noxious to health",
#     "279": "Rash driving or riding on a public way",
#     "280": "Rash navigation of vessel",
#     "281": "Exhibition of false light, mark or buoy",
#     "282": "Conveying person by water for hire in unsafe or overloaded vessel",
#     "283": "Danger or obstruction in public way or line of navigation",
#     "284": "Negligent conduct with respect to poisonous substance",
#     "285": "Negligent conduct with respect to fire or combustible matter",
#     "286": "Negligent conduct with respect to explosive substance",
#     "287": "Negligent conduct with respect to machinery",
#     "288": "Negligent conduct with respect to pulling down or repairing buildings",
#     "289": "Negligent conduct with respect to animal",
#     "290": "Punishment for public nuisance in cases not otherwise provided for",
#     "291": "Continuance of nuisance after injunction to discontinue",
#     "292": "Sale, etc., of obscene books, etc",
#     "293": "Sale, etc., of obscene objects to young person",
#     "294": "Obscene acts and songs",
#     "294A": "Keeping lottery office",
#     "295": "Injuring or defiling place of worship with intent to insult the religion of any class",
#     "295A": "Deliberate and malicious acts intended to outrage religious feelings of any class by insulting its religion or religious beliefs",
#     "296": "Disturbing religious assembly",
#     "297": "Trespassing on burial places, etc",
#     "298": "Uttering words, etc., with deliberate intent to wound religious feelings",
#     "299": "Culpable homicide",
#     "300": "Murder",
#     "301": "Culpable homicide by causing death of person other than person whose death was intended",
#     "302": "Punishment for murder",
#     "303": "Punishment for murder by life-convict",
#     "304": "Punishment for culpable homicide not amounting to murder",
#     "304A": "Causing death by negligence",
#     "304B": "Dowry death",
#     "305": "Abetment of suicide of child or insane person",
#     "306": "Abetment of suicide",
#     "307": "Attempt to murder",
#     "308": "Attempt to commit culpable homicide",
#     "309": "Attempt to commit suicide",
#     "310": "Thug",
#     "311": "Punishment",
#     "312": "Causing miscarriage",
#     "313": "Causing miscarriage without woman’s consent",
#     "314": "Death caused by act done with intent to cause miscarriage",
#     "315": "Act done with intent to prevent child being born alive or to cause it to die after birth",
#     "316": "Causing death of quick unborn child by act amounting to culpable homicide",
#     "317": "Exposure and abandonment of child under twelve years, by parent or person having care of it",
#     "318": "Concealment of birth by secret disposal of dead body",
#     "319": "Hurt",
#     "320": "Grievous hurt",
#     "321": "Voluntarily causing hurt",
#     "322": "Voluntarily causing grievous hurt",
#     "323": "Punishment for voluntarily causing hurt",
#     "324": "Voluntarily causing hurt by dangerous weapons or means",
#     "325": "Punishment for voluntarily causing grievous hurt",
#     "326": "Voluntarily causing grievous hurt by dangerous weapons or means",
#     "326A": "Voluntarily causing grievous hurt by use of acid, etc",
#     "326B": "Voluntarily throwing or attempting to throw acid",
#     "327": "Voluntarily causing hurt to extort property, or to constrain to an illegal act",
#     "328": "Causing hurt by means of poison, etc., with intent to commit an offence",
#     "329": "Voluntarily causing grievous hurt to extort property, or to constrain to an illegal act",
#     "330": "Voluntarily causing hurt to extort confession, or to compel restoration of property",
#     "331": "Voluntarily causing grievous hurt to extort confession, or to compel restoration of property",
#     "332": "Voluntarily causing hurt to deter public servant from his duty",
#     "333": "Voluntarily causing grievous hurt to deter public servant from his duty",
#     "334": "Voluntarily causing hurt on provocation",
#     "335": "Voluntarily causing grievous hurt on provocation",
#     "336": "Act endangering life or personal safety of others",
#     "337": "Causing hurt by act endangering life or personal safety of others",
#     "338": "Causing grievous hurt by act endangering life or personal safety of others",
#     "339": "Wrongful restraint",
#     "340": "Wrongful confinement",
#     "341": "Punishment for wrongful restraint",
#     "342": "Punishment for wrongful confinement",
#     "343": "Wrongful confinement for three or more days",
#     "344": "Wrongful confinement for ten or more days",
#     "345": "Wrongful confinement of person for whose liberation writ has been issued",
#     "346": "Wrongful confinement in secret",
#     "347": "Wrongful confinement to extort property, or constrain to illegal act",
#     "348": "Wrongful confinement to extort confession, or compel restoration of property",
#     "349": "Force",
#     "350": "Criminal force",
#     "351": "Assault",
#     "352": "Punishment for assault or criminal force otherwise than on grave provocation",
#     "353": "Assault or criminal force to deter public servant from discharge of his duty",
#     "354": "Assault or criminal force to woman with intent to outrage her modesty",
#     "354A": "Sexual harassment and punishment for sexual harassment",
#     "354B": "Assault or use of criminal force to woman with intent to disrobe",
#     "354C": "Voyeurism",
#     "354D": "Stalking",
#     "355": "Assault or criminal force with intent to dishonour person, otherwise than on grave provocation",
#     "356": "Assault or criminal force in attempt to commit theft of property carried by a person",
#     "357": "Assault or criminal force in attempt wrongfully to confine a person",
#     "358": "Assault or criminal force on grave provocation",
#     "359": "Kidnapping",
#     "360": "Kidnapping from India",
#     "361": "Kidnapping from lawful guardianship",
#     "362": "Abduction",
#     "363": "Punishment for kidnapping",
#     "363A": "Kidnapping or maiming a minor for purposes of begging",
#     "364": "Kidnapping or abducting in order to murder",
#     "364A": "Kidnapping for ransom, etc",
#     "365": "Kidnapping or abducting with intent secretly and wrongfully to confine person",
#     "366": "Kidnapping, abducting or inducing woman to compel her marriage, etc",
#     "366A": "Procuration of minor girl",
#     "366B": "Importation of girl from foreign country",
#     "367": "Kidnapping or abducting in order to subject person to grievous hurt, slavery, etc",
#     "368": "Wrongfully concealing or keeping in confinement, kidnapped or abducted person",
#     "369": "Kidnapping or abducting child under ten years with intent to steal from its person",
#     "370": "Trafficking of persons",
#     "370A": "Exploitation of a trafficked person",
#     "371": "Habitual dealing in slaves",
#     "372": "Selling minor for purposes of prostitution, etc",
#     "373": "Buying minor for purposes of prostitution, etc",
#     "374": "Unlawful compulsory labour",
#     "375": "Rape",
#     "376": "Punishment for rape",
#     "376A": "Intercourse by a man with his wife during separation",
#     "376B": "Intercourse by public servant with woman in his custody",
#     "376C": "Intercourse by superintendent of jail, remand home, etc",
#     "376D": "Intercourse by any member of the management or staff of a hospital with any woman in that hospital",
#     "376E": "Punishment for repeat offenders",
#     "377": "Unnatural offences",
#     "378": "Theft",
#     "379": "Punishment for theft",
#     "380": "Theft in dwelling house, etc",
#     "381": "Theft by clerk or servant of property in possession of master",
#     "382": "Theft after preparation made for causing death, hurt or restraint in order to the committing of the theft",
#     "383": "Extortion",
#     "384": "Punishment for extortion",
#     "385": "Putting person in fear of injury in order to commit extortion",
#     "386": "Extortion by putting a person in fear of death or grievous hurt",
#     "387": "Putting a person in fear of death or of grievous hurt, in order to commit extortion",
#     "388": "Extortion by threat of accusation of an offence punishable with death or imprisonment for life, etc",
#     "389": "Putting person in fear of accusation of offence, in order to commit extortion",
#     "390": "Robbery",
#     "391": "Dacoity",
#     "392": "Punishment for robbery",
#     "393": "Attempt to commit robbery",
#     "394": "Voluntarily causing hurt in committing robbery",
#     "395": "Punishment for dacoity",
#     "396": "Dacoity with murder",
#     "397": "Robbery or dacoity, with attempt to cause death or grievous hurt",
#     "398": "Attempt to commit robbery or dacoity when armed with deadly weapon",
#     "399": "Making preparation to commit dacoity",
#     "400": "Punishment for belonging to a gang of dacoits",
#     "401": "Punishment for belonging to a gang of thieves",
#     "402": "Assembling for purpose of committing dacoity",
#     "403": "Dishonest misappropriation of property",
#     "404": "Dishonest misappropriation of property possessed by deceased person at the time of his death",
#     "405": "Criminal breach of trust",
#     "406": "Punishment for criminal breach of trust",
#     "407": "Criminal breach of trust by carrier, etc",
#     "408": "Criminal breach of trust by clerk or servant",
#     "409": "Criminal breach of trust by public servant, or by banker, merchant or agent",
#     "410": "Stolen Property",
#     "411": "Dishonestly receiving stolen property",
#     "412": "Dishonestly receiving property stolen in the commission of a dacoity",
#     "413": "Habitually dealing in stolen property",
#     "414": "Assisting in concealment of stolen property",
#     "415": "Cheating",
#     "416": "Cheating by personation",
#     "417": "Punishment for cheating",
#     "418": "Cheating with knowledge that wrongful loss may ensue to person whose interest offender is bound to protect",
#     "419": "Punishment for cheating by personation",
#     "420": "Cheating and dishonestly inducing delivery of property",
#     "421": "Dishonest or fraudulent removal or concealment of property to prevent distribution among creditors",
#     "422": "Dishonestly or fraudulently preventing debt being available for creditors",
#     "423": "Dishonest or fraudulent execution of deed of transfer containing false statement of consideration",
#     "424": "Dishonest or fraudulent removal or concealment of property",
#     "425": "Mischief",
#     "426": "Punishment for mischief",
#     "427": "Mischief causing damage to the amount of fifty rupees",
#     "428": "Mischief by killing or maiming animal of the value of ten rupees",
#     "429": "Mischief by killing or maiming cattle, etc., of any value or any animal of the value of fifty rupees",
#     "430": "Mischief by injury to works of irrigation or by wrongfully diverting water",
#     "431": "Mischief by injury to public road, bridge, river or channel",
#     "432": "Mischief by causing inundation or obstruction to public drainage attended with damage",
#     "433": "Mischief by destroying, moving or rendering less useful a light-house or sea-mark",
#     "434": "Mischief by destroying or moving, etc., a land-mark fixed by public authority",
#     "435": "Mischief by fire or explosive substance with intent to cause damage to amount of one hundred or (in case of agricultural produce) ten rupees",
#     "436": "Mischief by fire or explosive substance with intent to destroy house, etc",
#     "437": "Mischief with intent to destroy or make unsafe a decked vessel or one of twenty tons burden",
#     "438": "Punishment for the mischief described in section 437 committed by fire or explosive substance",
#     "439": "Punishment for intentionally running vessel aground or ashore with intent to commit theft, etc",
#     "440": "Mischief committed after preparation made for causing death or hurt",
#     "441": "Criminal trespass",
#     "442": "House-trespass",
#     "443": "Lurking house-trespass",
#     "444": "Lurking house-trespass by night",
#     "445": "House-breaking",
#     "446": "House-breaking by night",
#     "447": "Punishment for criminal trespass",
#     "448": "Punishment for house-trespass",
#     "449": "House-trespass in order to commit offence punishable with death",
#     "450": "House-trespass in order to commit offence punishable with imprisonment for life",
#     "451": "House-trespass in order to commit offence punishable with imprisonment",
#     "452": "House-trespass after preparation for hurt, assault or wrongful restraint",
#     "453": "Punishment for lurking house-trespass or house-breaking",
#     "454": "Lurking house-trespass or house-breaking in order to commit offence punishable with imprisonment",
#     "455": "Lurking house-trespass or house-breaking after preparation for hurt, assault or wrongful restraint",
#     "456": "Punishment for lurking house-trespass or house-breaking by night",
#     "457": "Lurking house-trespass or house-breaking by night in order to commit offence punishable with imprisonment",
#     "458": "Lurking house-trespass or house-breaking by night after preparation for hurt, assault or wrongful restraint",
#     "459": "Grievous hurt caused whilst committing lurking house-trespass or house-breaking",
#     "460": "All persons jointly concerned in lurking house-trespass or house-breaking by night punishable where death or grievous hurt caused by one of them",
#     "461": "Dishonestly breaking open receptacle containing property",
#     "462": "Punishment for same offence when committed by person entrusted with custody",
#     "463": "Forgery",
#     "464": "Making a false document",
#     "465": "Punishment for forgery",
#     "466": "Forgery of record of Court or of public register, etc",
#     "467": "Forgery of valuable security, will, etc",
#     "468": "Forgery for purpose of cheating",
#     "469": "Forgery for purpose of harming reputation",
#     "470": "Forged document",
#     "471": "Using as genuine a forged document",
#     "472": "Making or possessing counterfeit seal, etc., with intent to commit forgery punishable under section 467",
#     "473": "Making or possessing counterfeit seal, etc., with intent to commit forgery punishable otherwise",
#     "474": "Having possession of document described in section 466 or 467, knowing it to be forged and intending to use it as genuine",
#     "475": "Counterfeiting device or mark used for authenticating documents described in section 467, or possessing counterfeit marked material",
#     "476": "Counterfeiting device or mark used for authenticating documents other than those described in section 467, or possessing counterfeit marked material",
#     "477": "Fraudulent cancellation, destruction, etc., of will, authority to adopt, or valuable security",
#     "477A": "Falsification of accounts",
#     "478": "Trade marks",
#     "479": "Property mark",
#     "480": "Using a false trade mark",
#     "481": "Using a false property mark",
#     "482": "Punishment for using a false property mark",
#     "483": "Counterfeiting a property mark used by another",
#     "484": "Counterfeiting a mark used by a public servant",
#     "485": "Making or possession of any instrument for counterfeiting a property mark",
#     "486": "Selling goods marked with a counterfeit property mark",
#     "487": "Making a false mark upon any receptacle containing goods",
#     "488": "Punishment for making use of any such false mark",
#     "489": "Tampering with property mark with intent to cause injury",
#     "489A": "Counterfeiting currency-notes or bank-notes",
#     "489B": "Using as genuine, forged or counterfeit currency-notes or bank-notes",
#     "489C": "Possession of forged or counterfeit currency-notes or bank-notes",
#     "489E": "Making or using documents resembling currency-notes or bank-notes",
#     "490": "Breach of contract of service during voyage or journey",
#     "491": "Breach of contract to attend on and supply wants of helpless person",
#     "492": "Breach of contract to serve at distant place to which servant is conveyed at master’s expense",
#     "493": "Cohabitation caused by a man deceitfully inducing a belief of lawful marriage",
#     "494": "Marrying again during lifetime of husband or wife",
#     "489D": "Making or possessing instruments or materials for forgoing or counterfeiting currency-notes or bank-notes",
#     "495": "Same offence with concealment of former marriage from person with whom subsequent marriage is contracted",
#     "496": "Marriage ceremony fraudulently gone through without lawful marriage",
#     "497": "Adultery",
#     "498": "Enticing or taking away or detaining with criminal intent a married woman",
#     "499": "Defamation",
#     "500": "Punishment for defamation",
#     "501": "Printing or engraving matter known to be defamatory",
#     "502": "Sale of printed or engraved substance containing defamatory matter",
#     "503": "Criminal intimidation",
#     "504": "Intentional insult with intent to provoke breach of the peace",
#     "505": "Statements conducing to public mischief",
#     "506": "Punishment for criminal intimidation",
#     "507": "Criminal intimidation by an anonymous communication",
#     "509": "Word, gesture or act intended to insult the modesty of a woman",
#     "510": "Misconduct in public by a drunken person",
#     "508": "Act caused by inducing person to believe that he will be rendered an object of the Divine displeasure",
#     "511": "Punishment for attempting to commit offences punishable with imprisonment for life or other imprisonment"

# }
    

    

# def extract_articles_sections(text):
#     # Comprehensive pattern for articles
#     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
#     # Comprehensive pattern for sections (including plural 'Sections')
#     section_pattern = re.compile(r'\b(?:Section|Sec\.?|Sections|Secs\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
#     # Pattern for clauses
#     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
#     # Pattern for sub-sections
#     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

#     unique_references = set()  # Using a set to remove duplicates

#     # Function to process matches
#     def process_matches(pattern, prefix):
#         for match in pattern.finditer(text):
#             reference = match.group().strip()
#             if prefix not in reference.lower():
#                 reference = f"{prefix} : {reference}"
#             unique_references.add(reference)
    
#     # Processing all patterns
#     process_matches(article_pattern, "Article")
#     process_matches(section_pattern, "Section")
#     process_matches(clause_pattern, "Clause")
#     process_matches(subsection_pattern, "Sub-section")

#     # Adding article and section details if available
#     detailed_references = []
#     for reference in unique_references:
#         ref_parts = reference.split(":", 1)
#         if len(ref_parts) == 2:
#             ref_type, ref_number = ref_parts
#             ref_type = ref_type.strip()
#             ref_number = ref_number.strip()
#             # Extract just the numeric part for dictionary lookup
#             numeric_part = re.search(r'\d+', ref_number)
#             if numeric_part:
#                 numeric_part = numeric_part.group()
#                 if ref_type.lower() == "article":
#                     detail = article_details.get(numeric_part, "")
#                 elif ref_type.lower() == "section":
#                     detail = section_details.get(numeric_part, "")
#                 else:
#                     detail = ""
                
#                 if detail:
#                     detailed_references.append(f"{reference} - {detail}")
#                 else:
#                     detailed_references.append(reference)
#             else:
#                 detailed_references.append(reference)
#         else:
#             detailed_references.append(reference)

#     if detailed_references:
#         return "\n".join(sorted(detailed_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
#     else:
#         return "No articles, sections, clauses, or sub-sections found."

# # article_details = {
# #     "226": "Power of High Courts to issue certain writs.",
# #     "64" : "The Vice-President to be ex officio Chairman of the Council of States."
# #     # Add more articles and their details as needed
# # }

# # def extract_articles_sections(text):
# #     # Comprehensive pattern for articles
# #     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
# #     # Comprehensive pattern for sections (including plural 'Sections')
# #     section_pattern = re.compile(r'\b(?:Section|Sec\.?|Sections|Secs\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
# #     # Pattern for clauses
# #     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
# #     # Pattern for sub-sections
# #     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

# #     unique_references = set()  # Using a set to remove duplicates

# #     # Function to process matches
# #     def process_matches(pattern, prefix):
# #         for match in pattern.finditer(text):
# #             reference = match.group().strip()
# #             if prefix not in reference.lower():
# #                 reference = f"{prefix} : {reference}"
# #             unique_references.add(reference)

    
# #     # Processing all patterns
# #     process_matches(article_pattern, "Article :")
# #     process_matches(section_pattern, "Section :")
# #     process_matches(clause_pattern, "Clause :")
# #     process_matches(subsection_pattern, "Sub-section :")


# #     # Adding article details if available
# #     detailed_references = []
# #     for reference in unique_references:
# #         ref_parts = reference.split(":", 1)
# #         if len(ref_parts) == 2:
# #             ref_type, ref_number = ref_parts
# #             ref_number = ref_number.strip()
# #             # Extract just the numeric part for dictionary lookup
# #             numeric_part = re.search(r'\d+', ref_number)
# #             if numeric_part:
# #                 numeric_part = numeric_part.group()
# #                 detail = article_details.get(numeric_part, "")
# #                 if detail:
# #                     detailed_references.append(f"{reference} - {detail}")
# #                 else:
# #                     detailed_references.append(reference)
# #             else:
# #                 detailed_references.append(reference)
# #         else:
# #             detailed_references.append(reference)

# #     if detailed_references:
# #         return "\n".join(sorted(detailed_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
# #     else:
# #         return "No articles, sections, clauses, or sub-sections found."



# # def extract_articles_sections(text):
# #     # Comprehensive pattern for articles
# #     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
# #     # Comprehensive pattern for sections (including plural 'Sections')
# #     section_pattern = re.compile(r'\b(?:Section|Sec\.?|Sections|Secs\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
# #     # Pattern for clauses
# #     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
# #     # Pattern for sub-sections
# #     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

# #     unique_references = set()  # Using a set to remove duplicates

# #     # Function to process matches
# #     def process_matches(pattern, prefix):
# #         for match in pattern.finditer(text):
# #             reference = match.group().strip()
# #             if prefix not in reference.lower():
# #                 reference = f"{prefix} {reference}"
# #             unique_references.add(reference)

#     # # Processing all patterns
#     # process_matches(article_pattern, "Article :")
#     # process_matches(section_pattern, "Section :")
#     # process_matches(clause_pattern, "Clause :")
#     # process_matches(subsection_pattern, "Sub-section :")

# #     if unique_references:
# #         return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
# #     else:
# #         return "No articles, sections, clauses, or sub-sections found."
    
# def sanitize_text(text):
#     # Remove unwanted symbols using regular expressions
#     sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)
#     return sanitized_text

# # Updated `resolve_coreferences` function
# def resolve_coreferences(text):
#     doc = nlp(text)
#     resolved_text = []
#     for token in doc:
#         if token.dep_ == 'pronoun':
#             antecedent = token.head.text
#             resolved_text.append(antecedent)
#         else:
#             resolved_text.append(token.text)
    
#     return ' '.join(resolved_text)

# # Function to preprocess text with coreference resolution
# def preprocess_text_with_coref_resolution(text):
#     text = resolve_coreferences(text)
#     text = preprocess_text(text)
#     return text

# def print_all_information(text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     print("Case Number:", extract_case_number(text))
#     print("Governing Law:", extract_governing_law(text))
#     print("Final Verdict:", extract_final_verdict(text))
#     print("Parties:", extract_parties(text))
#     print("Date:", extract_date(text))
#     print("Title of the Case:", extract_case_title(text))
#     print("Name of the Court:", extract_court_name(text))
#     print("Articles:", extract_articles_sections(text))
#     print("Summary of the Case:", generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors))

# # Main interaction loop
# try:
#     print("Bot: Here is the summary of the uploaded legal document.")
#     print_all_information(raw_text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model)
# except KeyboardInterrupt:
#     print("\nBot: Thanks for talking, Bye!")













# import streamlit as st
# import re
# import nltk
# import contractions
# import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from autocorrect import Speller
# import emoji
# import regex 
# import gensim.downloader as api
# import fitz  # PyMuPDF
# from nltk import pos_tag
# import warnings
# import numpy as np

# warnings.simplefilter(action='ignore', category=UserWarning)

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Load the English language model
# nlp = spacy.load('en_core_web_sm')

# # Set max_length to a value that accommodates your text length
# nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

# # Load pre-trained Word2Vec model
# w2v_model = api.load('word2vec-google-news-300')

# # Initialize spell checker
# spell = Speller()

# # Define stopword2Vec
# stop_words = set(stopwords.words('english'))
    
# # Preprocessing function with lemmatization, spell checking, and NER tagging
# def preprocess_text(text):
    
#     # Correct spelling errors
#     text = spell(text)
    
#     # Remove HTML tags, URLs, and special characters
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Expand contractions
#     text = contractions.fix(text)
    
#     # Remove citations
#     text = re.sub(r'\[[0-9]+\]', '', text)
    
#     # Remove extra whitespaces
#     text = re.sub(r'\s+', ' ', text)
    
#     # Tokenization and NER tagging using spaCy
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if token.ent_type_:
#             tokens.append(token.ent_type_)
#         else:
#             tokens.append(token.text)
    
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Remove repeated characters
#     tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]
    
#     # Remove single characters and numeric tokens
#     tokens = [word for word in tokens if len(word) > 1 and not word.isdigit()]
    
#     # Handle emojis
#     text = emoji.demojize(text)
#     text = text.replace(":", "")
    
#     # Handle emoticons
#     emoticons = regex.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = regex.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
#     text = text + ' '.join(emoticons)
    
#     # Join tokens back into a string
#     preprocessed_text = ' '.join(tokens)
    
#     return preprocessed_text

# def calculate_similarity(sentence1, sentence2):
#     # Tokenize the sentences
#     tokens1 = word_tokenize(sentence1)
#     tokens2 = word_tokenize(sentence2)
    
#     # Filter out stopwords
#     tokens1 = [word for word in tokens1 if word.lower() not in stop_words]
#     tokens2 = [word for word in tokens2 if word.lower() not in stop_words]
    
#     # Get the Word2Vec vectors for each word
#     vectors1 = [w2v_model[word] for word in tokens1 if word in w2v_model]
#     vectors2 = [w2v_model[word] for word in tokens2 if word in w2v_model]

#     # Calculate the average vectors for each sentence
#     if vectors1 and vectors2:
#         avg_vector1 = np.mean(vectors1, axis=0)
#         avg_vector2 = np.mean(vectors2, axis=0)
        
#         # Calculate the cosine similarity between the average vectors
#         similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
#         return similarity
#     else:
#         return 0.0  # Return 0 if no vectors are found or all words are OOV

# # Keep track of previous questions and responses
# previous_questions = []
# previous_responses = []

# def generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, top_k=10, similarity_threshold=0.5):
#     global previous_questions, previous_responses

#     bot_response = ''
    
#     # Preprocess user input
#     processed_input = preprocess_text(user_input)
    
#     # Check if the processed input is empty or contains only stopwords
#     if not processed_input or all(word in stop_words for word in processed_input.split()):
#         return "I am sorry, I don't understand."
    
#     # Check if the current question is the same as a previous one
#     if processed_input in previous_questions:
#         index = previous_questions.index(processed_input)
#         return previous_responses[index]
    
#     # If not, continue with Word2Vec processing
#     similarities = []
#     for sent in sent_tokens:
#         similarity = calculate_similarity(processed_input, sent)
#         similarities.append(similarity)
    
#     # Convert similarities to a NumPy array for easier processing
#     similarities = np.array(similarities)
    
#     # Sort the similarities in descending order
#     sorted_indices = np.argsort(similarities)[::-1]
    
#     # Find the top k most similar sentences that are not in previous responses
#     top_k_sentences = []
#     for index in sorted_indices:
#         if len(top_k_sentences) < top_k and similarities[index] >= similarity_threshold:
#             top_k_sentences.append(sent_tokens[index])
    
#     # Assign the top k sentences to bot_response
#     if top_k_sentences:
#         bot_response = '\n'.join(top_k_sentences)
#         # Filter out URLs and unwanted tags from the response
#         bot_response = re.sub(r'http\S+', '', bot_response)  # Remove URLs
        
#         # Track previous questions and responses
#         previous_questions.append(processed_input)
#         previous_responses.append(bot_response)
#     else:
#         bot_response = "I am sorry, I don't get enough details."
    
#     return bot_response  # Return the top k sentences separated by newline characters



# # Generate response function
# def generate_response(user_input, text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
#     # Check if the user is asking for specific information
#     if 'case number' in user_input or 'case no' in user_input:
#         return extract_case_number(text)
#     elif 'governing law' in user_input:
#         return extract_governing_law(text)
#     elif 'final verdict' in user_input:
#         return extract_final_verdict(text)
#     elif 'party' in user_input:
#         return extract_parties(text)
#     elif 'date' in user_input:
#         return extract_date(text)
#     elif 'title of the case' in user_input or 'case title' in user_input:
#         return extract_case_title(text)
#     elif 'summary of the case' in user_input or 'case summary' in user_input:
#         return generate_response_word2vec(user_input, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
#     elif 'name of the court' in user_input or 'court name' in user_input:
#         return extract_court_name(text)
#     elif 'article' in user_input:
#         return extract_articles_sections(text)
#     else:
#         # Handle other types of questions
#         return "I'm sorry, I don't know."
    
# def extract_case_number(text):
#     # Regular expression pattern for matching case numbers
#     pattern = r'\b(?:[A-Z]{1,2}\s*\(\s*[A-Za-z]*\s*\)\s*)?\d{1,}\s*(?:of|OF)\s*\d{4}\b'
    
#     # Find all matches of the pattern in the text
#     case_numbers = re.findall(pattern, text)
    
#     # Return a list of unique case numbers
#     return list(set(case_numbers))

# def extract_governing_law(text):
#     # Define keywords for criminal law and civil law
#     criminal_law_keywords = [
#         'criminal', 'crime', 'offense', 'prosecution', 'defendant', 'felony', 'misdemeanor'
#     ]
#     civil_law_keywords = [
#         'civil', 'plaintiff', 'defendant', 'contract', 'tort', 'property', 'family', 'probate'
#     ]
    
#     # Tokenize the text and tag the parts of speech
#     tokens = word_tokenize(text.lower())
#     tagged_tokens = pos_tag(tokens)
    
#     # Extract nouns and adjectives from the tagged tokens
#     nouns_adjectives = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]
    
#     # Check for criminal law keywords
#     for keyword in criminal_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Criminal Law"
    
#     # Check for civil law keywords
#     for keyword in civil_law_keywords:
#         if keyword in nouns_adjectives:
#             return "Civil Law"
    
#     return "Governing law not identified"

# def extract_final_verdict(text):
#     # Define regular expressions to match common patterns for final verdicts and dates
#     verdict_patterns = [
#         r'final\s*verdict[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{2}/\d{2}/\d{4})(?:\.|\n)',
#         r'final\s*verdict[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'judgment[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'conclusion[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'decision[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'order[:\s]*([^\n.]*\d{4}-\d{2}-\d{2})(?:\.|\n)',
#         r'DATED[:\s]*\d{2}\.\d{2}\.\d{4}',  # Pattern for DATED: dd.mm.yyyy
#     ]
    
#     # Search for the patterns in the text
#     final_verdict = None
#     for pattern in verdict_patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             final_verdict = match.group(0).strip()
#             break
    
#     return final_verdict

# def extract_parties(text):
#     # New patterns for petitioner and respondents
#     new_petitioner_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner', re.IGNORECASE | re.DOTALL)
#     new_respondents_pattern = re.compile(r'Vs\.\s*(.*)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
#     # New patterns provided
#     new_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-vs-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
#     new_pattern_2 = re.compile(r'([^\n\r]+?)\s*\.\.\s*Petitioner\s*-versus-\s*([^\n\r]+?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
#     new_pattern_3 = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*\n*Vs\.\n*([^\n\r]+?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
#     new_pattern_4 = re.compile(r'([^\n\r]+?)\s*\.{4}PETITIONER\s*V/S\s*([^\n\r]+?)\s*\.{4}RESPONDENT', re.IGNORECASE | re.DOTALL)
    
#     # Additional new pattern to be added
#     additional_new_pattern = re.compile(r'([^\n\r]+)\s*\.\.\s*Petitioner\s*Vs\s*(.*?)\s*\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
    
#     # New pattern to add (provided by user)
#     new_pattern_5 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER\s*(?:\n*AND\n*|\s+AND\s+)([^\n\r]+?)\s*\.{3,}\s*RESPONDENT',
#         re.IGNORECASE | re.DOTALL
#     )
#     # Existing patterns
#     petitioner_pattern = re.compile(r'PETITIONER:\s*(.*?)\s*(?=Vs\.|RESPONDENT:)', re.IGNORECASE | re.DOTALL)
#     respondent_pattern = re.compile(r'RESPONDENT:\s*(.*?)$', re.IGNORECASE | re.DOTALL | re.MULTILINE)
#     pattern_ellipsis = re.compile(r'([^\n\r]+?)\s*…\s*Petitioner\s*.*?Versus\s*([^\n\r]+?)\s*…\s*Respondents', re.IGNORECASE | re.DOTALL)
#     pattern_dots = re.compile(r'([^\n\r]+?)\s*\.{4}\s*Petitioner\s*\(s\)\s*.*?Versus\s*([^\n\r]+?)\s*\.{4}\s*Respondent\s*\(s\)', re.IGNORECASE | re.DOTALL)
#     pattern_provided = re.compile(r'([^\n\r]+?)\s*…\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*.*?RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
#     pattern_appellant = re.compile(r'([^\n\r]+?)\s*…\s*APPELLANT\(S\)\s*.*?Versus:\s*([^\n\r]+?)\s*…\s*RESPONDENT\(S\)', re.IGNORECASE | re.DOTALL)
#     pattern_specific_parties = re.compile(r'([^\n\r]+?)\s*Versus\s*([^\n\r]+)', re.IGNORECASE | re.DOTALL)

#     # Try matching the specific pattern first
#     specific_pattern = re.compile(r'([^\n\r]+?)\s*\.\.\.\s*Petitioner\s*-vs-\s*(.*?)\s*\.\.\.\s*Respondents', re.IGNORECASE | re.DOTALL)
#     match = specific_pattern.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents = "\n".join([f"{resp}" for resp in respondents])
#         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
     
#     # Try matching the new pattern 6
#     new_pattern_6 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED|/COMPLAINANT)?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/ACCUSED|/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )
#     match = new_pattern_6.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondent = match.group(2).strip()
#         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
#     # Try matching the new pattern next
#     match = new_pattern.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('and')
#         respondents = [resp.strip() for resp in respondents]
#         return {
#             'petitioner': petitioner,
#             'respondents': respondents
#         }
 
#     # Try matching the new pattern next
#     new_pattern_7 = re.compile(
#         r'([^\n\r]+?)\s*\.{3,}\s*PETITIONER(?:/ACCUSED\(S\))?'
#         r'\s*(?:\n*AND\n*|\s+AND\s+)'
#         r'([^\n\r]+?)\s*\.{3,}\s*RESPONDENT(?:/COMPLAINANT)?',
#         re.IGNORECASE | re.DOTALL
#     )
#     match = new_pattern_7.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondent = match.group(2).strip()
#         return f"Petitioner: {petitioner}\nRespondent: {respondent}"
    
#     # Try matching the new pattern 2
#     match = new_pattern_2.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('and')
#         respondents = [resp.strip() for resp in respondents]
#         return {
#             'petitioner': petitioner,
#             'respondents': respondents
#         }

#     # Try matching the new pattern 3
#     match = new_pattern_3.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('\n')
#         respondents = [resp.strip() for resp in respondents if resp.strip()]
#         respondents = "\n".join([f"{resp}" for resp in respondents])
#         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"
    
#     # Try matching the new pattern 4
#     match = new_pattern_4.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondent = match.group(2).strip()
#         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

#     # Try matching the additional new pattern
#     match = additional_new_pattern.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = [resp.strip() for resp in respondents_text.split('\n') if resp.strip()]
#         return {
#             'petitioner': petitioner,
#             'respondents': respondents
#         }
    
#     # Try matching the new pattern 5 (provided by user)
#     match = new_pattern_5.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondent = match.group(2).strip()
#         return f"Petitioner: {petitioner}\nRespondent: {respondent}"

#     # Try matching the new petitioner and respondents pattern
#     petitioner_match = new_petitioner_pattern.search(text)
#     respondents_match = new_respondents_pattern.search(text)

#     if petitioner_match and respondents_match:
#         petitioner = petitioner_match.group(1).strip()
#         respondents_text = respondents_match.group(1).strip()
#         respondents_list = respondents_text.split('\n')
#         respondents = [resp.strip() for resp in respondents_list if resp.strip()]
#         respondents = "\n".join([f"{resp}" for resp in respondents])
#         return f"Petitioner: {petitioner}\nRespondents:\n{respondents}"

#     # Try matching the standard pattern
#     petitioners = petitioner_pattern.findall(text)
#     respondents = respondent_pattern.findall(text)
    
#     if petitioners and respondents:
#         petitioners = [p.strip() for p in petitioners]
#         respondents = [r.strip() for r in respondents]
#         parties = []
#         for petitioner, respondent in zip(petitioners, respondents):
#             parties.append(f"Petitioner: {petitioner}\nRespondent: {respondent}")
#         return "\n\n".join(parties)
    
#     # Try matching the ellipses pattern
#     matches = pattern_ellipsis.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     # Try matching the dots pattern
#     matches = pattern_dots.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     # Try matching the provided specific format
#     matches = pattern_provided.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     # Try matching the appellant pattern
#     matches = pattern_appellant.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     # Try matching the specific parties pattern
#     matches = pattern_specific_parties.findall(text)
#     if matches:
#         parties = []
#         for petitioner, respondent in matches:
#             parties.append(f"Petitioner: {petitioner.strip()}\nRespondent: {respondent.strip()}")
#         return "\n\n".join(parties)
    
#     new_pattern_8 = re.compile(
#           r'([A-Z\s\-\/]+)\s+VS\s+([A-Z\s\-\/]+)', re.IGNORECASE
#     ) 
#     match = new_pattern_8.search(text)
#     if match:
#         petitioner = match.group(1).strip()
#         respondents_text = match.group(2).strip()
#         respondents = respondents_text.split('AND')
#         respondents = [resp.strip() for resp in respondents]
#         return {
#             'petitioner': petitioner,
#             'respondents': respondents
#         }
    
#     return "Parties not found."

# def extract_date(text):
#     # Define regex pattern to match dates in various formats
#     date_pattern = r'(?:\d{1,2}(?:st|nd|rd|th)?(?:\s)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{4})|(?:\d{1,2}(?:\/|-)\d{1,2}(?:\/|-)\d{2,4})'
    
#     # Find all matches of the date pattern in the text
#     matches = re.findall(date_pattern, text)
    
#     return matches

# def extract_case_title(text):
#     # Pattern to match case titles with varying formats, including ellipsis and flexible date formats
#     pattern = r'((?:(?:State Of|M/S)\s)?[A-Z][a-zA-Z\s.,&()\'\-/]+(?:\s+\.{3}\s+[A-Z][a-zA-Z\s.,&()\'\-/]+)?(?:\s+vs\.?\s+(?:(?:State Of|M/S)\s)?[A-Z][a-zA-Z\s.,&()\'\-/]+(?:\s+\.{3}\s+[A-Z][a-zA-Z\s.,&()\'\-/]+)?)?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})'
    
#     # Alternative pattern for cases without a date
#     pattern_alt = r'((?:(?:State Of|M/S)\s)?[A-Z][a-zA-Z\s.,&()\'\-/]+(?:\s+\.{3}\s+[A-Z][a-zA-Z\s.,&()\'\-/]+)?(?:\s+vs\.?\s+(?:(?:State Of|M/S)\s)?[A-Z][a-zA-Z\s.,&()\'\-/]+(?:\s+\.{3}\s+[A-Z][a-zA-Z\s.,&()\'\-/]+)?)?)'
    
#     # First try matching the pattern with date
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     if match:
#         title = match.group(1).strip()
#         date = match.group(2).strip()
#         return f"{title} on {date}"
    
#     # If no match, try the alternative pattern
#     match = re.search(pattern_alt, text, re.IGNORECASE)
    
#     if match:
#         return match.group(1).strip()
    
#     return "Title and date not found"

# def extract_court_name(text):
#     # Define a more comprehensive pattern for court names, including the new format
#     comprehensive_pattern = (
#         r'(?:Supreme Court|High Court of Judicature at \w+|High Court of Judicature for \w+|'
#         r'High Court of \w+|District Court|Bench of \w+ High Court|High Court at \w+)'
#     )
    
#     # Search for the comprehensive pattern in the text
#     match = re.search(comprehensive_pattern, text, re.IGNORECASE)
    
#     if match:
#         return match.group(0).strip()
#     else:
#         # Define a fallback pattern for court names
#         fallback_pattern = r'(?:Supreme|High|District) Court'
        
#         # Search for the fallback pattern in the text
#         match = re.search(fallback_pattern, text, re.IGNORECASE)
        
#         if match:
#             return match.group(0).strip()
#         else:
#             return "Court name not found"



# def extract_articles_sections(text):
#     # Comprehensive pattern for articles
#     article_pattern = re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Comprehensive pattern for sections
#     section_pattern = re.compile(r'\b(?:Section|Sec\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and|to)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for clauses
#     clause_pattern = re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?(?:\s*(?:,|and)\s*\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?)*)', re.IGNORECASE)
    
#     # Pattern for sub-sections
#     subsection_pattern = re.compile(r'\bsub-section\s*(\(\d+\)(?:\([a-z]\))?(?:\s*(?:,|and)\s*\(\d+\)(?:\([a-z]\))?)*)', re.IGNORECASE)

#     unique_references = set()  # Using a set to remove duplicates

#     # Function to process matches
#     def process_matches(pattern, prefix):
#         for match in pattern.finditer(text):
#             reference = match.group().strip()
#             if prefix not in reference.lower():
#                 reference = f"{prefix} {reference}"
#             unique_references.add(reference)

#     # Processing all patterns
#     process_matches(article_pattern, "Article :")
#     process_matches(section_pattern, "Section :")
#     process_matches(clause_pattern, "Clause :")
#     process_matches(subsection_pattern, "Sub-section :")

#     if unique_references:
#         return "\n".join(sorted(unique_references, key=lambda x: (x.split()[0].lower(), x.split()[1:])))
#     else:
#         return "No articles, sections, clauses, or sub-sections found."  
    
# def sanitize_text(text):
#     # Remove unwanted symbols using regular expressions
#     sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)
#     return sanitized_text

# # Updated `resolve_coreferences` function
# def resolve_coreferences(text):
#     doc = nlp(text)
#     resolved_text = []
#     for token in doc:
#         if token.dep_ == 'pronoun':
#             antecedent = token.head.text
#             resolved_text.append(antecedent)
#         else:
#             resolved_text.append(token.text)
    
#     return ' '.join(resolved_text)

# # Function to preprocess text with coreference resolution
# def preprocess_text_with_coref_resolution(text):
#     text = resolve_coreferences(text)
#     text = preprocess_text(text)
#     return text

# def read_pdf(file):
#     pdf_document = fitz.open(stream=file.read(), filetype="pdf")
#     raw_text = ''
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         raw_text += page.get_text()
#     pdf_document.close()
#     return raw_text

# def main():
#     st.title("Legal Case Summarization")
    
#     # File uploader
#     uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
#     if uploaded_file is not None:
#         # Read and process the PDF
#         raw_text = read_pdf(uploaded_file)
        
#         # Preprocess the text
#         sent_tokens = nltk.sent_tokenize(raw_text)
#         preprocessed_sent_tokens = [preprocess_text(sent) for sent in sent_tokens]
        
#         # Create TF-IDF vectors
#         word_vectorizer = TfidfVectorizer(
#             tokenizer=word_tokenize,
#             stop_words=stopwords.words('english'),
#             ngram_range=(1, 3),
#             max_features=15000,
#             token_pattern=r'\b\w+\b',
#             sublinear_tf=True,
#             smooth_idf=True,
#             norm='l2'
#         )
#         word_vectors = word_vectorizer.fit_transform(preprocessed_sent_tokens)
        
  
#         # Display information
#         st.subheader("Case Information")
#         st.write("Case Number:", extract_case_number(raw_text))
#         st.write("Governing Law:", extract_governing_law(raw_text))
#         st.write("Final Verdict:", extract_final_verdict(raw_text))
#         st.write("Parties:", extract_parties(raw_text))
#         st.write("Date:", extract_date(raw_text))
#         st.write("Title of the Case:", extract_case_title(raw_text))
#         st.write("Name of the Court:", extract_court_name(raw_text))
#         st.write("Articles:", extract_articles_sections(raw_text))

#         # Case summary
#         st.subheader("Case Summary")
#         summary = generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors)
#         st.write(summary)

# if __name__ == "__main__":
#     main()
    








import streamlit as st
import re
import nltk
import contractions
import spacy
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

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Set max_length to a value that accommodates your text length
nlp.max_length = 5000000  # Set max_length to a value that accommodates your text length

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
    
    # Tokenization and NER tagging using spaCy
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.ent_type_:
            tokens.append(token.ent_type_)
        else:
            tokens.append(token.text)
    
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

# Updated `resolve_coreferences` function
def resolve_coreferences(text):
    doc = nlp(text)
    resolved_text = []
    for token in doc:
        if token.dep_ == 'pronoun':
            antecedent = token.head.text
            resolved_text.append(antecedent)
        else:
            resolved_text.append(token.text)
    
    return ' '.join(resolved_text)

# Function to preprocess text with coreference resolution
def preprocess_text_with_coref_resolution(text):
    text = resolve_coreferences(text)
    text = preprocess_text(text)
    return text

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
    

















    
# # def print_all_information(text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model):
# #     print("Case Number:", extract_case_number(text))
# #     print("Governing Law:", extract_governing_law(text))
# #     print("Final Verdict:", extract_final_verdict(text))
# #     print("Parties:", extract_parties(text))
# #     print("Date:", extract_date(text))
# #     print("Title of the Case:", extract_case_title(text))
# #     print("Name of the Court:", extract_court_name(text))
# #     print("Articles:", extract_articles_sections(text))
# #     print("Summary of the Case:", generate_response_word2vec("summary of the case", sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors))

# # # Main interaction loop
# # try:
# #     print("Bot: I will provide information about the legal document.")
# #     #print_all_information(raw_text, sent_tokens, preprocessed_sent_tokens, word_vectorizer, word_vectors, w2v_model)
# # except KeyboardInterrupt:
# #     print("\nBot: Thanks for talking, Bye!")





