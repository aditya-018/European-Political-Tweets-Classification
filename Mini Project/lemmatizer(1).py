import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import regex as re
from nltk.corpus import wordnet
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()

##Tags the words in the tweets
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return(wordnet.ADJ)
    elif nltk_tag.startswith('V'):
        return(wordnet.VERB)
    elif nltk_tag.startswith('N'):
        return(wordnet.NOUN)
    elif nltk_tag.startswith('R'):
        return(wordnet.ADV)
    else:          
        return(None)

#creating a function for cleraning tweet words 
def tweet_cleaner():
    stop_words = set(stopwords.words('english'))
    text = "".join([word for word in text.split() if word.lower() not in stop_words])
    #Remove words that are less than 3 chars
    text = "".join([word for word in text.split() if len(word) >= 3])
    
    #Removing Emojis
    text = re.sub(r'https\S+', '', text)
    
    emoji_pattern = re.compile("["
                               r"\U0001F600-\U0001F64F" 
                               r"\U0001F300-\U0001F5FF"  
                               r"\U0001F680-\U0001F6FF"  
                               r"\U0001F700-\U0001F77F"  
                               r"\U0001F780-\U0001F7FF"  
                               r"\U0001F800-\U0001F8FF"  
                               r"\U0001F900-\U0001F9FF"  
                               r"\U0001FA00-\U0001FA6F"  
                               r"\U0001FA70-\U0001FAFF"  
                               r"\U0001FB00-\U0001FBFF"  
                               r"\U0001FC00-\U0001FCFF"  
                               r"\U0001FD00-\U0001FDFF"  
                               r"\U0001F700-\U0001F77F"  
                               r"\U0001FE00-\U0001FEFF"  
                               r"\U0001FF00-\U0001FFFF"  
                               r"\U00020000-\U0002FFFF"  
                               r"\U00030000-\U0003FFFF"  
                               r"\U00040000-\U0004FFFF"  
                               r"\U000e000-\U000f8ff"    
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    #Remove Puntuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text
    

##Lemmatizes the words in tweets and returns the cleaned and lemmatized tweet
def lemmatize_tweet(tweet):
    #tokenize the tweet and find the POS tag for each token
    tweet = tweet_cleaner(tweet) #tweet_cleaner() will be the function you will write
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(tweet))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_tweet = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_tweet.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))
    return(" ".join(lemmatized_tweet))