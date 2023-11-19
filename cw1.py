from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import warnings
import numpy as np
import nltk
import string
import random


#Initalising the lists containing the welcoming messages for the user.

userinputsalute = ('suuuup' ,'hello' ,'hola' ,'hi' ,'yooo' ,'hello chatty' ,'hi there')
useroutputsalute = ('oooh hey' ,'yoooo' ,'hello' ,'*blinks*' ,'HEEEEEEEEEEY','hi there')

#this function checks if the input of the user is in the words of the lists userinputsalute , it returns one random word from the useroutputsalute.

def salute(his):
	for word in his.split():
		if word.lower() in userinputsalute:
			return random.choice(useroutputsalute)

#The list below , it is seperated into question:answer form.

quicktalking = {
'u good?':'peeeeerfect thanks',
'nice weather':'you joking right?',
'thank u':'nothing my beloved user',
'goodnight':'sweet dreams',
'do you consider yourself as the main character?':'u already know the answer',
'do you like chocoalates?':'cadbury is the best',
'do you play any sports?':'just football',
'what are your main skills?':'programming of course',
'exit':'If you want to leave just type Enough',
'thanks':'nop',
'u feeling good?':'hmm , not a lot to be honest',
'i am here for you chatty':'thats cute!!!'
}

#We take each value of the lists above and we transform it to string so it is easier for us to work.

smallintera = quicktalking.values()
smallintera = [str(i) for i in smallintera]

#In this function we process the quicktalking list , what we do is that we initalise one list and then we apply the function TfidfVectorizer. For each our st and request we use the fit_transform to our input data at a single time and transform to store into our variable an autonomous transformation dataframe with our transformed values.
#later we apply the cosine similarity to our Two variables tfst and tfrequest respectively and we use flatten function to return just only one dimension of the array.
#We then sort our findings using the slice that fitted well in my approach(i used different fittings and i concluded to this one as it worked better)
#We then check the similarity and if it is above 0.5 we know that most probably is what we want , we store the result of the list in a variable and we always take the first stored variable which is the most accurate. 

def quicktalkingtf(st,request):
	request=[request]
	TFIDF = TfidfVectorizer(sublinear_tf=True,use_idf=True)
	
	tfst = TFIDF.fit_transform(st)
	tfrequest = TFIDF.transform(request)
	
	cosinetf = cosine_similarity(tfst,tfrequest).flatten()
	relationtf = cosinetf.argsort()[:-2:-1]
	
	if(cosinetf[relationtf]>0.5):
		result = [smallintera[j] for j in relationtf[:1]]
		return result[0]


#In the below function nothing extreme happens , i just take the specific input from the user , and then i take out all the other words of the sentence except from the name of the user , i then take it and return it where i need it.

def pronouncing(username):
	p = username.split()

	if('You can call me' in username):
		for i in p:
			if(i!='You' and i!='can' and i!='call'and i!='me'):
				return i 
	
	elif('My name is' in username):
		for i in p:
			if(i!='My' and i!='name' and i!='is'):
				return i 
	
	elif('My people call me' in username):
		for i in p:
			if(i!='My' and i!='people' and i!='call'and i!='me'):
				return i 
	
	elif('is' in username):
		for i in p:
			if(i!='is'):
				return i 
	
	elif('Please now call me' in username):
		for i in p:
			if(i!='Please' and i!='now' and i!='call'and i!='me'):
				return i 
	
	elif('Change my name to' in username):
		for i in p:
			if(i!='Change' and i!='my' and i!='name'and i!='to'):
				return i 

	else:
		return username


#As we learnt in the labs , i download some packages and then i open the question-answer file , i make everything in lowercase letters to be able to use it easier and then i make sentences from the tokens that i create using my corpus.

nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)

f=open('QA.txt','r',errors='ignore')

corpus = f.read()

corpus = corpus.lower()

sentence_tokens = nltk.sent_tokenize(corpus)



#In this question and answer function what i actually do is simple. Firstly i append my variable containing the tokenized sentences with the query of the user.
#I also initialise one variable called chatty which is the name of the bot where the answer and the possible response will be stored.
#With the same approach as the quicktalk we use the TfidfVectorizer to remove stopwords that may cause us problems with commonly used words.
#We use the same fit_transform function as before with the same intentions using the sentence_tokens variables with the appended query of the user.
#We then do the cosine similarity with the sentence we just observed from the user ,  using the last element(thats why [-1]) and the tftr which we just fitted our sentence_tokens.
#Flatten to restrict our findings to one array, and then we clearly sorting our flatten array.
#We observe the sentence possibility in the position -2 which is what we want and we check that we have this answer , or something similar to our corpus.
#If this is the case we append to our empty chatty variable the index in the position where we do the argsorting of our cosine variable where we found the similarity between the sentence we wanted. I used the position below as it fitted to the way my corpus is placed ( i observed this after trying various positions).
#If our chatty doesnt know the answer , it returns the mesage below to the user.

def questionanswering(query):
	sentence_tokens.append(query)
	chatty=' '

	Vectorizer = TfidfVectorizer(stop_words='english')
	tftr = Vectorizer.fit_transform(sentence_tokens)
	cosine = cosine_similarity(tftr[-1],tftr)
	flatten = cosine.flatten()
	flatten.sort()
	answer = flatten[-2]

	if(answer!=0):
		chatty = chatty + sentence_tokens[cosine.argsort()[0][-2]]
		return chatty
	else:
		chatty = chatty+" I can not undestand you i am afraid :( "
		return chatty


#Here the intent matching happens , we observe what the user wants and we desplay it on the screen.
#We store the name of the user , using the i variable we change the display on the screen with the name of the user and ":" so that we know who is talking.
#The below code is obvious as we observe each case and we call the relevant functions using the query of the user(userpref).
#We change the name of the user when needed , when the user ask the name we output it, and by pressing enough the user can end the conversation.
#we then set the end to true so we can end the program.
#we use the strip function to concatinate any useless symbols that the user may accidentaly type so we can observe what the user wants to say.

end=False
while(end!=True):
	k = input('\nHi there, i am chatty the beloved bot. What is your name? ->')
	usernewname = ''
	username = pronouncing(k)
	stop=False
	i=0
	i=i+1
	while(stop!=True):
		if(i==1):
			userpref = input('\nChatty: Hello ' + (usernewname if len(usernewname)!=0 else username) + ' i am at your service!,if you want to leave :( please type enough. \n\n'+ (usernewname if len(usernewname)!=0 else username) +': ')
		else:
			userpref= input('\n' + (usernewname if len(usernewname)!=0 else username) + ': ')
		
		userpref = userpref.strip("!£$%^&*()<>,;?")

		if('You can call me' in userpref or 'My people call me' in userpref or 'My name is' in userpref or 'Please now call me' in userpref or 'Change my name to' in userpref):
			i=i+1	
			usernewname = pronouncing(userpref)
			print('\nChatty: Now i can call you ' + usernewname)
		elif(userpref=='enough'):
			stop=True
			print('\nChatty: Goodbye ' + (username if len(usernewname)==0 else usernewname) + ' is been great talking to you :)')
		elif(userpref=='Can you tell me my name chatty' or userpref=='What is my name'):
			i=i+1
			if(len(usernewname)==0):
				print('\nChatty: Your name is ' + username)
			else:
				print('\nChatty: Your name is ' + usernewname)

		else:
			i=i+1
			if(quicktalkingtf(quicktalking,userpref)!=None):
				print('\nChatty: ' + quicktalkingtf(quicktalking,userpref))
			elif(salute(userpref)!=None):
				print('\nChatty: '+ salute(userpref))
				
			else:
				print('\nChatty:' + questionanswering(userpref))
				sentence_tokens.remove(userpref)

	end=True
				
					
			
