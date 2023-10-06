# # #Step 1: Uncomment below and generate tamil.model
# import sentencepiece as spm
# user_def_symbols = ["<pad>","<s>","</s>","<mask>","\n","+","-","*","$","%","^","1","2","3","4","5","6","7","8","9","0","#","@",
#                                 "&","=","~","<",">",":",";","'","(",")",".",","]
# english_alphabets = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
# english_capital_letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
# tamil_stop_words = [
# "ஒரு",
# "என்று",
# "மற்றும்",
# "இந்த",
# "இது",
# "என்ற",
# "கொண்டு",
# "என்பது",
# "பல",
# "ஆகும்",
# "அல்லது",
# "அவர்",
# "நான்",
# "உள்ள",
# "அந்த",
# "இவர்",
# "என",
# "முதல்",
# "என்ன",
# "இருந்து",
# "சில",
# "என்",
# "போன்ற",
# "வேண்டும்",
# "வந்து",
# "இதன்",
# "அது",
# "அவன்",
# "தான்",
# "பலரும்",
# "என்னும்",
# "மேலும்",
# "பின்னர்",
# "கொண்ட",
# "இருக்கும்",
# "தனது",
# "உள்ளது",
# "போது",
# "என்றும்",
# "அதன்",
# "தன்",
# "பிறகு",
# "அவர்கள்",
# "வரை",
# "அவள்",
# "நீ",
# "ஆகிய",
# "இருந்தது",
# "உள்ளன",
# "வந்த",
# "இருந்த",
# "மிகவும்",
# "இங்கு",
# "மீது",
# "ஓர்",
# "இவை",
# "இந்தக்",
# "பற்றி",
# "வரும்",
# "வேறு",
# "இரு",
# "இதில்",
# "போல்",
# "இப்போது",
# "அவரது",
# "மட்டும்",
# "இந்தப்",
# "எனும்",
# "மேல்",
# "பின்",
# "சேர்ந்த",
# "ஆகியோர்",
# "எனக்கு",
# "இன்னும்",
# "அந்தப்",
# "அன்று",
# "ஒரே",
# "மிக",
# "அங்கு",
# "பல்வேறு",
# "விட்டு",
# "பெரும்",
# "அதை",
# "பற்றிய",
# "உன்",
# "அதிக",
# "அந்தக்",
# "பேர்",
# "இதனால்",
# "அவை",
# "அதே",
# "ஏன்",
# "முறை",
# "யார்",
# "என்பதை",
# "எல்லாம்",
# "மட்டுமே",
# "இங்கே",
# "அங்கே",
# "இடம்",
# "இடத்தில்",
# "அதில்",
# "நாம்",
# "அதற்கு",
# "எனவே",
# "பிற",
# "சிறு",
# "மற்ற",
# "விட",
# "எந்த",
# "எனவும்",
# "எனப்படும்",
# "எனினும்",
# "அடுத்த",
# "இதனை",
# "இதை",
# "கொள்ள",
# "இந்தத்",
# "இதற்கு",
# "அதனால்",
# "தவிர",
# "போல",
# "வரையில்",
# "சற்று",
# "எனக்"
# ]

# relationship_words = [
# "தந்தை",
# "தாய்",
# "சகோதரர்",
# "சகோதரி",
# "தங்கை",
# "மகன்",
# "மகள்",
# "மாமா",
# "அத்தை",
# "மருமகள்",
# "மருமகன்",
# ]

# tamil_verbs = [
#     "வா",  # come
#     "போ",  # go
#     "செய்",  # do
#     "கேள்",  # ask
#     "பேசு",  # speak
#     "கவனி",  # listen
#     "அறி",  # know
#     "வாங்கு",  # buy
#     "விடு",  # leave
#     "உதவ",  # help
#     "படிக்க",  # read
#     "எழுத",  # write
#     "சாப்பிட",  # eat
#     "குடிக்க",  # drink
#     "வேலை",  # work
#     "கண்டுபிடி",  # find
#     "விண்ணப்ப",  # request
#     "கிடைக்க",  # get
#     "விளக்கு",  # explain
#     "அனுப்பு",  # send
#     "ஆரம்பி",  # start
#     "முடி",  # end
#     "விளையாடு",  # play
#     "சிரி",  # laugh
#     "அழு",  # cry
#     "புது",  # renew
#     "கல்",  # learn
#     "மற",  # forget
#     "தோண்ட",  # dig
#     "விற்க",  # sell
#     "தூங்க",  # sleep
#     "பிடி",  # catch
#     "விய",  # wonder
#     "நம்ப",  # believe
#     "வளர்",  # grow
#     "கட்டு",  # build
#     "தேட",  # search
#     "நகர",  # move
#     "வெளி",  # exit
#     "இழு",  # pull
#     "தள்ளு",  # push
#     "பற",  # fly
#     "நட",  # walk
#     "ஓட",  # run
#     "குழு",  # shake
#     "வேகம்",  # speed up
#     "ஏற",  # climb
#     "விழ",  # fall
#     "சேர்",  # join
#     "கட",  # cross
#     "காய",  # hurt
#     "நுகர",  # smell
#     "சுவை",  # taste
#     "உண்",  # eat
#     "சொல்",  # tell
#     "காட்ட",  # show
#     "செலுத்த",  # pay
#     "செயல்",  # act
#     "அதிர்",  # surprise
#     "பார்",  # look
#     "சிந்தி",  # think
#     "விரும்ப",  # like
#     "வெறு",  # hate
#     "அழை",  # call
#     "சரி",  # correct
#     "தவிர்",  # avoid
#     "உயர்த்த",  # rise
#     "ஆட",  # dance
#     "சாவு",  # die
#     "கொலை",  # kill
#     "அடி",  # hit
#     "வீச",  # throw
#     "சுத்தம்",  # clean
#     "காத்திரு",  # wait
#     "வலி",  # hurt
#     "பிடிக்க",  # like
#     "பயம்",  # fear
#     "கேட்க",  # request
#     "திரும்ப",  # return
#     "முயற்சி",  # try
#     "சோதி",  # test
#     "வெல்ல",  # win
#     "ஓட்ட",  # drive
#     "குதி",  # jump
#     "நீந்த",  # swim
#     "வை",  # place
# ]

# frequent_sub_words = [
#     "க்க",
#     "க்கு",
#     "கள்",
#     "கும்",
#     "யும்",
#     "வும்",
#     "வார்",
#     "வர்",
#     "தார்",
#     "ப்ப",
#     "ம்ப",
#     "ந்த",
#     "ண்ட",
#     "த்த",
#     "ன்ற",
#     "ச்ச",
#     "ன்ன",
#     "ஞ்ச",
#     "ற்ற",
#     "ட்ட",
#     "ங்க",
#     "ல்ல",
#     "ர்க",
#     "ள்ள",
#     "அவ",
#     "இவ",
#     "அத",
#     "இத",
#     "உல"
#     "அழ",
#     "அள",
#     "அற",
#     "அர",
#     "அன",
#     "அண",
#     "அட",
#     "மன",
#     "மண",
#     "மர",
#     "மல",
#     "மள",
#     "இன",
# ]

# popular_words = [
#     "தமிழ",
#     "காதல",
# ]

# user_def_symbols.extend(tamil_stop_words)
# user_def_symbols.extend(relationship_words)
# user_def_symbols.extend(tamil_verbs)
# user_def_symbols.extend(frequent_sub_words)
# user_def_symbols.extend(popular_words)
# user_def_symbols.extend(english_alphabets)
# user_def_symbols.extend(english_capital_letters)

# # # print(len(user_def_symbols))
# # # print(user_def_symbols)

# spm.SentencePieceTrainer.train(input="./data/processed_content.csv", model_prefix='tamil',vocab_size=4000,user_defined_symbols=user_def_symbols,model_type="BPE")

#Step 2: Generate fast tokenizer tokenizer.json
from transformers import AutoTokenizer
fast_tokenizer = AutoTokenizer.from_pretrained("./tamil_300m_clean", use_fast=True)
fast_tokenizer.save_pretrained("./tamil_300m_clean")
