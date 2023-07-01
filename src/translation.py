from pathlib import Path
import translators as ts
from gtts import gTTS

class Translation:
    def __init__(self):
        self.input_lang = None
        self.input_lang_code = 'auto'
        self.output_lang = None
        self.output_lang_code = None
        #self.initial_summary = None
        self.output_summary = None
        self.lang_dict = {
            'English' : 'en',
            'Bahasa Melayu' : 'ms',
            'Chinese (Simplified)' : 'zh-CHS'
        }

    def convert_code(self):
        print(f'Input Language: {self.input_lang}')
        if self.input_lang in self.lang_dict:
            self.input_lang_code = self.lang_dict[self.input_lang]
            print(f"Language Code: {self.input_lang_code}\n")
        else:
            self.input_lang_code = 'auto'
            print('Language not found, will be auto-detected by translator.\n')
        print(f'Translate to: {self.output_lang}')
        if self.output_lang in self.lang_dict:
            self.output_lang_code = self.lang_dict[self.output_lang]
            print(f"Language Code: {self.output_lang_code}\n")
        else:
            self.output_lang_code = 'en'
            print(f"Language not found in the dictionary. Change to default English. \
                  \nLanguage Code: {self.output_lang_code}")
        return

    def translate_text(self, filepath=None, qa_str=None, output_lang=None, input_lang='auto', encoding='utf-8', audio = True):
        self.output_lang = output_lang
        self.input_lang = input_lang

        # Convert language codes
        self.convert_code()
        
        # For Summarised Text Translation
        if filepath != None:
            # Read .txt file
            input_summary = Path(filepath).read_text()
            print ('Input text: \n',input_summary)

            # Translate to preferred language
            self.output_summary = ts.translate_text(input_summary,from_language=self.input_lang_code,to_language=self.output_lang_code)
            print(self.output_summary)

            # Save the summary to a text file
            summary_file = 'summary_{}.txt'.format(self.output_lang_code)
            with open(f'summary_{self.output_lang_code}.txt', 'w', encoding=encoding) as f:
                f.write(self.output_summary)
            print(f'File saved to {summary_file}')

            if audio == True:
                self.convert_audio()
            else:
                print('No audio file as per request.')
            return

        # For QA Bot Input Translation (Work-in-Progress)
        if qa_str != None:
            # Translate to English for LLM to process
            output_qa = ts.translate_text(input_summary,from_language=self.input_lang_code,to_language='en')
            print(output_qa)
            return output_qa
    
    def convert_audio(self, filepath=None):
        # Different coding for Chinese Simplified for audio. No change for English/Malay.
        if self.output_lang_code == 'zh-CHS':
            lang = 'zh-CN'
        else:
            lang = self.output_lang_code

        # No filepath input into function, convert text from translate_text function to audio
        if filepath == None:
            # Passing the text and language to the engine,
            # here we have marked slow=False. Which tells
            # the module that the converted audio should
            # have a high speed
            myobj = gTTS(text=self.output_summary, lang=lang, slow=False)

            # Saving the converted audio in a mp3 file named
            # output
            myobj.save("output_{}.mp3".format(lang))
            print('Audio reading saved to output_{}.mp3'.format(lang))
        
        # If filepath provided
        else:
            # Read text file
            input_text = Path(filepath).read_text()

            # Convert to audio
            myobj = gTTS(text=input_text, lang=lang, slow=False)

            # Save audio in mp3 file
            myobj.save("output_{}.mp3".format(lang))
