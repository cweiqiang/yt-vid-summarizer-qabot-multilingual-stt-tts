from pathlib import Path

import translators as ts
from gtts import gTTS


class Translation:
    def __init__(self):
        """
        Initialising variables and dictionaries to be used throughout
        """
        self.input_lang = None
        self.input_lang_code = 'auto'
        self.output_lang = None
        self.output_lang_code = None
        self.output_summary = None
        self.lang_dict = {
            'English' : 'en',
            'Bahasa Melayu' : 'ms',
            'Chinese (Simplified)' : 'zh-CHS'
        }

    def convert_code(self):
        """
        Takes in the desired input and output languages, and converts them to the
        language code used by the translator API.
        """
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

    def translate_text(self, 
                       filepath=None, 
                       qa_str=None, 
                       output_lang=None, 
                       input_lang='auto', 
                       encoding='utf-8', 
                       audio = True,
                       qa_response=False):
        """
        If filepath is given, translates text document to the desired language 
        and writes the translated document to a text file. Audio reading of
        the translated document is also generated unless opted out.

        If qa_str is given, where qa_response is False, translates text string 
        from QA bot to English for LLM to process. If qa_response is True, 
        translates LLM's response from English to user's language, along with
        audio clip generated

        Args:
            filepath (str, optional): filepath to text document that is 
                pending translation. Defaults to None.
            qa_str (str, optional): text string pending translation to 
                English. Defaults to None.
            output_lang (str, optional): Language that user desires to translate 
                input to. Defaults to None.`
            input_lang (str, optional): Language of input text. Defaults to 'auto'.
            encoding (str, optional): Type of encoding for translated text file. 
                Defaults to 'utf-8'.
            audio (bool, optional): Choice of whether to generate audio file for
                translated. Defaults to True.
            qa_response (bool, optional): Whether input is response from QAbot 
                pending to translate for user. Defaults to False.

        Returns:
            _type_: _description_
        """
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
            try:
                translator = 'google'
                self.output_summary = ts.translate_text(
                    input_summary,
                    from_language=self.input_lang_code,
                    to_language=self.output_lang_code,
                    translator=translator
                    )
            except:
                try:
                    translator = 'yandex'
                    self.output_summary = ts.translate_text(
                        input_summary,
                        from_language=self.input_lang_code,
                        to_language=self.output_lang_code,
                        translator=translator
                        )
                except:
                    translator = 'bing'
                    self.output_summary = ts.translate_text(
                        input_summary,
                        from_language=self.input_lang_code,
                        to_language=self.output_lang_code,
                        translator=translator
                        )
            print(self.output_summary)
            print('Translated by: ',translator)

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

        # For QA Bot Translation (Work-in-Progress)
        if qa_str != None:
            if qa_response == False:
                # Translate to English for LLM to process
                input_qa = ts.translate_text(
                    qa_str,
                    from_language=self.input_lang_code,
                    to_language='en'
                    )
                print(input_qa)
                return input_qa
            else:
                # Translate response from LLM (English) to output language to respond to user
                output_qa = ts.translate_text(
                    qa_str,
                    from_language='en',
                    to_language=self.output_lang_code
                    )
                print(output_qa)
                
                if audio == True:
                    # Convert response to audio
                    self.convert_audio(qa=output_qa,audio_lang=self.output_lang_code)
                return output_qa
    
    def convert_audio(self, filepath=None, audio_lang = None, qa = None):
        """
        Converts translated text to audio file.

        Args:
            filepath (str, optional): filepath to translated text. Defaults to None.
            audio_lang (str, optional): specify audio file output language. Defaults to None.
        """

        # No filepath input into function, convert text from translate_text function to audio
        if filepath == None:
            # Different coding for Chinese Simplified for audio. No change for English/Malay.
            if self.output_lang_code == 'zh-CHS':
                lang = 'zh-CN'
            else:
                lang = self.output_lang_code

            if qa == None:
                # Convert summarised text to audio
                text = self.output_summary
            else:
                # Convert string to audio for QA Bot response
                text = qa

            # Passing the text and language to the engine,
            # here we have marked slow=False. Which tells
            # the module that the converted audio should
            # have a high speed
            myobj = gTTS(text=text, lang=lang, slow=False)

            # Saving the converted audio in a mp3 file named output
            myobj.save("output_{}.mp3".format(lang))
            print('Audio reading saved to output_{}.mp3'.format(lang))
        
        # If filepath provided, direct conversion from text file.
        # Output language required
        else:
            # Convert audio file language to language code
            if audio_lang != None:
                self.output_lang = audio_lang
            self.convert_code()
            if self.output_lang_code == 'zh-CHS':
                lang = 'zh-CN'
            else:
                lang = self.output_lang_code

            # Read text file
            input_text = Path(filepath).read_text()

            # Convert to audio
            myobj = gTTS(text=input_text, lang=lang, slow=False)

            # Save audio in mp3 file
            myobj.save("output_{}.mp3".format(lang))
            print('Audio reading saved to output_{}.mp3'.format(lang))
