import string
from anthropic import Client

class GlobalRelativeTokenizer:
    def __init__(self):
        """
        Initializes the tokenizer with default values and tokens.
        """
        self.context_size = 500
        self.token_to_value = {'<EOP>': 0}  # Assign 0 to the <EOP> token
        self.value_to_token = {0: '<EOP>'}  # Assign <EOP> to the value 0
        self.next_value = 1  # Start assigning values from 1

    def tokenize(self, text):
        """
        Tokenizes the given text into numerical values based on unique words.

        Parameters:
        text (str): The text to be tokenized.

        Returns:
        list: A list of numerical tokens representing the text.
        """
        tokens = text.split(" ")
        relative_tokens = []
        for token in tokens:
            if token not in self.token_to_value:
                self.token_to_value[token] = self.next_value
                self.value_to_token[self.next_value] = token
                self.next_value += 1
            relative_tokens.append(self.token_to_value[token])
        return relative_tokens

    def detokenize(self, relative_tokens):
        """
        Converts numerical tokens back into text.

        Parameters:
        relative_tokens (list): A list of numerical tokens.

        Returns:
        str: The detokenized text.
        """
        tokens = [self.value_to_token[value] for value in relative_tokens]
        return ' '.join(tokens)

def filter_text(text):
    """
    Filters and cleans the input text by converting to lowercase and removing punctuation.

    Parameters:
    text (str): The text to be filtered.

    Returns:
    str: The filtered text.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation.replace(":", '').replace("\\", "")))
    return text

class ClaudeMetaTranslator:
    def __init__(self, key: str, model: str = "claude-3-opus-20240229"):
        """
        Initializes the translator with API key and model.

        Parameters:
        key (str): The API key for accessing the translation service.
        model (str): The model name to be used for translation.
        """
        self.client = Client(api_key=key)
        self.model = model
        self.tokenizer = GlobalRelativeTokenizer()

    def translate(self, input_prompt: str, pairs: str):
        """
        Translates the input prompt based on the provided translation pairs.

        Parameters:
        input_prompt (str): The source sequence to be translated.
        pairs (str): The translation pairs for reference.

        Returns:
        str: The translated output.
        """
        pairs, input_prompt = self.preprocess(pairs, input_prompt)
        system = SYSTEM_PROMPT
        message = MESSAGE_PROMPT.format(pairs=pairs, input_prompt=input_prompt)
        output = self.send_message(message, system)
        return output

    def send_message(self, message, system):
        """
        Sends a message to the translation service and retrieves the output.

        Parameters:
        message (str): The message to be sent.
        system (str): The system prompt.

        Returns:
        str: The output from the translation service.
        """
        chat_completion = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        output = chat_completion.content[0].text
        print(output)
        return output
    
    def preprocess(self, pairs, input_prompt):
        """
        Preprocesses the translation pairs and input prompt for translation.

        Parameters:
        pairs (str): The translation pairs.
        input_prompt (str): The source sequence to be translated.

        Returns:
        tuple: Processed pairs and input prompt.
        """
        lines = pairs.split("\n")
        new_pairs = []
        for line in lines:
            if line.startswith("source:"):
                new_pairs.append(line)
            elif line.startswith("target:"):
                target_text = filter_text(line[len("target: "):])
                tokenized_target = 'target: ' + ' '.join(map(str, self.tokenizer.tokenize(target_text)))
                new_pairs.append(tokenized_target)
        new_pairs = "\n".join(new_pairs)
        return new_pairs, input_prompt

def extract_numbers(text):
    """
    Extracts numerical tokens from the given text.

    Parameters:
    text (str): The text containing numerical tokens.

    Returns:
    list: A list of numerical tokens extracted from the text.
    """
    import re
    pattern = r'target:\s*(\d+(?:\s+\d+)*)'
    matches = re.findall(pattern, text)
    if matches:
        last_match = matches[-1]
        return last_match.split()
    return []

# Constants for prompts
SYSTEM_PROMPT = """You Claude, skilled at finding general sequence-to-sequence patterns in data. In this case, there are 'source' and 'target' sequences.
I'll provide you with some translation pairs and your task is to generate the 'target' sequence corresponding to the given 'source' sequence.
These are language translation pairings, but each unique word in the target has been assigned a number. Thus, the overall patterns should be linguistic in nature.
Take a deep breath, clear your mind, and think this through step by step, outlining your evidence for each decision.
"""

MESSAGE_PROMPT = """Based on the provided translation pairs and your own intelligence, generate the 'target' sequence corresponding to the given 'source' sequence.

1. First, create a 'target word bank' of target numbers that will probably occur given the source sequence. Write out the probable target numbers, and say why you think each is probable. Essentially create a mapping of source phrases to target sequences.
2. Formulate grammatical rules that seem apparent in the target texts, and are relevant to the sequence you will translate.
3. Next, focus on arranging these target numbers in the correct order to form a coherent translation, also say why.
4. Remember: don't copy any of the examples verbatim, just use the underlying linguistic rules that seem apparent.
5. Avoid repetition.
Here are the translation pairs:
6. Only predict tokens that have already been used.
Here are the pairs:
{pairs}

Source Sequence:
{input_prompt}

Remember, the last part of your response should follow this format 'target: <some output>'
Think through each step, write out your thought process, be concise, and use the examples provided.
Do your best, don't stress about it, use the format. My friend doesn't think you can do this well, but I am fully confident you can prove him wrong.
"""

# Example usage:
claude_key = key here
translator = ClaudeMetaTranslator(claude_key)

pairs = """
source: zu der Stätte des Altars, den er zuvor daselbst gemacht hatte. Und Abram rief daselbst den Namen Jehovas an.
target: He went to the place where the altar was that he had built previously. Here he called on the name of Yahweh.
source: Und als Asa diese Worte und die Weissagung Odeds, des Propheten, hörte, faßte er Mut; und er schaffte die Greuel weg aus dem ganzen Lande Juda und Benjamin und aus den Städten, die er vom Gebirge Ephraim eingenommen hatte, und er erneuerte den Altar Jehovas, der vor der Halle Jehovas stand.
target: When Asa heard these words, the prophecy of Oded the prophet, he took courage and drove away the disgusting things from all the land of Judah and Benjamin, and from the cities that he had captured from the hill country of Ephraim, and he rebuilt Yahweh's altar, which was in front of the portico of Yahweh's house.
source: Und Salomo baute das Haus und vollendete es. -
target: So Solomon built the house and finished it.
source: Und hernach baute er die äußere Mauer der Stadt Davids, westlich gegen den Gihon hin, im Tale, und bis zum Eingang des Fischtores, und umgab den Ophel mit einer Mauer und machte sie sehr hoch. Und er legte Kriegsoberste in alle festen Städte in Juda.
target: After this, Manasseh built an outer wall to the city of David, on the west side of Gihon, in the valley, to the entrance at the Fish Gate. He surrounded the hill of Ophel with it and raised the wall up to a very great height. He put courageous commanders in all the fortified cities of Judah.
source: Und er kaufte den Berg Samaria von Schemer um zwei Talente Silber; und er bebaute den Berg und gab der Stadt, die er gebaut hatte, den Namen Samaria, nach dem Namen Schemers, des Herrn des Berges.
target: He bought the hill of Samaria from Shemer for two talents of silver. He built a city on the hill and called the name of the city Samaria, after the name of Shemer, the past owner of the hill.
source: Und er zog aus und stritt wider sie Philister, und riß nieder die Mauer von Gath und die Mauer von Jabne und die Mauer von Asdod; und er baute Städte um Asdod her und unter den Philistern.
target: Uzziah went out and fought against the Philistines. He broke down the city walls of Gath, Jabneh, and Ashdod; he built cities in the country of Ashdod and among the Philistines.
source: Und er brach auf von dannen nach dem Gebirge ostwärts von Bethel und schlug sein Zelt auf, Bethel gegen Westen und Ai gegen Osten; und er baute daselbst Jehova einen Altar und rief den Namen Jehovas an.
target: From there he moved to the hill country to the east of Bethel, where he pitched his tent, with Bethel to the west and Ai to the east. There he built an altar to Yahweh and called on the name of Yahweh.
source: Und Ussija baute Türme in Jerusalem auf dem Ecktor und auf dem Taltor und auf dem Winkel, und befestigte sie.
target: In addition, Uzziah built towers in Jerusalem at the Corner Gate, at the Valley Gate, and at the turning of the wall, and fortified them.
source: Und der Mann zog aus der Stadt, aus Bethlehem-Juda, um sich aufzuhalten, wo er es treffen würde. Und indem er seines Weges zog, kam er in das Gebirge Ephraim bis zum Hause Michas.
target: The man left Bethlehem in Judah to go and find a place to live. As he journeyed, he came to Micah's house in the hill country of Ephraim.
source: wendet euch und brechet auf und ziehet nach dem Gebirge der Amoriter und zu allen ihren Anwohnern in der Ebene, auf dem Gebirge und in der Niederung und im Süden und am Ufer des Meeres, in das Land der Kanaaniter und zum Libanon, bis zu dem großen Strome, dem Strome Phrat.
target: Turn and take your journey, and go to the hill country of the Amorites and to all the places near there in the plain of the Jordan River valley, in the hill country, in the lowland, in the Negev, and by the seashore—the land of the Canaanites, and in Lebanon as far as the great river, the Euphrates.
source: Und das Übrige der ganzen Geschichte Asas und alle seine Macht, und alles, was er getan, und die Städte, die er gebaut hat, ist das nicht geschrieben in dem Buche der Chronika der Könige von Juda? Doch zur Zeit seines Alters erkrankte er an seinen Füßen.
target: As for the other matters of Asa, all his might, all that he did, and the cities he built, are they not written in the book of the events of the kings of Judah? But during his old age he was diseased in his feet.
source: Auch er machte Höhen auf den Bergen Judas, und er verleitete die Bewohner von Jerusalem, Hurerei zu treiben, und verführte Juda dazu.
target: In addition, Jehoram had also built high places in the mountains of Judah and he made the inhabitants of Jerusalem to live like prostitutes, and he led Judah astray.
source: und Baalath und alle Vorratsstädte, die Salomo hatte; und alle Wagenstädte und die Reiterstädte; und alles, was Salomo Lust hatte zu bauen in Jerusalem und auf dem Libanon und im ganzen Lande seiner Herrschaft.
target: He built Baalath and all the store cities that he possessed, and all the cities for his chariots and the cities for his horsemen, and whatever he wished to build for his pleasure in Jerusalem, in Lebanon, and in all the lands under his rule.
source: Und er baute Türme in der Wüste und grub viele Zisternen; denn er hatte viel Vieh, sowohl in der Niederung als auch in der Ebene, und Ackerleute und Weingärtner im Gebirge und im Fruchtgefilde; denn er liebte den Ackerbau.
target: He built watchtowers in the wilderness and dug many cisterns, for he had much cattle, in the lowlands as well as in the plains. He had farmers and vine growers in the hill country and in the fruitful fields, for he loved farming.
source: Man wird Felder um Geld kaufen und Kaufbriefe schreiben und sie versiegeln und Zeugen nehmen im Lande Benjamin und in den Umgebungen von Jerusalem und in den Städten Judas, sowohl in den Städten des Gebirges als auch in den Städten der Niederung und in den Städten des Südens. Denn ich werde ihre Gefangenschaft wenden, spricht Jehova.
target: They will buy fields with silver and write in sealed scrolls. They will assemble witnesses in the land of Benjamin, all around Jerusalem and the cities of Judah, in the cities in the hill country and in the lowlands, and in the cities of the Negev. For I will bring back their fortunes—this is Yahweh's declaration.'”
source: Und Saul baute Jehova einen Altar; mit diesem fing er an, Jehova einen Altar zu bauen.
target: Saul built an altar to Yahweh, which was the first altar that he built to Yahweh.
source: Und Josaphat wurde immerfort größer, bis er überaus groß war. Und er baute in Juda Burgen und Vorratsstädte;
target: Jehoshaphat became very powerful. He built fortresses and store cities in Judah.
source: Denn jedes Haus wird von jemand bereitet; der aber alles bereitet hat, ist Gott.
target: For every house is built by someone, but the one who built everything is God.
source: Und er baute feste Städte in Juda; denn das Land hatte Ruhe, und es war kein Krieg wider ihn in jenen Jahren, denn Jehova hatte ihm Ruhe geschafft.
target: He built fortified cities in Judah, for the land was quiet, and he had no war in those years, because Yahweh had given him peace.
source: In den Städten des Gebirges, in den Städten der Niederung und in den Städten des Südens, und im Lande Benjamin und in den Umgebungen von Jerusalem und in den Städten Judas werden wiederum die Herden unter den Händen des Zählers vorüberziehen, spricht Jehova.
target: In the cities in the hill country, the lowlands, and the Negev, in the land of Benjamin and all around Jerusalem, and in the cities of Judah, the flocks will again pass under the hands of the ones counting them,' says Yahweh.
source: Und er faßte Mut und baute die ganze Mauer, wo sie eingerissen war, und führte sie auf bis an die Türme, und die andere Mauer außerhalb, und befestigte das Millo der Stadt Davids; und er verfertigte Waffen in Menge und Schilde.
target: Hezekiah took courage and built up all the wall that was broken down. He built the towers higher, and also the other wall outside. He also strengthened the Millo in the city of David, and he made large amounts of weapons and shields.
source: Und sie heiligten Kedes in Galiläa, im Gebirge Naphtali, und Sichem im Gebirge Ephraim, und Kirjath-Arba, das ist Hebron, im Gebirge Juda.
target: So the Israelites selected Kedesh in Galilee in the hill country of Naphtali, Shechem in the hill country of Ephraim, and Kiriath Arba (the same as Hebron) in the hill country of Judah.
source: Und er errichtete dem Baal einen Altar im Hause des Baal, das er zu Samaria gebaut hatte;
target: He built an altar for Baal in the house of Baal, which he had built in Samaria.
source: Und das Übrige der Geschichte Ahabs und alles, was er getan und das elfenbeinerne Haus, das er gebaut, und alle Städte, die er gebaut hat, ist das nicht geschrieben in dem Buche der Chronika der Könige von Israel?
target: As for the other matters concerning Ahab, all that he did, the ivory house that he built, and all the cities that he built, are they not written in the book of the events of the kings of Israel?
source: Und Rehabeam wohnte in Jerusalem; und er baute Städte zu Festungen in Juda.
target: Rehoboam lived in Jerusalem and built cities in Judah for defense.
source: Und Josua kam in selbiger Zeit und rottete die Enakim aus von dem Gebirge, von Hebron, von Debir, von Anab und von dem ganzen Gebirge Juda und von dem ganzen Gebirge Israel: mit ihren Städten verbannte sie Josua.
target: Then Joshua came at that time and he destroyed the Anakim. He did this in the hill country, at Hebron, Debir, Anab, and in all the hill country of Judah, and in all the hill country of Israel. Joshua completely destroyed them and their cities.
source: Und er baute Tadmor in der Wüste und alle Vorratsstädte, die er in Hamath baute.
target: He built Tadmor in the wilderness, and all the store cities, which he built in Hamath.
source: Er baute das obere Tor des Hauses Jehovas; auch an der Mauer des Ophel baute er viel.
target: He built the upper gate of the house of Yahweh, and on the hill of Ophel he built much.
source: Und Jerobeam baute Sichem im Gebirge Ephraim und wohnte darin; und er zog von dannen aus und baute Pnuel.
target: Then Jeroboam built Shechem in the hill country of Ephraim, and lived there. He went out from there and built Peniel.
""".lower()

input_prompt = "source: Und er baute Städte im Gebirge Juda; und in den Wäldern baute er Burgen und Türme.".lower()
output = translator.translate(input_prompt, pairs)

# Extract numerical representations from the output
target_numbers = extract_numbers(output)
print("numbers: ", target_numbers)

# Translate these numbers back into text using value_to_token
translated_tokens = [translator.tokenizer.value_to_token.get(int(num), 'unk') for num in target_numbers]
translated_text = ' '.join(translated_tokens)
print("Translated Text:", translated_text)
print("Baseline: ", "target: Moreover he built cities in the hill country of Judah, and in the forests he built castles and towers.")
