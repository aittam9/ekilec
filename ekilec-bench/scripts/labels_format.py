#different labels formats to test
LABELS = {
        "letter_columns": ["A:", "B:", "C:", "D:"],
        "letter_parenthesis": ["A)", "B)", "C)", "D)"],
        "digit_dot" : ["1.", "2.", "3.", "4."],
        "digit_partenthesis": ["1)", "2)", "3)", "4)"]
        }

PROMPT = """###ISTRUZIONI###
Sei un esperto in materia di giurisprudenza italiana e diritto privato italiano.
Di seguito ti verrà sottoposto un quesito e delle possibili risposte.
Il tuo compito è selezionare la risposta corretta in base al quesito.
Restituisci in output solo il label corrispondente alla risposta corretta selezionando tra [A, B, C, D] e null'altro.
**Non devi** assolutamente aggiungere altro testo o spiegazioni alla risposta, ma solo il label della risposta corretta.

###QUESITO###
{quesito_}

###RISPOSTE POSSIBILI###
{risposte_possibili}

Risposta:
"""


MODEL_NAME_2_MODEL_ID = { 
        "gpt-5" :"openai/gpt-5.1-chat",
        "gemini-1.5-pro" :"google/gemini-2.5-pro",
        "claude-sonnet" : "anthropic/claude-sonnet-4.5",
        "deepseek-r1" : "deepseek/deepseek-r1-0528",
        "gemini-3-flash" : "google/gemini-3-flash-preview",
        "llama-3-405b" : "meta-llama/llama-3.1-405b-instruct",
        "kimi-k2": "moonshotai/kimi-k2",
        "llama-3.1-405b":"meta-llama/llama-3.1-405b-instruct",
        "gemini-flash-2.5" : "google/gemini-2.5-flash",
        "gpt-4o": "openai/gpt-4o",
        "deepseek-v3": "deepseek/deepseek-chat-v3-0324"
                        }