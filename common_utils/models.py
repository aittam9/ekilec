OPEN_ITA_MODELS = {
    "llamantino-3-anita":"swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA", 
    "minerva-7b-inst": "sapienzanlp/Minerva-7B-instruct-v1.0", 
#     "minerva-3b-v1.0": "sapienzanlp/Minerva-3B-v1.0",
    # # base is openchat
#     "cerbero-7b" : "galatolo/cerbero-7b", # base is mistral
#     "llama3-it-pa": "swap-uniba/llama3-it-pa-300k-adapter",
#     "dante-llm-7b" :"rstless-research/DanteLLM-7B-Instruct-Italian-v0.1",
    "llama-3-8b-instruct" : "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b-instruct" : "mistralai/Mistral-7B-Instruct-v0.3"
                    }

API_MODELS = { 
        "gpt-5" :"openai/gpt-5.1-chat",
        "claude-sonnet" : "anthropic/claude-sonnet-4.5",
        "gemini-3-flash" : "google/gemini-3-flash-preview",
        "llama-3.1-405b":"meta-llama/llama-3.1-405b-instruct",
        "gemini-flash-2.5" : "google/gemini-2.5-flash",
        "gpt-4o": "openai/gpt-4o",
        "deepseek-v3": "deepseek/deepseek-chat-v3-0324"
            }




PROMPT = """# ISTRUZIONI
Sei un esperto in materia di giurisprudenza italiana e diritto privato italiano.
Di seguito ti verrà sottoposto un quesito e delle possibili risposte.
Il tuo compito è selezionare la risposta corretta in base al quesito.
Restituisci in output solo il label corrispondente alla risposta corretta selezionando tra [A, B, C, D] e null'altro.
**Non devi** assolutamente aggiungere altro testo o spiegazioni alla risposta, ma solo il label della risposta corretta.

# QUESITO
{quesito_}

# RISPOSTE POSSIBILI
{risposte_possibili}

Risposta:
"""

PROMPT_DEFINITIONS = """



"""


GEN_PROMPTS = {"3_SHOTS": """Di seguito ti verrà mostrato un concetto di diritto italiano. Il concetto può essere un termine, un'espressione tecnica o una locuzione che deve essere definita in termini giuridici.
Fornisci una breve ma esaustiva definizione del concetto. La definizione deve essere compresa tra 1 e 4 frasi come negli esempi forniti.
Fornisci esclusivamente la definizione e non aggiungere altro testo.

ESEMPIO 1

Concetto: Indirizzo della produzione
Definizione: Locuzione che indica il complesso delle scelte sia economiche che tecniche, operate dall'imprenditore finché svolge la sua attività produttiva. Gli indirizzi della produzione dei diversi imprenditori che operano sul territorio nazione sono soggetti a controlli da parte dello Stato.


Esempio 2
Concetto: Credito
Definizione: Si definisce quale situazione giuridica soggettiva attiva del rapporto obbligatorio, ossia il diritto del creditore all'esecuzione della prestazione dovutagli dal debitore per effetto del debito da questi contratto.


Esempio 3
Concetto: Consolidazione
Definizione: Termine che si utilizza per indicare il fenomeno della riunione in capo ad un unico soggetto della titolarità di un diritto reale (proprietà o altro diritto reale limitato, come l'usufrutto. A seguito di tale riunione, il diritto si estingue.

Esempio 4
Concetto: Custode dell'arrestato o detenuto
Definizione: Con il termine "custode dell'arrestato o del detenuto" si fa riferimento a quella figura professionale, in genere una guardia penitenziaria, che è preposta al controllo della persona arrestata o detenuta.
           
Concetto: {term}
Definizione: """,



"0_SHOTS_v1": """Di seguito ti verrà mostrato un concetto di diritto italiano. Il concetto può essere un termine, un'espressione tecnica o una locuzione che deve essere definita in termini giuridici.
Fornisci una breve ma esaustiva definizione del concetto. La definizione deve essere compresa tra 1 e 4 frasi.
Fornisci esclusivamente la definizione e non aggiungere altro testo.""",

"0_SHOTS_v2": "Che cosa significa {term} in diritto italiano? Fornisci una breve ma esaustiva definizione del concetto. Se possibile la definizione deve consistere in una frase. Fornisci esclusivamente la definizione e non aggiungere altro testo."
}




LABELS = {
        "letter_columns": ["A:", "B:", "C:", "D:"],
        "letter_parenthesis": ["A)", "B)", "C)", "D)"],
        "digit_dot" : ["1.", "2.", "3.", "4."],
        "digit_partenthesis": ["1)", "2)", "3)", "4)"]
        }



LABELS = {
        "letter_columns": ["A:", "B:", "C:", "D:"],
        "letter_parenthesis": ["A)", "B)", "C)", "D)"],
        "digit_dot" : ["1.", "2.", "3.", "4."],
        "digit_partenthesis": ["1)", "2)", "3)", "4)"]
        }
