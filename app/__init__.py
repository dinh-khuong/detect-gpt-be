from detectgptmodel.fastgpt import GPTChecker
import sys

def get_model():
    # Only load if we aren't running a migration/test command
    if 'runserver' in sys.argv :
        return GPTChecker(model="roberta-large-mnli")

    return None
    
gptDetectModel = get_model()
