def CaseDecorate(case):
    def wrapper(self):
        print(f"{self.nome}, banana")
    
    def setValore(self, valore):
        self.nome = valore - 2

    case.getNome = wrapper
    case.setValue = setValore

    return case 

@CaseDecorate
class Case():

    def __init__(self, nome):
        self.nome = nome

    def getValue(self):
        print(f"{self.nome}")
        return self.nome

    def setValue(self, value):
        self.nome = value


if __name__ == "__main__":
    casetta = Case(1)
    casetta.getNome()
    casetta.setValue(34)
    casetta.getNome()
