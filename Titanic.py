import csv



data = []
testData = []

def getSurvivedAge(ageMin,ageMax):
    survived = 0
    for entry in data:
        try:
            if (float(entry['age']) >= ageMin and float(entry['age']) <= ageMax) and entry['survived'] == 1:
                survived = survived + 1
        except ValueError:
            pass
    return  survived


def getSurvivedGender(gender):
    survived = 0
    for entry in data:
        if entry['sex'] == gender and entry['survived'] == '1':
            survived = survived + 1
    return  survived

def getQuantGender(gender):
    quant = 0
    for entry in data:
        if entry['sex'] == gender:
            quant = quant + 1
    return  quant

def getQuantSurvived():
    survived = 0
    for entry in data:
        if  entry['survived'] == '1':
            survived = survived + 1
    return  survived

def addData(row):
    age = 0.0
    data.append({'number':row[0],'survived':row[1],'sex':row[3],'age':row[4],'sibsp':row[5],'parch':row[6],'fare':row[8],'embarked':row[10]})


def readData():
    with open('train.csv','rb') as csvfile:
            spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
            for row in spamreader:
                    addData(row)


def readTestData():
    with open('test.csv','rb') as csvfile:
            spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
            for row in spamreader:
                if row[0] != 'PassengerId':
                    testData.append({'number':row[0],'survived':'0','sex':row[2],'age':row[3],'sibsp':row[4],'parch':row[5],'fare':row[7],'embarked':row[9]})

def getOracle():
    for entry in testData:
        if entry['sex'] == 'female':
            entry['survived'] = '1'

def writeToCsv():
    with open('out.csv','w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for entry in testData:
            writer.writerow([entry['number'],entry['survived']])

            


def main():
    readData()
    print float(getSurvivedGender('female')) / float(getQuantSurvived())
    print float(getSurvivedGender('female')) / float(getQuantGender('female'))

    print float(getSurvivedGender('male')) / float(getQuantSurvived())
    print float(getSurvivedGender('male')) / float(getQuantGender('male'))
    readTestData()
    getOracle()
    writeToCsv()



main()
